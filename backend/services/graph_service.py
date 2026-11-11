"""Attach actual connector strokes; never invent reading-order connections."""
from __future__ import annotations
from collections import deque, defaultdict
from dataclasses import dataclass
import math
import re
import cv2
import numpy as np

from backend.models import Arrow, Edge, FlowGraph, Node, PipelineContext
from backend.services.connector_geometry import (connector_stroke_width, endpoint_has_arrowhead, ink_mask,
                                                 stroke_exits)
from backend.notation import terminal_role, normalize_label
from backend.services.vision_service import erase_enclosed_components

def _next_edge_id(graph: FlowGraph) -> str:
    """Return a stable unused edge ID after inferred edges are appended."""
    used = {edge.id for edge in graph.edges}
    index = len(graph.edges) + 1
    while f"e{index}" in used:
        index += 1
    return f"e{index}"


@dataclass
class Port:
    node: Node
    point: tuple[int, int]
    head: bool
    distance: float = 0.0  # closest approach of the stroke to the node polygon


def arrowhead_evidence(points: np.ndarray, point: tuple[int, int]) -> bool:
    """Detect a flared V/triangle near a tip followed by a narrower shaft."""
    delta = points.astype(float) - np.asarray(point)
    radius = np.linalg.norm(delta, axis=1)
    for span in (12, 18, 28, 40, 56, 80, 120, 160):
        shaft = delta[(radius >= span * .55) & (radius <= span * 1.2)]
        if len(shaft) < 4:
            continue
        outward = np.median(shaft, axis=0)
        norm = np.linalg.norm(outward)
        if norm < 8:
            continue
        direction = outward / norm
        perpendicular = np.array([-direction[1], direction[0]])
        longitudinal, transverse = delta @ direction, delta @ perpendicular
        widths, flares = [], []
        step = max(4, span // 14)
        for start in range(0, span, step):
            selected = transverse[(longitudinal >= start) & (longitudinal < start + step) & (radius < span * 1.4)]
            widths.append(float(np.ptp(selected)) + 1 if len(selected) else 0)
            flares.append(bool(len(selected) and selected.min() < -4 and selected.max() > 4))
        boundary = max(2, int(len(widths) * .63))
        early = max((widths[i] for i in range(1, boundary) if flares[i]), default=0)
        reference = [width for width in widths[boundary + 1:] if width > 0]
        if reference:
            narrow = float(np.median(reference))
            if early > max(9, narrow * 1.8 + 3) and narrow < 18:
                return True
    return False


def contacts(points: np.ndarray, nodes: list[Node], shape: tuple[int, int], distance: int, *,
             stroke_width: float | None = None, head_mask: np.ndarray | None = None,
             touch_distance: int | None = None) -> list[Port]:
    """Find where a connector component meets node boundaries.

    With the default arguments this is the original proximity test. When
    ``touch_distance`` is given, a contact must also be a genuine stroke
    *termination*: either the stroke comes within ``touch_distance`` of the
    polygon, or it leaves the contact band through a single exit. A line that
    merely runs beside a node (two exits) is no longer attached to it. When
    ``stroke_width`` is given, arrowheads are detected with the localised,
    scale-aware profile test, using ``head_mask`` to recover arrowhead pixels
    that node cleanup erased.
    """
    ports = []
    height, width = shape
    sw = float(stroke_width or 3)
    ring_width = int(max(6, round(2 * sw))) if touch_distance is not None else 0
    band_width = int(round(3 * sw + 6)) if touch_distance is not None else 0
    reach = distance + band_width + ring_width
    for node in nodes:
        # Distance is measured from the actual polygon, not its enclosing rectangle.
        bx, by, bw, bh = node.bbox.to_list()
        nearby = points[(points[:, 0] >= bx - reach) & (points[:, 0] <= bx + bw + reach) &
                        (points[:, 1] >= by - reach) & (points[:, 1] <= by + bh + reach)]
        if len(nearby) == 0:
            continue
        x0, y0 = max(0, bx - reach - 3), max(0, by - reach - 3)
        x1, y1 = min(width, bx + bw + reach + 3), min(height, by + bh + reach + 3)
        exterior = np.full((y1 - y0, x1 - x0), 255, np.uint8)
        polygon = np.asarray(node.polygon or [(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)], np.int32)
        cv2.fillPoly(exterior, [polygon - np.array([x0, y0])], 0)
        distances = cv2.distanceTransform(exterior, cv2.DIST_L2, 5)
        values = distances[nearby[:, 1] - y0, nearby[:, 0] - x0]
        near = nearby[values <= distance]
        if len(near) == 0:
            continue
        band = np.zeros_like(exterior)
        band[near[:, 1] - y0, near[:, 0] - x0] = 255
        count, labels = cv2.connectedComponents(band, connectivity=8)
        for component in range(1, count):
            ys, xs = np.where(labels == component)
            if len(xs) < 3:
                continue
            d = distances[ys, xs]
            if float(d.min()) > distance:
                continue
            if touch_distance is not None and float(d.min()) > touch_distance:
                if _passes_beside(nearby - np.array([x0, y0]), values, float(d.min()),
                                  (xs[d <= d.min() + 1.5], ys[d <= d.min() + 1.5]),
                                  exterior.shape, band_width, ring_width):
                    continue  # a line running beside the node, not ending at it
            closest = np.column_stack((xs[d <= d.min() + 1.5] + x0, ys[d <= d.min() + 1.5] + y0))
            center = closest.mean(axis=0)
            point_array = closest[np.argmin(np.sum((closest - center) ** 2, axis=1))]
            point = int(point_array[0]), int(point_array[1])
            head = (endpoint_has_arrowhead(points, point, stroke_width, head_mask)
                    if stroke_width is not None else arrowhead_evidence(points, point))
            ports.append(Port(node, point, head, float(d.min())))
    return sorted(ports, key=lambda port: (port.point[1], port.point[0], port.node.id))


def _passes_beside(local_points: np.ndarray, values: np.ndarray, closest_distance: float,
                   closest: tuple[np.ndarray, np.ndarray], shape: tuple[int, int],
                   band_width: int, ring_width: int) -> bool:
    """True when the stroke near its closest approach leaves in two or more places.

    The band is anchored at the stroke's own closest distance to the node, so a
    line running parallel to a side is cut across (two exits) rather than along
    its length. A connector that ends near the node leaves through one exit.
    """
    inner_cut = closest_distance + band_width
    inner = np.zeros(shape, np.uint8)
    ring = np.zeros(shape, np.uint8)
    inside = local_points[values <= inner_cut]
    shell = local_points[(values > inner_cut) & (values <= inner_cut + ring_width)]
    inner[inside[:, 1], inside[:, 0]] = 255
    ring[shell[:, 1], shell[:, 0]] = 255
    # Only shell pieces that continue through to its outer edge are exits;
    # arrowhead barbs and glyph specks poke into the shell and stop.
    outer = np.zeros(shape, np.uint8)
    far = local_points[(values > inner_cut + ring_width / 2) & (values <= inner_cut + ring_width)]
    outer[far[:, 1], far[:, 0]] = 255
    count, labels = cv2.connectedComponents(inner, connectivity=8)
    anchor = labels[closest[1], closest[0]]
    anchor = anchor[anchor != 0]
    if anchor.size == 0:
        return False
    own = np.where(np.isin(labels, np.unique(anchor)), 255, 0).astype(np.uint8)
    return stroke_exits(own, ring, outer) >= 2


def stroke_path(points: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    width, height = int(x1 - x0 + 1), int(y1 - y0 + 1)
    mask = np.zeros((height, width), np.uint8)
    mask[points[:, 1] - y0, points[:, 0] - x0] = 1
    sx, sy = int(start[0] - x0), int(start[1] - y0)
    tx, ty = int(end[0] - x0), int(end[1] - y0)
    begin, finish = sy * width + sx, ty * width + tx
    queue = deque([begin])
    parent = np.full(width * height, -1, np.int32)
    parent[begin] = begin
    found = False
    while queue:
        current = queue.popleft()
        if current == finish:
            found = True
            break
        cy, cx = divmod(current, width)
        for dy, dx in ((-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < width and 0 <= ny < height:
                index = ny * width + nx
                if mask[ny, nx] and parent[index] == -1:
                    parent[index] = current
                    queue.append(index)
    if not found:
        return []
    route = []
    current = finish
    while current != begin:
        cy, cx = divmod(current, width)
        route.append((int(cx + x0), int(cy + y0)))
        current = int(parent[current])
    route.append(start)
    route.reverse()
    simplified = cv2.approxPolyDP(np.asarray(route, np.int32).reshape(-1, 1, 2), 3, False)
    return [(int(x), int(y)) for x, y in simplified.reshape(-1, 2)]


def distance_to_path(point: tuple[float, float], path: list[tuple[int, int]]) -> float:
    p = np.asarray(point)
    distances = []
    for a, b in zip(path, path[1:]):
        a, b = np.asarray(a), np.asarray(b)
        vector = b - a
        t = np.clip(np.dot(p - a, vector) / max(1, np.dot(vector, vector)), 0, 1)
        distances.append(float(np.linalg.norm(p - (a + t * vector))))
    return min(distances, default=math.inf)


def route_has_visible_stroke(
    mask: np.ndarray,
    route: list[tuple[int, int]],
    *,
    min_coverage: float = 0.35,
    min_segments: int = 1,
    radius: int = 3,
) -> bool:
    """Check that a proposed connector follows visible ink, not just layout.

    Samples each meaningful route segment while tolerating small raster gaps,
    antialiasing, and text interruptions. Every checked segment must have
    sufficient nearby stroke evidence; a single unrelated pixel is not enough.
    """
    coverages = []
    for first, second in zip(route, route[1:]):
        length = math.dist(first, second)
        if length < 18:
            continue
        steps = max(2, int(math.ceil(length / 6)))
        hits = 0
        checked = 0
        for index in range(1, steps):
            ratio = index / steps
            px = int(round(first[0] + (second[0] - first[0]) * ratio))
            py = int(round(first[1] + (second[1] - first[1]) * ratio))
            x0, x1 = max(0, px - radius), min(mask.shape[1], px + radius + 1)
            y0, y1 = max(0, py - radius), min(mask.shape[0], py + radius + 1)
            checked += 1
            hits += int(bool(np.any(mask[y0:y1, x0:x1])))
        coverages.append(hits / max(1, checked))
    return len(coverages) >= min_segments and min(coverages, default=0.0) >= min_coverage


def bridge_aligned_connector_gaps(mask: np.ndarray, nodes: list[Node]) -> np.ndarray:
    """Reconnect line fragments separated by erased branch-label text.

    Only thin, strongly aligned vertical/horizontal fragments are bridged, and
    only across a bounded gap with no intervening node. This avoids global
    morphological closing, which can join unrelated arrows and node borders.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        if area < 20 or max(width, height) < 18:
            continue
        # Admit short connector remnants only when they remain shaft-like;
        # this captures short post-label fragments without treating ordinary
        # OCR glyph clusters as bridge candidates.
        if max(width, height) < 35 and min(width, height) > 20:
            continue
        components.append((index, x, y, width, height, area))
    result = mask.copy()
    max_gap = max(45, int(round(mask.shape[1] * .05)))
    x_tolerance = max(24, int(round(mask.shape[1] * .015)))

    def crosses_node(a, b):
        ax, ay = a
        bx, by = b
        for node in nodes:
            nx, ny, nw, nh = node.bbox.to_list()
            # Ignore a node only if an endpoint lies within its expanded bounds;
            # otherwise any crossing through a shape makes the bridge unsafe.
            if min(ax, bx) < nx + nw and max(ax, bx) > nx and min(ay, by) < ny + nh and max(ay, by) > ny:
                return True
        return False

    max_endpoint_span = 20
    for i, a in enumerate(components):
        _, ax, ay, aw, ah, _ = a
        for b in components[i + 1:]:
            _, bx, by, bw, bh, _ = b

            # Vertical continuation. Use the narrow pixel runs at the facing
            # component boundaries; an elbow connector's overall bbox may be
            # wide/tall and must not disqualify its narrow endpoint.
            if by >= ay + ah:
                gap = by - (ay + ah)
                if 8 < gap <= max_gap:
                    upper_xs = np.where(labels[ay + ah - 1, :] == a[0])[0]
                    lower_xs = np.where(labels[by, :] == b[0])[0]
                    if (0 < len(upper_xs) <= max_endpoint_span and
                            0 < len(lower_xs) <= max_endpoint_span):
                        upper_x = float(np.median(upper_xs))
                        lower_x = float(np.median(lower_xs))
                        if abs(upper_x - lower_x) <= x_tolerance:
                            start = (int(round(upper_x)), ay + ah - 1)
                            end = (int(round(lower_x)), by)
                            if not crosses_node(start, end):
                                thickness = max(3, min(9, int(round(
                                    (len(upper_xs) + len(lower_xs)) / 2
                                ))))
                                cv2.line(result, start, end, 255, thickness)

            # Horizontal continuation, with the same endpoint-based safeguards.
            if bx >= ax + aw:
                gap = bx - (ax + aw)
                if 8 < gap <= max_gap:
                    left_ys = np.where(labels[:, ax + aw - 1] == a[0])[0]
                    right_ys = np.where(labels[:, bx] == b[0])[0]
                    # An arrowhead widens at its tip, so its first image
                    # column can contain more pixels than a normal shaft.
                    # Permit one broad endpoint only when the opposite endpoint
                    # is a thin stroke and their centerlines remain aligned.
                    left_thin = 0 < len(left_ys) <= max_endpoint_span
                    right_thin = 0 < len(right_ys) <= max_endpoint_span
                    left_head = max_endpoint_span < len(left_ys) <= 50
                    right_head = max_endpoint_span < len(right_ys) <= 50
                    if ((left_thin and (right_thin or right_head)) or
                            (right_thin and left_head)):
                        left_y = float(np.median(left_ys))
                        right_y = float(np.median(right_ys))
                        if abs(left_y - right_y) <= max(12, x_tolerance * 0.5):
                            start = (ax + aw - 1, int(round(left_y)))
                            end = (bx, int(round(right_y)))
                            if not crosses_node(start, end):
                                thickness = max(3, min(9, int(round(
                                    min(len(left_ys), len(right_ys))
                                ))))
                                cv2.line(result, start, end, 255, thickness)
            elif ax >= bx + bw:
                # Connected-component IDs are assigned in scan order, not
                # left-to-right order. A broad arrowhead can begin higher than
                # its shaft and therefore receive the earlier component ID.
                gap = ax - (bx + bw)
                if 8 < gap <= max_gap:
                    left_ys = np.where(labels[:, bx + bw - 1] == b[0])[0]
                    right_ys = np.where(labels[:, ax] == a[0])[0]
                    left_thin = 0 < len(left_ys) <= max_endpoint_span
                    right_thin = 0 < len(right_ys) <= max_endpoint_span
                    left_head = max_endpoint_span < len(left_ys) <= 50
                    right_head = max_endpoint_span < len(right_ys) <= 50
                    if ((left_thin and (right_thin or right_head)) or
                            (right_thin and left_head)):
                        left_y = float(np.median(left_ys))
                        right_y = float(np.median(right_ys))
                        if abs(left_y - right_y) <= max(12, x_tolerance * 0.5):
                            start = (bx + bw - 1, int(round(left_y)))
                            end = (ax, int(round(right_y)))
                            if not crosses_node(start, end):
                                thickness = max(3, min(9, int(round(
                                    min(len(left_ys), len(right_ys))
                                ))))
                                cv2.line(result, start, end, 255, thickness)
    return result


def node_contains(node: Node, point: tuple[int, int], margin: float = 0.0) -> bool:
    """Whether ``point`` lies inside, or within ``margin`` of, a node polygon."""
    bx, by, bw, bh = node.bbox.to_list()
    polygon = np.asarray(node.polygon or [(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)],
                         np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), True) >= -margin


def bridge_through_labels(mask: np.ndarray, words, nodes: list[Node], stroke_width: int) -> np.ndarray:
    """Reconnect a connector that a label interrupts ("—Yes→").

    Recognised label boxes are erased from the connector mask so glyphs cannot
    masquerade as strokes; when a label sits *on* the line this also cuts the
    connector in two. A bridge is drawn only when stroke ink touches opposite
    sides of the label box at aligned positions (left/right at the same height
    or top/bottom at the same column) and the bridge crosses no node.
    """
    result = mask.copy()
    height, width = mask.shape[:2]
    thickness = max(2, int(stroke_width))
    node_boxes = [node.bbox.to_list() for node in nodes]

    def crosses_node(a, b) -> bool:
        # Sample the segment against the node polygons (not their bounding
        # boxes, which overlap the empty corners beside diamonds).
        steps = max(2, int(math.dist(a, b) // 4))
        for i in range(1, steps):
            t = i / steps
            px, py = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            if any(node_contains(node, (px, py), -2.0) for node in nodes):
                return True
        return False

    for word in words:
        if not any(character.isalnum() for character in word.text):
            continue
        bx, by, bw, bh = word.bbox.to_list()
        pad = max(4, int(round(bh * .3)))
        strip = max(6, int(round(bh * .6)), 2 * thickness)
        x0, y0 = max(0, bx - pad), max(0, by - pad)
        x1, y1 = min(width, bx + bw + pad), min(height, by + bh + pad)
        tolerance = max(6.0, .35 * bh)
        sides = {}
        left = mask[y0:y1, max(0, x0 - strip):x0]
        right = mask[y0:y1, x1:min(width, x1 + strip)]
        top = mask[max(0, y0 - strip):y0, x0:x1]
        bottom = mask[y1:min(height, y1 + strip), x0:x1]

        def enters(region: np.ndarray, across_axis: int) -> bool:
            """A stroke entering through this side crosses the strip (runs
            across its depth) and is narrow along the side; a line running
            alongside the label, or an arrowhead wing, does not."""
            if not region.size or not np.any(region):
                return False
            depth = region.shape[across_axis]
            covered = np.count_nonzero(np.any(region > 0, axis=1 - across_axis))
            spread = np.count_nonzero(np.any(region > 0, axis=across_axis))
            return covered >= .6 * depth and spread <= 2.5 * thickness + 4

        entering = set()
        if left.size and np.any(left):
            sides["left"] = (x0 - 1, int(round(float(np.median(np.where(left > 0)[0])) + y0)))
            if enters(left, 1):
                entering.add("left")
        if right.size and np.any(right):
            sides["right"] = (x1, int(round(float(np.median(np.where(right > 0)[0])) + y0)))
            if enters(right, 1):
                entering.add("right")
        if top.size and np.any(top):
            sides["top"] = (int(round(float(np.median(np.where(top > 0)[1])) + x0)), y0 - 1)
            if enters(top, 0):
                entering.add("top")
        if bottom.size and np.any(bottom):
            sides["bottom"] = (int(round(float(np.median(np.where(bottom > 0)[1])) + x0)), y1)
            if enters(bottom, 0):
                entering.add("bottom")
        route = None
        if "left" in sides and "right" in sides and abs(sides["left"][1] - sides["right"][1]) <= tolerance:
            route = [sides["left"], sides["right"]]
        elif "top" in sides and "bottom" in sides and abs(sides["top"][0] - sides["bottom"][0]) <= tolerance:
            route = [sides["top"], sides["bottom"]]
        elif len(sides) == 2 and set(sides) == entering:
            # A label at the corner of an elbow connector: the line enters
            # through one side and leaves through an adjacent side. Rejoin
            # it with the right-angle route the connector itself follows.
            # Both strokes must genuinely enter the box (see ``enters``).
            (first_side, a), (second_side, b) = sorted(sides.items())
            horizontal = {"left", "right"}
            if (first_side in horizontal) != (second_side in horizontal):
                h_point, v_point = (a, b) if first_side in horizontal else (b, a)
                route = [h_point, (v_point[0], h_point[1]), v_point]
        if route and not any(crosses_node(p, q) for p, q in zip(route, route[1:])):
            for p, q in zip(route, route[1:]):
                cv2.line(result, p, q, 255, thickness)
    return result


def infer_obvious_missing_branch_to_end(context: PipelineContext, graph: FlowGraph) -> bool:
    """Recover one likely omitted connector when a decision's sibling branch reaches End.

    This is deliberately narrow: there must be exactly one End, a decision with
    exactly two output-shaped branch targets, one output must already reach End,
    and its sibling output must have no outgoing edge. This gives a little grace
    for a faint/missing connector without guessing from reading order on general
    diagrams. The inferred edge is marked with a warning for review.
    """
    ends = [n for n in graph.nodes if n.type == "terminal" and terminal_role(n.text) == "end"]
    if len(ends) != 1:
        return False
    end = ends[0]
    by_id = {node.id: node for node in graph.nodes}
    outgoing: dict[str, list[Edge]] = defaultdict(list)
    incoming: dict[str, list[Edge]] = defaultdict(list)
    for edge in graph.edges:
        outgoing[edge.source].append(edge)
        incoming[edge.target].append(edge)

    def reaches_end(start_id: str) -> bool:
        pending, seen = [start_id], set()
        while pending:
            current = pending.pop()
            if current == end.id:
                return True
            if current in seen:
                continue
            seen.add(current)
            pending.extend(edge.target for edge in outgoing[current])
        return False

    for decision in (node for node in graph.nodes if node.type == "decision"):
        branches = outgoing[decision.id]
        if len(branches) != 2 or any(edge.label and edge.label.casefold() not in {"true", "false", "yes", "no", "y", "n"} for edge in branches):
            continue
        targets = [by_id.get(edge.target) for edge in branches]
        if any(node is None or node.type != "output" for node in targets):
            continue
        dangling = [node for node in targets if not outgoing[node.id] and not reaches_end(node.id)]
        connected = [node for node in targets if outgoing[node.id] and reaches_end(node.id)]
        if len(dangling) != 1 or len(connected) != 1:
            continue
        source = dangling[0]
        if len(incoming[source.id]) != 1 or incoming[source.id][0].source != decision.id:
            continue
        # Route outside the rightmost shape, then enter End from its right edge.
        # The path is only a reviewable hypothesis; the warning makes that clear.
        sx, sy, sw, sh = source.bbox.to_list()
        ex, ey, ew, eh = end.bbox.to_list()
        image_width = context.stroke_mask.shape[1] if context.stroke_mask is not None else max(sx + sw, ex + ew) + 80
        rightmost = max(node.bbox.x + node.bbox.width for node in graph.nodes)
        route_x = min(image_width - 3, max(sx + sw, rightmost) + 18)
        start = (sx + sw, sy + sh // 2)
        finish = (ex + ew, ey + eh // 2)
        route = [start, (route_x, start[1]), (route_x, finish[1]), finish]

        # Topology alone cannot establish that a connector exists. In particular,
        # two mutually exclusive outputs may intentionally have different
        # continuations (or one may be unfinished); reaching End on the sibling
        # is not evidence for drawing a second edge. Require visible connector
        # ink along the proposed orthogonal route before recovering it.
        stroke_mask = context.stroke_mask
        if stroke_mask is None:
            continue

        if not route_has_visible_stroke(
            stroke_mask, route, min_coverage=0.35, min_segments=2
        ):
            continue

        edge = Edge(_next_edge_id(graph), source.id, end.id, points=route)
        graph.edges.append(edge)
        context.arrows.append(Arrow(edge.id, route, finish, source_hint=source.id, target_hint=end.id))
        context.diagnose(
            "warning", "INFERRED_BRANCH_TO_END",
            "A missing or faint connector was inferred because the sibling output reaches End and this output branch otherwise has no continuation.",
            "Review the inferred connection; remove it if the source diagram intentionally leaves this branch unfinished.",
            [decision.id, source.id, end.id],
        )
        return True
    return False


def infer_high_confidence_missing_edges(context: PipelineContext, graph: FlowGraph) -> int:
    """Recover a small class of missing edges from strong geometry/topology evidence.

    This complements pixel-traced edges rather than replacing them. A vertical
    edge requires visible connector ink in the gap. A missing decision branch is
    only inferred when the other branch is already traced and the sole unconnected
    process/output candidate itself leads to the same End terminal. All recovered
    edges are reviewable warnings.
    """
    if context.stroke_mask is None:
        return 0
    nodes = {node.id: node for node in graph.nodes}
    outgoing: dict[str, list[Edge]] = defaultdict(list)
    incoming: dict[str, list[Edge]] = defaultdict(list)
    for edge in graph.edges:
        outgoing[edge.source].append(edge)
        incoming[edge.target].append(edge)

    def reaches_end(start_id: str, end_id: str) -> bool:
        pending, seen = [start_id], set()
        while pending:
            current = pending.pop()
            if current == end_id:
                return True
            if current in seen:
                continue
            seen.add(current)
            pending.extend(edge.target for edge in outgoing[current])
        return False

    ends = [n for n in graph.nodes if n.type == "terminal" and terminal_role(n.text) == "end"]
    inferred_count = 0
    mask = context.stroke_mask
    height, width = mask.shape[:2]

    # First: a genuinely visible vertical stroke between otherwise unconnected
    # adjacent nodes. The gap must be clear of any intervening node. This is
    # intentionally limited to one unique aligned source/target pair; a line is
    # never promoted to an edge from proximity alone.
    for source in graph.nodes:
        if source.type == "terminal" and terminal_role(source.text) == "end":
            continue
        if outgoing[source.id]:
            continue
        sx, sy, sw, sh = source.bbox.to_list()
        source_cx = sx + sw / 2
        candidates = []
        for target in graph.nodes:
            if target.type == "terminal" and terminal_role(target.text) == "start":
                continue
            if incoming[target.id]:
                continue
            tx, ty, tw, th = target.bbox.to_list()
            gap = ty - (sy + sh)
            if gap <= 0 or gap > max(220, int(0.9 * max(sh, th))):
                continue
            if abs(source_cx - (tx + tw / 2)) > 0.18 * min(sw, tw):
                continue
            if any(other.id not in {source.id, target.id} and
                   other.bbox.y < ty and other.bbox.y + other.bbox.height > sy + sh and
                   abs(other.bbox.center[0] - source_cx) < max(sw, tw) * 0.35
                   for other in graph.nodes):
                continue
            x0 = int(round((source_cx + tx + tw / 2) / 2))
            y0, y1 = sy + sh, ty
            x0 = min(max(x0, 1), width - 2)
            if y0 < 0 or y1 >= height or y1 <= y0:
                continue
            # Count pixels in a narrow corridor, tolerating antialiasing and gaps.
            corridor = mask[max(0, y0):min(height, y1 + 1), max(0, x0 - 3):min(width, x0 + 4)]
            row_hits = np.count_nonzero(corridor, axis=1) > 0
            coverage = float(np.count_nonzero(row_hits)) / max(1, len(row_hits))
            if coverage >= 0.35:
                candidates.append((coverage, target))
        if len(candidates) == 1:
            coverage, target = candidates[0]
            tx, ty, tw, th = target.bbox.to_list()
            start = (int(round(source_cx)), sy + sh)
            finish = (int(round(tx + tw / 2)), ty)
            edge = Edge(_next_edge_id(graph), source.id, target.id,
                        points=[start, finish])
            graph.edges.append(edge)
            context.arrows.append(Arrow(edge.id, edge.points, finish, source_hint=source.id, target_hint=target.id))
            context.diagnose("warning", "INFERRED_VISIBLE_GAP_CONNECTION",
                "A connector was recovered between aligned nodes because continuous connector pixels were visible in the gap.",
                "Review the inferred connection.", [source.id, target.id])
            outgoing[source.id].append(edge); incoming[target.id].append(edge)
            inferred_count += 1

    # A loop-back can be lost when its arrowhead is absorbed by a filled
    # decision. Recover only a same-row process/output -> decision connection
    # when the target already has an incoming path and the complete visible
    # corridor is present. This is geometry-backed, not reading-order guessing.
    for source in graph.nodes:
        if source.type not in {"process", "output"} or outgoing[source.id]:
            continue
        sx, sy, sw, sh = source.bbox.to_list()
        for target in graph.nodes:
            if target.type != "decision" or target.id == source.id or not incoming[target.id]:
                continue
            tx, ty, tw, th = target.bbox.to_list()
            if abs((sy + sh / 2) - (ty + th / 2)) > 0.28 * max(sh, th):
                continue
            if sx + sw <= tx:
                route = [(sx + sw, int(round(sy + sh / 2))), (tx, int(round(ty + th / 2)))]
            elif tx + tw <= sx:
                route = [(sx, int(round(sy + sh / 2))), (tx + tw, int(round(ty + th / 2)))]
            else:
                continue
            if not route_has_visible_stroke(mask, route, min_coverage=0.45, min_segments=1):
                continue
            edge = Edge(_next_edge_id(graph), source.id, target.id, points=route)
            graph.edges.append(edge)
            context.arrows.append(Arrow(edge.id, route, route[-1], source_hint=source.id, target_hint=target.id))
            context.diagnose(
                "warning", "INFERRED_ARROW_DIRECTION",
                "A visibly drawn loop-back was attached to the adjacent decision after its arrowhead was obscured.",
                "Review and accept this inferred loop-back before generating Python.", [source.id, target.id],
            )
            outgoing[source.id].append(edge); incoming[target.id].append(edge)
            inferred_count += 1
            break

    # Second: a missing sibling branch into a downstream process/output node.
    # This handles decisions whose branches lead to different kinds of shapes,
    # not just two output nodes. The candidate must already have a path to End.
    if len(ends) == 1:
        end = ends[0]
        for decision in (n for n in graph.nodes if n.type == "decision"):
            branches = outgoing[decision.id]
            if len(branches) != 1:
                continue
            traced_target = nodes.get(branches[0].target)
            if traced_target is None or traced_target.type not in {"output", "process"}:
                continue
            dx, dy, dw, dh = decision.bbox.to_list()
            candidates = []
            for candidate in graph.nodes:
                if candidate.id in {decision.id, traced_target.id, end.id}:
                    continue
                if candidate.type not in {"process", "output"} or incoming[candidate.id]:
                    continue
                cx, cy, cw, ch = candidate.bbox.to_list()
                # Branch destinations usually leave the decision laterally or
                # below it; reject remote shapes and anything above the decision.
                if cy < dy + 0.25 * dh or cy > dy + dh + 1.6 * max(dh, ch):
                    continue
                horizontal_gap = max(0, cx - (dx + dw), dx - (cx + cw))
                if horizontal_gap > 1.1 * max(dw, cw):
                    continue
                if abs((cx + cw / 2) - (dx + dw / 2)) > 1.5 * max(dw, cw):
                    continue
                if not reaches_end(candidate.id, end.id):
                    continue
                candidates.append(candidate)
            if len(candidates) != 1:
                continue
            target = candidates[0]
            # There should be exactly one missing branch destination, not an
            # arbitrary isolated node elsewhere in the diagram.
            sx, sy, sw, sh = decision.bbox.to_list()
            tx, ty, tw, th = target.bbox.to_list()
            decision_cx, decision_cy = sx + sw / 2, sy + sh / 2
            target_cx, target_cy = tx + tw / 2, ty + th / 2
            polygon = np.asarray(decision.polygon or [], dtype=float)

            def polygon_anchor(side: str) -> tuple[int, int]:
                if len(polygon) >= 3:
                    if side == "bottom":
                        extreme = float(polygon[:, 1].max())
                        selected = polygon[polygon[:, 1] >= extreme - 2]
                        point = (float(np.median(selected[:, 0])), extreme)
                    elif side == "right":
                        extreme = float(polygon[:, 0].max())
                        selected = polygon[polygon[:, 0] >= extreme - 2]
                        point = (extreme, float(np.median(selected[:, 1])))
                    else:
                        extreme = float(polygon[:, 0].min())
                        selected = polygon[polygon[:, 0] <= extreme + 2]
                        point = (extreme, float(np.median(selected[:, 1])))
                    return int(round(point[0])), int(round(point[1]))
                if side == "bottom":
                    return int(round(decision_cx)), sy + sh
                if side == "right":
                    return sx + sw, int(round(decision_cy))
                return sx, int(round(decision_cy))

            vertical_branch = (
                target_cy > decision_cy
                and abs(target_cx - decision_cx) <= 0.35 * min(sw, tw)
            )
            if vertical_branch:
                start = polygon_anchor("bottom")
                finish = (int(round(target_cx)), ty)
            elif target_cx >= decision_cx:
                start = polygon_anchor("right")
                finish = (tx, int(round(target_cy)))
            else:
                start = polygon_anchor("left")
                finish = (tx + tw, int(round(target_cy)))
            route = [start, finish]

            # A unique downstream candidate is not enough: the proposed
            # connection must also follow visible connector ink in the image.
            if context.stroke_mask is None or not route_has_visible_stroke(
                context.stroke_mask, route, min_coverage=0.35, min_segments=1
            ):
                continue

            edge = Edge(_next_edge_id(graph), decision.id, target.id, points=route)
            graph.edges.append(edge)
            context.arrows.append(Arrow(edge.id, route, finish, source_hint=decision.id, target_hint=target.id))
            context.diagnose("warning", "INFERRED_DECISION_BRANCH",
                "A missing decision branch was inferred because the other branch is traced to End and this is the only plausible unconnected process/output destination.",
                "Review the inferred branch and its Yes/No label.", [decision.id, target.id])
            outgoing[decision.id].append(edge); incoming[target.id].append(edge)
            inferred_count += 1
    return inferred_count


def _reaches_through_plain_steps(start_id: str, target_id: str, nodes: dict, graph: FlowGraph) -> bool:
    """Whether a straight chain of single-exit, non-branching steps leads to target."""
    current, seen = start_id, set()
    while current not in seen:
        if current == target_id:
            return True
        seen.add(current)
        node = nodes.get(current)
        following = [edge.target for edge in graph.edges if edge.source == current]
        if node is None or node.type in {"decision", "iteration", "terminal"} or len(following) != 1:
            return False
        current = following[0]
    return False


def repair_semantic_loop_back_edges(context: PipelineContext, graph: FlowGraph) -> int:
    """Recover a lost loop-back from an unmistakable increment/update step."""
    nodes = {node.id: node for node in graph.nodes}
    removable = []
    for edge in graph.edges:
        source, target = nodes.get(edge.source), nodes.get(edge.target)
        if not source or not target or target.type not in {"decision", "iteration"}:
            continue
        if target.bbox.center[1] >= source.bbox.center[1]:
            continue
        if any(other.id not in {source.id, target.id}
               and ((target.bbox.y < other.bbox.center[1] < source.bbox.y)
                    or (other.bbox.center[1] > source.bbox.center[1]
                        and other.bbox.center[1] - source.bbox.center[1] < source.bbox.center[1] - target.bbox.center[1]))
               and abs(other.bbox.center[0] - source.bbox.center[0]) < 0.22 * max(source.bbox.width, target.bbox.width)
               and not any(candidate.target == other.id for candidate in graph.edges)
               for other in graph.nodes):
            removable.append(edge)
    if removable:
        graph.edges[:] = [edge for edge in graph.edges if edge not in removable]
        context.arrows[:] = [arrow for arrow in context.arrows if arrow.id not in {edge.id for edge in removable}]
    adjacency = defaultdict(list)
    for edge in graph.edges:
        adjacency[edge.source].append(edge.target)

    def distance(start, target):
        queue, seen = [(start, 0)], {start}
        while queue:
            current, depth = queue.pop(0)
            if current == target:
                return depth
            for nxt in adjacency[current]:
                if nxt not in seen:
                    seen.add(nxt); queue.append((nxt, depth + 1))
        return None

    repaired = 0
    for source in graph.nodes:
        if source.type not in {"process", "output"}:
            continue
        text = (source.text or "").replace(" ", "").casefold()
        variable_match = re.match(r"([a-z_]\w*)=", text)
        variable = variable_match.group(1) if variable_match else ""
        rhs = text.split("=", 1)[1] if "=" in text else ""
        is_increment = "+1" in text or bool(
            variable and re.search(rf"\b{re.escape(variable)}\s*\+\s*[1il]\b", rhs)
        ) or bool(re.search(r"=i(?:t)?[l1]$", text))
        candidates = []
        for target in graph.nodes:
            if target.type not in {"decision", "iteration"} or target.id == source.id:
                continue
            path_distance = distance(target.id, source.id)
            if is_increment and path_distance is not None:
                candidates.append((path_distance, target))
            elif not is_increment and target.bbox.y > source.bbox.y and variable and variable in (target.text or "").replace(" ", "").casefold():
                route = [
                    (source.bbox.x + source.bbox.width // 2, source.bbox.y + source.bbox.height),
                    (target.bbox.x + target.bbox.width // 2, target.bbox.y),
                ]
                if route_has_visible_stroke(context.stroke_mask, route, min_coverage=0.35, min_segments=1):
                    candidates.append((-(target.bbox.y - source.bbox.y), target))
        if not is_increment and not candidates:
            # OCR may lose the variable from a compact condition such as
            # ``<= 3?``. A unique aligned control directly below the step is
            # still safe when the visible corridor contains no node.
            below = [target for target in graph.nodes
                     if target.type in {"decision", "iteration"}
                     and target.bbox.y > source.bbox.y
                     and abs(target.bbox.center[0] - source.bbox.center[0]) < 0.22 * max(source.bbox.width, target.bbox.width)]
            below = [target for target in below if not any(
                other.id != source.id and other.id != target.id
                and source.bbox.y + source.bbox.height <= other.bbox.center[1] <= target.bbox.y
                and abs(other.bbox.center[0] - source.bbox.center[0]) < 0.22 * max(source.bbox.width, target.bbox.width)
                for other in graph.nodes)]
            if len(below) == 1:
                target = below[0]
                route = [(source.bbox.center[0], source.bbox.y + source.bbox.height),
                         (target.bbox.center[0], target.bbox.y)]
                if route_has_visible_stroke(context.stroke_mask, route, min_coverage=0.35, min_segments=1):
                    candidates.append((0, target))
        if not candidates:
            continue
        if is_increment:
            matching = [item for item in candidates if variable and variable in (item[1].text or "").replace(" ", "").casefold()]
            candidates = matching or candidates
        _, target = max(candidates, key=lambda item: (item[0], -item[1].bbox.y, item[1].id))
        existing = [edge for edge in graph.edges if edge.source == source.id]
        if any(edge.target == target.id for edge in existing):
            continue
        # The drawn arrow already continues to the inferred target through
        # ordinary steps (for example ``passed = passed + 1`` -> ``i = i + 1`` ->
        # loop condition): nothing was lost, so keep the recognized edge.
        if any(_reaches_through_plain_steps(edge.target, target.id, nodes, graph) for edge in existing):
            continue
        for edge in existing:
            graph.edges.remove(edge)
            context.arrows[:] = [arrow for arrow in context.arrows if arrow.id != edge.id]
        sx, sy, sw, sh = source.bbox.to_list()
        tx, ty, tw, th = target.bbox.to_list()
        route = [(sx + sw // 2, sy + sh // 2), (tx + tw // 2, ty + th // 2)]
        edge = Edge(_next_edge_id(graph), source.id, target.id, points=route)
        graph.edges.append(edge)
        context.arrows.append(Arrow(edge.id, route, route[-1], source_hint=source.id, target_hint=target.id))
        context.diagnose(
            "warning", "INFERRED_ARROW_DIRECTION",
            "A loop-back was recovered from an increment step and the reachable upstream loop condition.",
            "Review and accept this inferred loop-back before generating Python.", [source.id, target.id],
        )
        repaired += 1
        context.diagnostics[:] = [
            diagnostic for diagnostic in context.diagnostics
            if not (diagnostic.code == "AMBIGUOUS_ARROW_DIRECTION"
                    and {source.id, target.id}.issubset(diagnostic.element_ids))
        ]
    return repaired


def infer_dangling_arrow_to_existing_edge(
    context: PipelineContext, graph: FlowGraph, fragments: list[tuple[Node, np.ndarray]], image_shape: tuple[int, int]
) -> int:
    """Attach a visible arrowhead that terminates just short of another connector.

    Some diagrams draw a branch arrow into the side of an existing vertical or
    horizontal connector. Rasterization can leave a tiny white gap, so connected
    components report the branch as dangling even though its arrowhead clearly
    points at the receiving stroke. This recovery requires a detected arrowhead,
    a short geometrically aligned gap, a unique receiving edge, and a real source
    node contact. It never joins fragments based on reading order alone.
    """
    if not fragments or not graph.edges:
        return 0
    nodes = {node.id: node for node in graph.nodes}
    outgoing = {node.id for node in graph.nodes if any(edge.source == node.id for edge in graph.edges)}
    height, width = image_shape
    max_gap = max(36.0, min(180.0, width * 0.09))
    inferred = 0

    def sampled_path(points: list[tuple[int, int]], step: float = 5.0) -> list[tuple[float, float]]:
        samples: list[tuple[float, float]] = []
        for first, second in zip(points, points[1:]):
            length = math.dist(first, second)
            count = max(1, int(math.ceil(length / step)))
            for index in range(count + 1):
                ratio = index / count
                samples.append((first[0] + (second[0] - first[0]) * ratio,
                                first[1] + (second[1] - first[1]) * ratio))
        return samples

    for source, points in fragments:
        if source.id in outgoing or len(points) < 12:
            continue
        # Arrowheads are found at the four extremities of an orthogonal route.
        # Use a median coordinate to avoid choosing an antialiased corner pixel.
        tips = []
        for axis in (0, 1):
            minimum, maximum = int(points[:, axis].min()), int(points[:, axis].max())
            for value, sign in ((minimum, 1), (maximum, -1)):
                band = points[points[:, axis] == value]
                other = 1 - axis
                tip = ((value, int(round(float(np.median(band[:, other]))))) if axis == 0
                       else (int(round(float(np.median(band[:, other])))), value))
                # The main detector is strongest on clean triangular heads. A
                # second, local flare test handles outlined/V-shaped arrowheads
                # whose antialiased extreme consists of several pixels.
                flare_widths = []
                for offset in (3, 8, 14, 22, 32, 44):
                    coordinate = value + sign * offset
                    slab = points[np.abs(points[:, axis] - coordinate) <= 2]
                    if len(slab):
                        flare_widths.append(float(np.ptp(slab[:, other]) + 1))
                flared = (len(flare_widths) >= 4
                          and max(flare_widths[:5]) <= 80.0
                          and max(flare_widths[1:5]) >= max(12.0, flare_widths[0] * 1.8)
                          and any(flare_widths[i + 1] > flare_widths[i] * 1.20
                                  for i in range(min(3, len(flare_widths) - 1))))
                if arrowhead_evidence(points, tip) or flared:
                    tips.append(tip)
        if not tips:
            continue

        options = []
        for tip in tips:
            nearby_distances = np.linalg.norm(
                points.astype(float) - np.asarray(tip, dtype=float), axis=1
            )
            nearby = points[nearby_distances < max(70.0, width * 0.045)]
            if len(nearby) < 4:
                continue
            # Estimate the arrow's pointing vector from the local shaft toward
            # the tip, rather than from the whole elbow-shaped component.
            distances = np.linalg.norm(nearby.astype(float) - np.asarray(tip, dtype=float), axis=1)
            shaft = nearby[distances >= 12.0]
            if len(shaft) < 3:
                continue
            direction = np.asarray(tip, dtype=float) - np.median(shaft.astype(float), axis=0)
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm < 1.0:
                continue
            direction /= direction_norm

            for edge in graph.edges:
                target = nodes.get(edge.target)
                if target is None or edge.source == source.id or edge.target == source.id or len(edge.points) < 2:
                    continue
                route_samples = sampled_path(edge.points)
                if not route_samples:
                    continue
                distances_to_route = [math.dist(tip, point) for point in route_samples]
                nearest_index = int(np.argmin(distances_to_route))
                projection = route_samples[nearest_index]
                gap = distances_to_route[nearest_index]
                if gap < 5.0 or gap > max_gap:
                    continue
                toward = np.asarray(projection, dtype=float) - np.asarray(tip, dtype=float)
                toward_norm = float(np.linalg.norm(toward))
                if toward_norm <= 0:
                    continue
                alignment = float(np.dot(direction, toward / toward_norm))
                if alignment < 0.62:
                    continue
                # A receiving path must continue to a distinct node in the
                # arrow's forward direction. Keep the target endpoint grounded
                # in an already traced edge, not a guessed nearby shape.
                options.append((gap, -alignment, source, tip, edge, projection, route_samples, nearest_index))

        if not options:
            continue
        options.sort(key=lambda item: (item[0], item[1]))
        best = options[0]
        if len(options) > 1 and options[1][0] <= best[0] + max(8.0, best[0] * 0.12):
            continue
        gap, neg_alignment, source, tip, receiving_edge, projection, route_samples, nearest_index = best
        target_id = receiving_edge.target
        if any(edge.source == source.id and edge.target == target_id for edge in graph.edges):
            continue

        # Recover the physical branch route from its source contact to the
        # arrowhead, then bridge the small gap and follow the receiving edge to
        # its already-established destination.
        # contacts() may be slightly offset from the actual node boundary;
        # use the closest observed pixel as a robust path start.
        nearest_source = points[int(np.argmin(np.linalg.norm(
            points.astype(float) - np.asarray(source.bbox.center, dtype=float), axis=1)))]
        branch_points = stroke_path(points, (int(nearest_source[0]), int(nearest_source[1])), tip)
        if not branch_points:
            # A few antialiased pixels can disconnect the pixel-walk even though
            # the fragment was one component. Preserve its elbow geometry by
            # selecting the observed point farthest from the direct source-tip
            # chord; for a right-angle route this is the corner.
            start_point = np.asarray(nearest_source, dtype=float)
            end_point = np.asarray(tip, dtype=float)
            chord = end_point - start_point
            denominator = max(1.0, float(np.dot(chord, chord)))
            offsets = points.astype(float) - start_point
            ratios = np.clip((offsets @ chord) / denominator, 0.0, 1.0)
            projections = start_point + ratios[:, None] * chord
            deviations = np.linalg.norm(points.astype(float) - projections, axis=1)
            elbow = points[int(np.argmax(deviations))]
            branch_points = [tuple(map(int, nearest_source)), tuple(map(int, elbow)), tip]
        route = [(int(x), int(y)) for x, y in branch_points]
        bridge_point = (int(round(projection[0])), int(round(projection[1])))
        if route[-1] != bridge_point:
            route.append(bridge_point)
        # Continue along the receiver toward its target. The sampled path is
        # ordered source-to-target, so the nearest point's suffix is correct.
        suffix = route_samples[nearest_index + 1:]
        route.extend((int(round(x)), int(round(y))) for x, y in suffix)
        target_node = nodes[target_id]
        tx, ty, tw, th = target_node.bbox.to_list()
        target_anchor = (int(round(tx + tw / 2)), int(round(ty + th / 2)))
        if math.dist(route[-1], target_anchor) > max_gap * 2:
            route.append(target_anchor)
        edge = Edge(_next_edge_id(graph), source.id, target_id, points=route)
        graph.edges.append(edge)
        context.arrows.append(Arrow(edge.id, route, route[-1], source_hint=source.id, target_hint=target_id))
        # The dangling-fragment warning is no longer actionable once its
        # arrowhead has been safely attached to a receiving connector.
        context.diagnostics[:] = [
            diagnostic for diagnostic in context.diagnostics
            if not (diagnostic.code == "DANGLING_ARROW" and diagnostic.element_ids == [source.id])
        ]
        context.diagnose(
            "warning", "INFERRED_ARROW_TO_CONNECTOR",
            "A dangling arrow was connected to a nearby existing connector because its visible arrowhead points into that stroke.",
            "Review the recovered branch connection.", [source.id, target_id, receiving_edge.id],
        )
        outgoing.add(source.id)
        inferred += 1
    return inferred


def suppress_node_outline_artifacts(mask: np.ndarray, nodes: list[Node]) -> np.ndarray:
    """Remove residual node-border ink before connector component analysis.

    Filled-node contours do not always include the outer half of thick,
    anti-aliased outlines. Those residual borders can join otherwise separate
    arrows into one giant component. Erase a small, image-scaled band around
    each recognized polygon; connector endpoints are recovered by the bounded
    contact radius in GraphService, while the original context mask is retained
    for review and high-confidence recovery.
    """
    if not nodes:
        return mask
    height, width = mask.shape[:2]
    cleaned = mask.copy()
    page_margin = max(8, min(36, int(round(width * 0.01))))
    for node in nodes:
        bx, by, bw, bh = node.bbox.to_list()
        # Retain the established page-scaled cleanup for ordinary/larger
        # shapes, but cap it for compact nodes. A page-wide safety band can
        # consume most of a compact node's short side and erase nearby arrowheads.
        local_margin = max(4, int(round(min(bw, bh) * 0.06)))
        margin = min(page_margin, local_margin)
        kernel_size = margin * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        x0, y0 = max(0, bx - margin - 2), max(0, by - margin - 2)
        x1, y1 = min(width, bx + bw + margin + 3), min(height, by + bh + margin + 3)
        region = np.zeros((y1 - y0, x1 - x0), np.uint8)
        polygon = np.asarray(
            node.polygon or [(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)],
            np.int32,
        ) - np.array([x0, y0])
        cv2.fillPoly(region, [polygon], 255)
        region = cv2.dilate(region, kernel)
        cleaned[y0:y1, x0:x1][region > 0] = 0
    return cleaned

def arrowhead_ink(context: PipelineContext, nodes: list[Node], stroke_width: int) -> np.ndarray | None:
    """Raw connector ink with node bodies (and their outlines) removed.

    The connector mask has a wide safety band erased around every node, which
    also erases most of an arrowhead that touches its target. Arrowhead tests
    use this mask instead: only the polygon plus roughly one stroke width is
    removed, enough to drop the node's own outline but keep the arrowhead.
    """
    if context.grayscale is None:
        return None
    ink = ink_mask(context.grayscale)
    removal = np.zeros_like(ink)
    for node in nodes:
        bx, by, bw, bh = node.bbox.to_list()
        polygon = np.asarray(node.polygon or [(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)], np.int32)
        cv2.fillPoly(removal, [polygon], 255)
    size = int(2 * stroke_width + 3) | 1
    ink[cv2.dilate(removal, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))) > 0] = 0
    if stroke_width >= 3:
        # Drop 1-2 px anti-aliasing fringes of node outlines; they run off
        # sideways from an arrowhead tip and would look like a crossing line.
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # Recognised labels next to an arrowhead would otherwise widen its profile.
    # Erase only glyph components lying entirely inside a label box: an
    # arrowhead wing that the box happens to overlap continues outside it.
    ink = erase_enclosed_components(
        ink, [word.bbox for word in context.edge_words if any(ch.isalnum() for ch in word.text)],
        protect=cv2.dilate(removal, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))))
    return ink


class GraphService:
    name = "graph"

    def run(self, context: PipelineContext) -> PipelineContext:
        graph = FlowGraph(nodes=context.elements)
        context.graph = graph
        mask = context.stroke_mask
        if mask is None:
            context.diagnose("error", "MISSING_CONNECTORS", "No connector geometry is available.", "Run vision before graph construction.")
            return context
        # VisionService already performs a 3x3 close. Do not dilate/close again
        # here: on diagrams with tight spacing this can join a connector to a
        # neighboring node outline or to a second branch, producing a fabricated
        # path that follows unrelated borders. Keep connected-component topology
        # as close to the observed pixels as possible.
        mask = suppress_node_outline_artifacts(mask, graph.nodes)
        mask = bridge_aligned_connector_gaps(mask, graph.nodes)
        stroke_width = connector_stroke_width(mask)
        mask = bridge_through_labels(mask, context.edge_words, graph.nodes, stroke_width)
        head_mask = arrowhead_ink(context, graph.nodes, stroke_width)
        touch_distance = max(context.settings.ARROW_ATTACH_DISTANCE,
                             int(round(mask.shape[1] * context.settings.CONNECTOR_TOUCH_PERMILLE / 1000)))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        components = sorted(range(1, count), key=lambda index: (int(stats[index, cv2.CC_STAT_TOP]), int(stats[index, cv2.CC_STAT_LEFT])))
        candidates = []
        deferred: list[tuple[list[str], tuple]] = []
        dangling_fragments: list[tuple[Node, np.ndarray]] = []
        for index in components:
            if stats[index, cv2.CC_STAT_AREA] < 20 or max(stats[index, cv2.CC_STAT_WIDTH], stats[index, cv2.CC_STAT_HEIGHT]) < 18:
                continue
            x, y, width, height = stats[index, :4]
            ys, xs = np.where(labels[y:y + height, x:x + width] == index)
            points = np.column_stack((xs + x, ys + y))
            # Vision removes a narrow safety band around node outlines so
            # borders cannot masquerade as arrows. Compensate with a modest
            # image-scaled attachment radius, bounded by the configured minimum.
            attach_distance = max(
                context.settings.ARROW_ATTACH_DISTANCE,
                int(round(mask.shape[1] * 0.05)),
            )
            ports = contacts(points, graph.nodes, mask.shape, attach_distance,
                             stroke_width=stroke_width, head_mask=head_mask,
                             touch_distance=touch_distance)
            # One physical connector end per node: a branch label glued to the
            # shaft, or an elbow that runs along the node before entering it,
            # can create several contacts on the same node. Prefer the
            # arrowhead-backed contact; otherwise keep the closest approach.
            # One stroke end can lie near two nodes (e.g. under the slanted
            # corner of a neighbouring parallelogram); it belongs to the node it
            # actually approaches more closely.
            same_end = 3.0 * stroke_width + 6.0
            ports = [port for port in ports if not any(
                other.node.id != port.node.id and other.distance + 2.0 < port.distance
                and math.dist(other.point, port.point) <= same_end for other in ports)]
            ports_by_node = defaultdict(list)
            for port in ports:
                ports_by_node[port.node.id].append(port)
            filtered_ports = []
            for node_ports in ports_by_node.values():
                head_ports = [port for port in node_ports if port.head]
                filtered_ports.append(min(head_ports or node_ports, key=lambda port: port.distance))
            ports = sorted(filtered_ports, key=lambda port: (port.point[1], port.point[0], port.node.id))
            # Node-border cleanup can leave a visible connector tip just beyond
            # the normal attachment band. Expand only when the observed stroke
            # currently touches exactly one node and the second contact is a
            # Start/End terminal. This narrowly recovers terminal connectors
            # without broadening ordinary branch matching or joining nearby
            # decision branches by proximity alone.
            if len({port.node.id for port in ports}) == 1:
                expanded_distance = max(attach_distance, int(round(mask.shape[1] * 0.035)))
                if expanded_distance > attach_distance:
                    expanded_ports = contacts(points, graph.nodes, mask.shape, expanded_distance)
                    expanded_nodes = {port.node.id: port.node for port in expanded_ports}
                    original_ids = {port.node.id for port in ports}
                    terminal_nodes = [node for node in expanded_nodes.values()
                                      if node.type == "terminal"
                                      and terminal_role(node.text) is not None]
                    if (len(expanded_nodes) == 2 and len(terminal_nodes) == 1
                            and set(expanded_nodes) != original_ids):
                        ports = expanded_ports
            ids = list(dict.fromkeys(port.node.id for port in ports))
            sources, targets = [port for port in ports if not port.head], [port for port in ports if port.head]

            # Arrowheads that terminate at a node boundary can be partly erased
            # when VisionService removes the node fill. Use strong terminal
            # semantics as a constrained fallback only when this component
            # touches exactly two distinct nodes: Start is an origin and End is
            # a destination. This recovers curved incoming paths without using
            # general reading-order guesses, and cannot merge a component that
            # touches three or more nodes.
            node_ports = defaultdict(list)
            for port in ports:
                node_ports[port.node.id].append(port)
            distinct_nodes = list(node_ports)
            anchored_pair = None
            if len(distinct_nodes) == 2:
                start_nodes = [node for node in graph.nodes
                               if node.id in node_ports and node.type == "terminal"
                               and terminal_role(node.text) == "start"]
                end_nodes = [node for node in graph.nodes
                             if node.id in node_ports and node.type == "terminal"
                             and terminal_role(node.text) == "end"]
                if len(start_nodes) == 1:
                    start_node = start_nodes[0]
                    other = next(node for node in graph.nodes if node.id in node_ports and node.id != start_node.id)
                    anchored_pair = (node_ports[start_node.id][0], node_ports[other.id][0])
                elif len(end_nodes) == 1:
                    end_node = end_nodes[0]
                    other = next(node for node in graph.nodes if node.id in node_ports and node.id != end_node.id)
                    anchored_pair = (node_ports[other.id][0], node_ports[end_node.id][0])
            if anchored_pair is not None:
                source, target = anchored_pair
                path = stroke_path(points, source.point, target.point)
                if path:
                    # Warn only when the drawn arrowhead does not confirm the
                    # Start/End direction: no head at the target end, or a head
                    # seen at the source end. A source port is normally a tail,
                    # so testing ``not source.head`` warned on every terminal
                    # connector, including ones with clear arrowheads.
                    if source.head or not target.head:
                        context.diagnose(
                            "warning", "INFERRED_TERMINAL_DIRECTION",
                            "Connector direction was inferred from a Start/End terminal because its arrowhead was partly obscured.",
                            "Confirm this connection during review.",
                            [source.node.id, target.node.id],
                        )
                    candidates.append((source, target, path))
                else:
                    context.diagnose("error", "DANGLING_ARROW", "A terminal-anchored connector has no continuous pixel path.", "Reconnect the arrow in the source diagram.", [source.node.id, target.node.id])
                continue

            if len(ports) < 2:
                if len(ports) == 1 and not ports[0].head:
                    dangling_fragments.append((ports[0].node, points.copy()))
                context.diagnose("warning", "DANGLING_ARROW", "A connector fragment does not reach two shape boundaries.", "Attach each arrow to its source and destination; remove stray lines.", ids)
                continue

            # Conservative semantic fallback for common node-role transitions
            # when a small arrowhead was lost at a filled-shape boundary. This
            # does not use page reading order: it is allowed only for a single
            # physical connector with exactly two contacts and a unique role
            # transition (input -> decision, or decision -> output).
            if not targets and len(ports) == 2:
                first, second = ports
                role_pair = {first.node.type, second.node.type}
                inferred = None
                if role_pair == {"input", "decision"}:
                    source = first if first.node.type == "input" else second
                    target = first if first.node.type == "decision" else second
                    inferred = (source, target)
                elif role_pair == {"decision", "output"}:
                    decision = first if first.node.type == "decision" else second
                    output = first if first.node.type == "output" else second
                    # A lateral branch leaves the decision; a vertically
                    # aligned output above a decision is the preceding step.
                    if abs(output.node.bbox.center[0] - decision.node.bbox.center[0]) > 0.35 * max(output.node.bbox.width, decision.node.bbox.width):
                        inferred = (decision, output)
                    elif output.node.bbox.center[1] < decision.node.bbox.center[1]:
                        inferred = (output, decision)
                    else:
                        inferred = (decision, output)
                elif role_pair == {"output", "terminal"}:
                    output = first if first.node.type == "output" else second
                    terminal = first if first.node.type == "terminal" else second
                    inferred = (output, terminal) if terminal_role(terminal.text) == "end" else (terminal, output)
                elif role_pair == {"output", "decision"}:
                    output = first if first.node.type == "output" else second
                    decision = first if first.node.type == "decision" else second
                    inferred = (output, decision) if output.node.bbox.center[1] < decision.node.bbox.center[1] else (decision, output)
                elif role_pair in ({"decision", "decision"}, {"output", "output"}, {"process", "process"}):
                    # With the arrowhead partially covered by a filled node,
                    # a clearly vertical connector between aligned nodes has a
                    # strong top-to-bottom direction cue. Do not generalize
                    # this to horizontal same-role pairs, which are ambiguous.
                    upper, lower = sorted(
                        (first, second), key=lambda port: port.node.bbox.center[1]
                    )
                    vertical_gap = lower.node.bbox.center[1] - upper.node.bbox.center[1]
                    horizontal_gap = abs(lower.node.bbox.center[0] - upper.node.bbox.center[0])
                    if vertical_gap > 0.65 * max(upper.node.bbox.height, lower.node.bbox.height):
                        inferred = (upper, lower)
                elif role_pair <= {"process", "input", "output", "decision"}:
                    # The same top-to-bottom cue applies to any other pair of
                    # ordinary flow steps (for example Declare -> Read), but
                    # only when the two shapes are stacked on one column so
                    # the connector is a straight vertical step, never a
                    # lateral branch or a routed loop-back.
                    upper, lower = sorted(
                        (first, second), key=lambda port: port.node.bbox.center[1]
                    )
                    vertical_gap = lower.node.bbox.center[1] - upper.node.bbox.center[1]
                    horizontal_gap = abs(lower.node.bbox.center[0] - upper.node.bbox.center[0])
                    if (vertical_gap > 0.65 * max(upper.node.bbox.height, lower.node.bbox.height)
                            and horizontal_gap <= 0.18 * min(upper.node.bbox.width, lower.node.bbox.width)
                            and abs(upper.point[0] - lower.point[0]) <= 0.18 * min(upper.node.bbox.width, lower.node.bbox.width)):
                        inferred = (upper, lower)
                if inferred is not None:
                    source, target = inferred
                    path = stroke_path(points, source.point, target.point)
                    if path:
                        context.diagnose(
                            "warning", "INFERRED_ARROW_DIRECTION",
                            "Arrow direction was inferred from a clear input/decision/output role transition because the arrowhead was not legible.",
                            "Confirm this connection during review.", [source.node.id, target.node.id],
                        )
                        candidates.append((source, target, path))
                        continue

            if not targets:
                deferred.append((ids, ("error", "AMBIGUOUS_ARROW_DIRECTION", "A connector has no reliably visible arrowhead.", "Use a larger, visible V-shaped or filled triangular arrowhead. Direction is never inferred from layout.", ids)))
                continue
            if not sources:
                deferred.append((ids, ("error", "AMBIGUOUS_CONNECTOR", "A connector has arrowheads but no reliable source boundary.", "Check that each arrow starts on a shape boundary.", ids)))
                continue

            # Two curved branches can touch the same terminal boundary and be
            # reported as one connected component. Pair multiple sources and
            # arrowheads by the actual pixel route, not reading order or layout.
            # This is intentionally limited to balanced multi-port components;
            # unbalanced junctions remain ambiguous rather than inventing edges.
            pairs_to_trace = []
            if len(sources) > 1 and len(targets) > 1:
                routes = []
                for source in sources:
                    for target in targets:
                        route = stroke_path(points, source.point, target.point)
                        if route:
                            length = sum(math.dist(a, b) for a, b in zip(route, route[1:]))
                            routes.append((length, source, target, route))
                used_sources, used_targets = set(), set()
                for _, source, target, route in sorted(routes, key=lambda item: item[0]):
                    source_key = (source.node.id, source.point)
                    target_key = (target.node.id, target.point)
                    if source_key in used_sources or target_key in used_targets:
                        continue
                    used_sources.add(source_key); used_targets.add(target_key)
                    pairs_to_trace.append((source, target, route))
                if len(pairs_to_trace) != min(len(sources), len(targets)):
                    context.diagnose("error", "AMBIGUOUS_CONNECTOR", "A multi-branch connector could not be separated into complete paths.", "Separate overlapping paths or increase the gap between the arrows.", ids)
                    continue
            elif len(sources) == 1:
                pairs_to_trace = [(sources[0], target, stroke_path(points, sources[0].point, target.point)) for target in targets]
            elif len(targets) == 1:
                # Flow lines that join before a single arrowhead (a loop-back
                # collecting several branches, or every branch converging on
                # End) are standard flowchart notation: each arrowhead-free
                # end is a tail and the one arrowhead is their common target.
                # Every edge follows visible ink to the visible arrowhead.
                target = targets[0]
                pairs_to_trace = [(source, target, stroke_path(points, source.point, target.point))
                                  for source in sources]
            else:
                context.diagnose("error", "AMBIGUOUS_CONNECTOR", "A crossing or multi-way connector cannot be resolved safely.", "Separate crossing lines and use labeled connector circles for distant paths.", ids)
                continue

            for source, target, path in pairs_to_trace:
                if not path:
                    context.diagnose("error", "DANGLING_ARROW", "A connector path is interrupted.", "Reconnect the line in the source diagram.", [source.node.id, target.node.id])
                    continue
                candidates.append((source, target, path))
        # The same node pair can be reported by two components when a stray
        # fragment (e.g. part of a label glyph) sits beside the real arrow.
        unique, seen_pairs = [], set()
        for item in candidates:
            pair = (item[0].node.id, item[1].node.id)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique.append(item)
        candidates = unique
        # A direction-ambiguous fragment is only an error when it is the sole
        # evidence for its connection; if a clean connector already joins the
        # same nodes it is a redundant fragment, reported as a warning only.
        connected = {frozenset(pair) for pair in seen_pairs}
        for ids, arguments in deferred:
            if len(ids) == 2 and frozenset(ids) in connected:
                context.diagnose("warning", "REDUNDANT_CONNECTOR_FRAGMENT",
                                 "A short stroke beside an already-traced connector was ignored.",
                                 "No action is needed unless a second, separate connection was intended.", ids)
            else:
                context.diagnose(*arguments)
        candidates.sort(key=lambda item: (min(point[1] for point in item[2]), min(point[0] for point in item[2]), item[0].node.id, item[1].node.id))
        for index, (source, target, path) in enumerate(candidates, 1):
            edge = Edge(f"e{index}", source.node.id, target.node.id, points=path)
            graph.edges.append(edge)
            context.arrows.append(Arrow(edge.id, path, target.point, source_hint=edge.source, target_hint=edge.target))
        # Recover a clearly arrowheaded branch that stops just short of an
        # existing connector stroke (a common merge/junction rasterization gap).
        infer_dangling_arrow_to_existing_edge(context, graph, dangling_fragments, mask.shape)
        # Be forgiving of a single likely omitted/faint branch-to-End connector,
        # but only when the decision's sibling output already establishes the
        # intended convergence. General reading-order edge guessing remains off.
        infer_obvious_missing_branch_to_end(context, graph)
        infer_high_confidence_missing_edges(context, graph)
        repair_semantic_loop_back_edges(context, graph)
        node_types = {node.id: node.type for node in graph.nodes}
        for word in context.edge_words:
            if not any(character.isalnum() for character in word.text):
                continue
            distances = sorted((distance_to_path(word.bbox.center, edge.points), index) for index, edge in enumerate(graph.edges))
            # Yes/No/True/False only has meaning on a decision's outgoing
            # branch. When such a branch is a comparably close candidate (for
            # example where a merging arrow shares the branch's trunk), assign
            # the label among decision branches only.
            if normalize_label(word.text) and distances:
                branch = [(d, i) for d, i in distances
                          if node_types.get(graph.edges[i].source) in ("decision", "iteration")]
                if branch and branch[0][0] <= max(75, word.bbox.height * 3) and branch[0][0] <= distances[0][0] * 1.5 + 5:
                    distances = branch
            if not distances or distances[0][0] > max(75, word.bbox.height * 3):
                continue
            if len(distances) > 1 and abs(distances[1][0] - distances[0][0]) < 5:
                if normalize_label(word.text):
                    context.diagnose("error", "AMBIGUOUS_EDGE_LABEL", "A branch label is equally close to multiple arrows.", "Move each Yes/No label closer to its own branch.", [graph.edges[distances[0][1]].id, graph.edges[distances[1][1]].id])
                continue
            edge = graph.edges[distances[0][1]]
            label = normalize_label(word.text) or word.text
            if edge.label:
                context.diagnose("error", "AMBIGUOUS_EDGE_LABEL", "Multiple text labels are attached to one arrow.", "Use one clearly positioned label for each branch.", [edge.id])
            else:
                edge.label = label
                if word.confidence * 100 < context.settings.OCR_MIN_CONFIDENCE:
                    context.diagnose("warning", "LOW_OCR_CONFIDENCE", "An arrow label may have been misread.", "Verify the label or export larger text.", [edge.id])
        # When branch-label OCR is missing, infer Yes/No only for the common,
        # unambiguous left/right split of a decision. Keep this explicit as a
        # warning so the review step can confirm or change the labels.
        for node in graph.nodes:
            if node.type != "decision":
                continue
            outgoing_edges = [edge for edge in graph.edges if edge.source == node.id]
            if len(outgoing_edges) != 2 or any(edge.label for edge in outgoing_edges):
                continue
            center_x = node.bbox.center[0]
            target_nodes = {item.id: item for item in graph.nodes}
            positions = [(edge, target_nodes[edge.target].bbox.center[0]) for edge in outgoing_edges
                         if edge.target in target_nodes]
            # A target is a left/right branch only when its horizontal
            # displacement dominates its vertical displacement. Otherwise a
            # lower node slightly offset from the diamond center can be
            # misclassified as the left branch of a vertical/sideways split.
            left = [edge for edge, x in positions
                    if x < center_x - 10 and
                    abs(x - center_x) > abs(target_nodes[edge.target].bbox.center[1] - node.bbox.center[1])]
            right = [edge for edge, x in positions
                     if x > center_x + 10 and
                     abs(x - center_x) > abs(target_nodes[edge.target].bbox.center[1] - node.bbox.center[1])]
            if len(left) == 1 and len(right) == 1:
                left[0].label, right[0].label = "Yes", "No"
                context.diagnose(
                    "warning", "INFERRED_BRANCH_LABELS",
                    "Yes/No labels were inferred from a clear left/right decision split because label OCR was unavailable.",
                    "Confirm the inferred Yes/No labels during review.",
                    [node.id, left[0].id, right[0].id],
                )
            elif len(positions) == 2:
                # Another common layout has one branch continuing vertically
                # and the other leaving sideways. Keep this inference explicit
                # and warning-tagged rather than presenting it as OCR evidence.
                vertical = [edge for edge, x in positions
                            if abs(target_nodes[edge.target].bbox.center[1] - node.bbox.center[1]) >
                            abs(target_nodes[edge.target].bbox.center[0] - center_x)]
                horizontal = [edge for edge, x in positions if edge not in vertical]
                if len(vertical) == len(horizontal) == 1:
                    # With OCR unavailable, use the conventional
                    # vertical-continuation / lateral-exit mapping. Keep this
                    # explicitly warning-tagged: geometry is a fallback, not
                    # evidence that overrides a recognized branch label.
                    vertical[0].label, horizontal[0].label = "True", "False"
                    context.diagnose(
                        "warning", "INFERRED_BRANCH_LABELS",
                        "True/False labels were inferred from the decision's vertical-versus-sideways split because label OCR was unavailable.",
                        "Confirm the inferred branch labels during review.",
                        [node.id, vertical[0].id, horizontal[0].id],
                    )

            # Some diagrams route both branches downward after leaving opposite
            # sides of a diamond. Target-centre geometry alone cannot distinguish
            # them because both destinations are below the decision. Use the
            # actual connector departure points as a second, warning-tagged cue.
            # Explicit OCR labels always take priority; inference requires exactly
            # two unlabeled branches, opposite-side departures, and downstream targets.
            if len(outgoing_edges) == 2 and not any(edge.label for edge in outgoing_edges):
                center_x, center_y = node.bbox.center
                side_edges = {"left": [], "right": []}
                target_nodes = {item.id: item for item in graph.nodes}
                for edge in outgoing_edges:
                    target = target_nodes.get(edge.target)
                    if not target or not edge.points:
                        continue
                    # Paths are source-to-target ordered, so the first point is
                    # the connector's departure at the decision boundary.
                    sx, sy = edge.points[0]
                    dx = sx - center_x
                    if target.bbox.center[1] <= center_y or abs(dx) < max(12, node.bbox.width * 0.22):
                        continue
                    side_edges["left" if dx < 0 else "right"].append(edge)
                if len(side_edges["left"]) == len(side_edges["right"]) == 1:
                    left_edge = side_edges["left"][0]
                    right_edge = side_edges["right"][0]
                    left_edge.label, right_edge.label = "Yes", "No"
                    context.diagnose(
                        "warning", "INFERRED_BRANCH_LABELS",
                        "Yes/No labels were inferred from connector departure sides because label OCR was unavailable and both branches continue downward.",
                        "Confirm the inferred Yes/No labels during review.",
                        [node.id, left_edge.id, right_edge.id],
                    )

        # A connected-component pass can report an ambiguous junction before
        # the constrained recovery above reconstructs its individual edges. Do
        # not let that stale error block validation when the graph now contains
        # the incoming edge and both branches of the same decision. Keep a
        # warning for review; genuinely unresolved components remain errors.
        remaining_diagnostics = []
        for diagnostic in context.diagnostics:
            if diagnostic.code != "AMBIGUOUS_CONNECTOR" or len(diagnostic.element_ids) < 3:
                remaining_diagnostics.append(diagnostic)
                continue
            mentioned = set(diagnostic.element_ids)
            decisions = [node for node in graph.nodes
                         if node.id in mentioned and node.type in ("decision", "iteration")]
            resolved = False
            for decision in decisions:
                incoming_from_component = [edge for edge in graph.edges
                                           if edge.target == decision.id and edge.source in mentioned]
                outgoing = [edge for edge in graph.edges if edge.source == decision.id]
                branch_targets_in_component = [edge for edge in outgoing if edge.target in mentioned]
                if incoming_from_component and len(outgoing) == 2 and branch_targets_in_component:
                    resolved = True
                    break
            if resolved:
                context.diagnose(
                    "warning", "RESOLVED_CONNECTOR_AMBIGUITY",
                    "A connector junction initially looked ambiguous, but the recovered graph now contains the incoming connection and both decision branches.",
                    "Review the reconstructed paths; no source diagram change is needed if they match the arrows.",
                    diagnostic.element_ids,
                )
            else:
                remaining_diagnostics.append(diagnostic)
        context.diagnostics = remaining_diagnostics

        # Circle pairs bridge an incoming physical arrow to an outgoing physical arrow.
        pairs = defaultdict(list)
        for node in graph.nodes:
            if node.type == "connector":
                pairs[node.text.strip().casefold()].append(node)
        for label in sorted(pairs):
            pair = pairs[label]
            if len(pair) != 2:
                continue  # Validation supplies the matching-pair diagnostic.
            first, second = pair
            incoming_first = any(edge.target == first.id for edge in graph.edges)
            outgoing_first = any(edge.source == first.id for edge in graph.edges)
            incoming_second = any(edge.target == second.id for edge in graph.edges)
            outgoing_second = any(edge.source == second.id for edge in graph.edges)
            if incoming_first and not outgoing_first and outgoing_second and not incoming_second:
                graph.edges.append(Edge(_next_edge_id(graph), first.id, second.id, structural=True))
            elif incoming_second and not outgoing_second and outgoing_first and not incoming_first:
                graph.edges.append(Edge(_next_edge_id(graph), second.id, first.id, structural=True))
            else:
                context.diagnose("error", "AMBIGUOUS_CONNECTOR_PAIR", "A connector pair has ambiguous entry and exit arrows.", "One circle receives flow; its matching circle sends flow onward.", [first.id, second.id])
        starts = [node.id for node in graph.nodes if node.type == "terminal" and terminal_role(node.text) == "start"]
        graph.start_id = starts[0] if len(starts) == 1 else None
        graph.end_ids = [node.id for node in graph.nodes if node.type == "terminal" and terminal_role(node.text) == "end"]
        return context


def run(context: PipelineContext) -> PipelineContext:
    return GraphService().run(context)
