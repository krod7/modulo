"""Scale-aware connector geometry: ink extraction, endpoints and arrowheads.

These helpers are deliberately independent of node semantics and text. They
answer three purely geometric questions for GraphService:

* which pixels are connector ink (relative to the page's own background),
* whether a stroke *terminates* at a node or merely passes beside it, and
* whether a stroke end carries an arrowhead.

Every threshold is expressed in multiples of the drawing's measured stroke
width so the same rules apply to thin vector exports, thick screenshots and
up-scaled low-resolution images.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def background_level(grayscale: np.ndarray) -> int:
    """Return the page background tone (the most common bright gray level)."""
    histogram = np.bincount(grayscale.ravel(), minlength=256)
    bright = histogram[128:]
    return int(128 + bright.argmax()) if bright.sum() else 255


def ink_mask(grayscale: np.ndarray, margin: int = 12, ceiling: int = 245) -> np.ndarray:
    """Pixels darker than the page background by a clear margin.

    A fixed ``gray < 245`` treats tinted/off-white backgrounds as ink. Anchoring
    the cut to the measured background keeps pale connectors on white pages and
    still rejects a lavender or gray page tone.
    """
    threshold = min(ceiling, background_level(grayscale) - margin)
    return np.where(grayscale < threshold, 255, 0).astype(np.uint8)


def connector_stroke_width(mask: np.ndarray, maximum: int = 40) -> int:
    """Typical connector thickness, with one vote per connector component.

    A pixel-weighted run-length mode is dominated by whichever single line is
    longest; a page frame or screenshot edge would then define the scale. Here
    every component votes with its own modal run length, and components that
    touch the image border (frames, scan edges) are ignored.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    height, width = mask.shape[:2]
    votes = []
    for index in range(1, count):
        x, y, w, h, area = map(int, stats[index])
        if area < 20 or max(w, h) < 18 or x == 0 or y == 0 or x + w >= width or y + h >= height:
            continue
        component = (labels[y:y + h, x:x + w] == index).astype(np.int8)
        runs = []
        for grid in (component, component.T):
            delta = np.diff(np.pad(grid, ((0, 0), (1, 1))), axis=1)
            runs.append(np.where(delta == -1)[1] - np.where(delta == 1)[1])
        lengths = np.concatenate(runs)
        lengths = lengths[(lengths >= 1) & (lengths <= maximum)]
        if lengths.size:
            votes.append(int(np.bincount(lengths).argmax()))
    return int(np.median(votes)) if votes else 3


def _local_stroke(points: np.ndarray, point: tuple[int, int], radius: float,
                  head_mask: np.ndarray | None, stroke_width: float = 3.0) -> np.ndarray:
    """Pixels of this stroke near ``point``, recovered from raw ink if available.

    Node cleanup erases a band around every shape, which often removes most of
    an arrowhead that touches the node. Raw ink connected to the surviving
    stroke pixels restores it without admitting unrelated nearby strokes.
    """
    p = np.asarray(point, dtype=float)
    local = points[np.linalg.norm(points.astype(float) - p, axis=1) <= radius]
    if head_mask is None or len(local) == 0:
        return local
    height, width = head_mask.shape[:2]
    r = int(math.ceil(radius))
    x0, y0 = max(0, int(p[0]) - r), max(0, int(p[1]) - r)
    x1, y1 = min(width, int(p[0]) + r + 1), min(height, int(p[1]) + r + 1)
    window = head_mask[y0:y1, x0:x1]
    if window.size == 0:
        return local
    seeds = np.zeros_like(window)
    inside = local[(local[:, 0] >= x0) & (local[:, 0] < x1) & (local[:, 1] >= y0) & (local[:, 1] < y1)]
    seeds[inside[:, 1] - y0, inside[:, 0] - x0] = 255
    combined = cv2.bitwise_or(window, seeds)
    count, labels = cv2.connectedComponents(combined, connectivity=8)
    keep = set(np.unique(labels[inside[:, 1] - y0, inside[:, 0] - x0])) - {0}
    # Arrowhead barbs whose meeting point lies inside the node-cleanup band are
    # detached from the shaft; admit small pieces lying wholly near the end.
    near, extent = 3.0 * stroke_width + 6.0, 6.0 * stroke_width + 12.0
    centre = p - np.array([x0, y0])
    for index in range(1, count):
        if index in keep:
            continue
        ys_, xs_ = np.where(labels == index)
        offsets = np.hypot(xs_ - centre[0], ys_ - centre[1])
        if offsets.min() <= near and offsets.max() <= extent:
            keep.add(index)
    ys, xs = np.where(np.isin(labels, list(keep)))
    recovered = np.column_stack((xs + x0, ys + y0))
    distance = np.linalg.norm(recovered.astype(float) - p, axis=1)
    return recovered[distance <= radius]


def endpoint_has_arrowhead(points: np.ndarray, point: tuple[int, int], stroke_width: float,
                           head_mask: np.ndarray | None = None) -> bool:
    """Decide whether the stroke end at ``point`` carries an arrowhead.

    The stroke is profiled across its own axis, moving away from the end. An
    arrowhead (filled triangle, open V, or one truncated by node cleanup) is a
    widening that

    * is clearly wider than the line (>= ~2.2 stroke widths),
    * starts at the end itself rather than somewhere along the shaft,
    * is symmetric about the shaft (a corner or a branch leaves to one side),
    * is laterally bounded (a crossing line or T-junction is not), and
    * lies within half of the connector's length, so the arrowhead at the
      *other* end of a short connector is never attributed to this end.
    """
    if points is None or len(points) < 6:
        return False
    sw = max(1.0, float(stroke_width))
    p = np.asarray(point, dtype=float)
    radius = 12.0 * sw + 24.0
    local = _local_stroke(points, point, radius, head_mask, sw)
    if len(local) < 6:
        return False
    delta = local.astype(float) - p
    # The axis comes from the stroke's own pixels, which lie only on the
    # connector side of the end (recovered ink may surround the end point).
    own = points.astype(float) - p
    own_dist = np.linalg.norm(own, axis=1)
    axis_sample = own[(own_dist >= 2.0 * sw) & (own_dist <= 6.0 * sw + 12.0)]
    if len(axis_sample) < 3:
        return False
    direction = axis_sample.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm < 1.0:
        return False
    direction /= norm
    # Ink recovered from the raw image can extend behind the cut stroke end,
    # towards the node; profile from the true extremity of this stroke end.
    # Follow only ink that continues the stroke contiguously along its axis;
    # residue of the node's outline across a gap must not move the anchor.
    behind = delta @ direction
    core = behind[np.abs(delta @ np.array([-direction[1], direction[0]])) <= sw + 2.0]
    present = set(np.floor(core[core < 0]).astype(int).tolist())
    extent = 0
    while any((extent - k) in present for k in (1, 2, 3)):
        extent -= 1
    if extent < -sw:
        p = p + direction * float(extent)
        delta = local.astype(float) - p
    normal = np.array([-direction[1], direction[0]])
    # Connector length along this axis, measured on the whole component.
    length = float(((points.astype(float) - p) @ direction).max())
    window = min(10.0 * sw + 16.0, .5 * length)
    if window < 2.0 * sw + 4.0:
        return False
    longitudinal, transverse = delta @ direction, delta @ normal
    lateral = 5.0 * sw + 14.0
    step = max(1.0, sw / 2.0)
    widths, centres, offsets, reaches = [], [], [], []
    t = -sw  # include pixels a little behind the end point (rounded tips)
    while t < window:
        selected = transverse[(longitudinal >= t) & (longitudinal < t + step) & (np.abs(transverse) <= lateral)]
        if len(selected):
            widths.append(float(selected.max() - selected.min()) + 1.0)
            centres.append(float(selected.max() + selected.min()) / 2.0)
            offsets.append(t)
            reaches.append(float(max(selected.max(), -selected.min())))
        t += step
    # Stop at the first corner, crossing line or junction (a slab reaching the
    # lateral window on either side): what lies beyond is another structure,
    # whereas an arrowhead stays well inside the window.
    clipped = next((i for i, r in enumerate(reaches) if r >= lateral - sw), None)
    if clipped is not None:
        widths, centres, offsets = widths[:clipped], centres[:clipped], offsets[:clipped]
    if len(widths) < 3:
        return False
    widths_arr = np.asarray(widths)
    peak = int(widths_arr.argmax())
    head_width = float(widths_arr[peak])
    if head_width < max(2.2 * sw, sw + 6.0):
        return False
    # A clipped profile reaches the lateral window on both sides: a crossing
    # or perpendicular line, never a bounded arrowhead.
    if head_width >= 2.0 * lateral - sw:
        return False
    # Shaft reference after the widening (if the window extends that far).
    # Shaft reference just behind the widening; farther along, a bend in the
    # connector would shift its centre and is not evidence about this end.
    after = [(w, c) for w, c, o in zip(widths, centres, offsets)
             if offsets[peak] + sw < o <= offsets[peak] + 5.0 * sw + 8.0]
    shaft_centre = float(np.median([c for _, c in after])) if after else 0.0
    if abs(centres[peak] - shaft_centre) > .3 * head_width + sw / 2.0:
        return False
    if after and min(w for w, _ in after) > .6 * head_width:
        return False  # no return to a narrow shaft: a blob or a junction
    # An arrowhead has length comparable to its width; a one-slab speck of
    # outline residue or JPEG noise at the boundary does not.
    wide_span = sum(step for w in widths if w >= .5 * head_width)
    if wide_span <= sw or wide_span < .25 * head_width:
        return False
    first_wide = next(o for w, o in zip(widths, offsets) if w >= .6 * head_width)
    if first_wide > 2.0 * sw + .6 * head_width:
        return False
    # From an arrowhead's tip the profile widens gradually; from the *tail* of
    # a short connector it is a narrow shaft plateau followed by an abrupt jump
    # at the far arrowhead's base. Reject ends that start with a long plateau.
    narrow = 1.4 * sw + 2.0
    plateau_end = offsets[0]
    plateau_widths = []
    for w, o in zip(widths, offsets):
        if w > narrow:
            break
        plateau_end = o + step
        plateau_widths.append(w)
    plateau = plateau_end - offsets[0]
    # A sharp tip also starts narrow, but its width grows steadily; a shaft
    # plateau keeps a constant width.
    if plateau_widths and max(plateau_widths) - min(plateau_widths) > .35 * sw + 2.0:
        trailing = [w for w in plateau_widths if w >= max(plateau_widths) - (.35 * sw + 2.0)]
        plateau = len(trailing) * step if len(trailing) < len(plateau_widths) else plateau
    rise = max(step, offsets[peak] - offsets[0])
    return plateau <= max(1.5 * sw, .35 * rise)


def stroke_exits(band: np.ndarray, ring: np.ndarray, outer: np.ndarray | None = None) -> int:
    """Count separate places where a stroke leaves a node's contact band.

    ``band`` holds the stroke pixels close to a node and ``ring`` the stroke
    pixels in a thin shell just beyond. A connector that *ends* at the node
    leaves the band once; a line that only runs past the node leaves it twice
    (and a junction three or more times). With ``outer`` (the shell's outer
    half), only pieces that continue all the way through the shell count.
    """
    if not np.any(ring):
        return 0
    touching = cv2.bitwise_and(ring, cv2.dilate(band, np.ones((5, 5), np.uint8)))
    count, labels = cv2.connectedComponents(cv2.bitwise_or(ring, touching), connectivity=8)
    exits = set(np.unique(labels[touching > 0])) - {0}
    if outer is not None:
        continuing = set(np.unique(labels[(outer > 0) & (ring > 0)])) - {0}
        exits &= continuing
    return len(exits)
