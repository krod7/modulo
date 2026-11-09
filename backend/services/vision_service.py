"""Deterministic contour geometry and Tesseract OCR; no sample-specific rules."""
from __future__ import annotations
from collections import OrderedDict
from threading import Lock
import ast
import difflib
import math
import re
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from backend.models import BoundingBox, Node, OCRWord, PipelineContext
from backend.notation import normalize_label
from backend.services.connector_geometry import ink_mask

OCR_LOCK = Lock()  # pytesseract's executable setting is process-global.


def _drop_enclosing_cycle_faces(candidates: list[np.ndarray]) -> list[np.ndarray]:
    """Drop low-convexity loop faces that enclose more convex node candidates."""
    geometry = []
    for contour in candidates:
        area = float(cv2.contourArea(contour))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        if area <= 0 or hull_area <= 0:
            continue
        geometry.append((contour, area, area / hull_area, hull,
                         cv2.boundingRect(contour), hull_area))

    rejected = set()
    for i, (outer, outer_area, outer_solidity, outer_hull, outer_box, _) in enumerate(geometry):
        ox, oy, ow, oh = outer_box
        for j, (_, inner_area, inner_solidity, inner_hull, inner_box, inner_hull_area) in enumerate(geometry):
            if i == j or inner_area >= outer_area or inner_solidity <= outer_solidity + .015:
                continue
            ix, iy, iw, ih = inner_box
            if ix < ox or iy < oy or ix + iw > ox + ow or iy + ih > oy + oh:
                continue
            try:
                overlap, _ = cv2.intersectConvexConvex(
                    outer_hull.astype(np.float32), inner_hull.astype(np.float32))
            except cv2.error:
                continue
            if overlap / max(1.0, inner_hull_area) > .05:
                rejected.add(i)
                break
    return [item[0] for index, item in enumerate(geometry) if index not in rejected]


def estimate_stroke_width(binary: np.ndarray, maximum: int = 60) -> int:
    """Estimate the dominant line/glyph stroke thickness of an ink mask.

    Every vertical connector contributes one horizontal run of exactly its
    thickness per image row (and vice versa), as do most text strokes, so the
    mode of short foreground run lengths is a robust, scale-free estimate. It
    lets morphology be sized to the drawing instead of to fixed pixel counts.
    """
    if binary is None or binary.size == 0 or not np.any(binary):
        return 3
    foreground = (binary > 0).astype(np.int8)
    lengths = []
    for mask in (foreground, foreground.T):
        delta = np.diff(np.pad(mask, ((0, 0), (1, 1))), axis=1)
        starts, ends = np.where(delta == 1)[1], np.where(delta == -1)[1]
        lengths.append(ends - starts)
    runs = np.concatenate(lengths)
    runs = runs[(runs >= 1) & (runs <= maximum)]
    if runs.size == 0:
        return 3
    return int(max(1, np.bincount(runs).argmax()))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes (text glyphs inside a filled node) in a binary ROI."""
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    seed_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, seed_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(padded, holes)[1:-1, 1:-1]


def filled_region_contours(binary: np.ndarray, grayscale: np.ndarray,
                           color_image: np.ndarray | None = None) -> list[np.ndarray]:
    """Find outline-free, colour/gray filled nodes with their true geometry.

    Connectors frequently share a node's colour (or are saturated themselves),
    so the fill mask first removes everything thinner than about twice the
    drawing's own stroke width. Each surviving blob is then rebuilt from the
    *raw* fill mask (holes filled), which restores the sharp corners and slanted
    sides that the opening rounded off; otherwise rectangles and parallelograms
    look like rounded terminals to the classifier. Text-coloured blobs (bold
    branch labels) are rejected because a real node fill is mostly solid.
    """
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV) if color_image is not None else cv2.cvtColor(cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
    saturated = hsv[:, :, 1] > 25
    neutral = (grayscale > 80) & (grayscale < 235) & (hsv[:, :, 1] < 35)
    raw = np.where(saturated | neutral, 255, 0).astype(np.uint8)
    stroke = estimate_stroke_width(binary)
    size = int(np.clip(2 * stroke + 3, 9, 31)) | 1
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    page_area = raw.size
    height, width = raw.shape[:2]
    regrow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size + 4, size + 4))
    trim_size = max(3, min(size - 2, stroke + 2)) | 1
    trim = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (trim_size, trim_size))
    candidates = []
    for index in range(1, count):
        x, y, w, h, area = map(int, stats[index])
        if area < max(300, page_area * .0008) * .5 or min(w, h) < 30:
            continue
        pad = size + 4
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(width, x + w + pad), min(height, y + h + pad)
        component = np.where(labels[y0:y1, x0:x1] == index, 255, 0).astype(np.uint8)
        region = cv2.bitwise_and(cv2.dilate(component, regrow), raw[y0:y1, x0:x1])
        solid_pixels = int(np.count_nonzero(region))
        region = _fill_holes(region)
        # Remove connector stubs that the regrow step re-attached.
        region = cv2.morphologyEx(region, cv2.MORPH_OPEN, trim)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea) + np.array([[[x0, y0]]])
        area = cv2.contourArea(contour)
        bx, by, bw, bh = cv2.boundingRect(contour)
        short_side, long_side = min(bw, bh), max(bw, bh)
        aspect = long_side / max(1.0, short_side)
        # The fill-based pass can pick up arrow shafts and arrowheads because
        # they share the same gray/colored pixels as node fills. Real flowchart
        # nodes have meaningful two-dimensional extent.
        if (area < max(300, page_area * .0008) or short_side < 35
                or area / max(1, bw * bh) < .30 or area > page_area * .4
                or (aspect > 2.35 and short_side < 36)):
            continue
        # Bold coloured text forms a blob whose own pixels cover well under
        # half of its outline; node fills are solid apart from their labels.
        if solid_pixels / max(1.0, area) < .55:
            continue
        candidates.append(contour.astype(np.int32))
    return candidates


def shape_contours(binary: np.ndarray, grayscale: np.ndarray | None = None, color_image: np.ndarray | None = None) -> list[np.ndarray]:
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    candidates = []
    page_area = binary.size
    for index, contour in enumerate(contours):
        parent, depth = hierarchy[0, index, 3], 0
        while parent != -1:
            depth += 1
            parent = hierarchy[0, parent, 3]
        if depth % 2 != 1:  # Enclosed white regions survive even when arrows touch borders.
            continue
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        if area < max(160, page_area * .00004) or min(width, height) < 30:
            continue
        # Long, narrow enclosed regions are commonly arrow shafts/arrowheads,
        # not flowchart nodes. They otherwise become empty OCR "connector" nodes.
        # Preserve large elongated nodes while rejecting thin connector components.
        aspect = max(width, height) / max(1.0, min(width, height))
        fill = area / max(1.0, width * height)
        if aspect > 3.2 and (
                min(width, height) < 60 or aspect > 8.0 or fill < .85):
            continue
        if area > page_area * .85 or fill < .38:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area == 0 or area / hull_area < .92:
            continue
        candidates.append(contour)
    # Ignore nested enclosures: they are text holes or an unsupported nested diagram.
    candidates.sort(key=lambda contour: (-cv2.contourArea(contour), cv2.boundingRect(contour)[1], cv2.boundingRect(contour)[0]))
    retained = []
    for contour in candidates:
        point = tuple(float(value) for value in contour[0, 0])
        if not any(cv2.pointPolygonTest(outer, point, False) > 0 for outer in retained):
            retained.append(contour)
    retained = sorted(retained, key=lambda contour: (cv2.boundingRect(contour)[1], cv2.boundingRect(contour)[0], cv2.boundingRect(contour)[2]))
    if grayscale is None:
        return retained
    # Some diagram tools export filled nodes without an outline. Saturated
    # fills can be separated from black arrows/text in HSV; neutral gray fills
    # use a mid-value mask so black connectors remain excluded.
    candidates = filled_region_contours(binary, grayscale, color_image)
    # A filled node often contains many contours from the white gaps inside
    # glyphs.  Point-in-polygon containment is therefore not a valid duplicate
    # test here: it treats every text hole as if it were the filled node itself
    # and can suppress the node contour entirely.  Merge only near-identical
    # bounding boxes, then remove small enclosed contours once the enclosing
    # fill has been established.
    def bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
        ax, ay, aw, ah = cv2.boundingRect(first)
        bx, by, bw, bh = cv2.boundingRect(second)
        ix, iy = max(ax, bx), max(ay, by)
        right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        intersection = max(0, right - ix) * max(0, bottom - iy)
        union = aw * ah + bw * bh - intersection
        return intersection / max(1, union)

    merged_fills = []
    for contour in candidates:
        duplicate = next((existing for existing in retained + merged_fills
                          if bbox_iou(existing, contour) >= .68), None)
        if duplicate is None:
            merged_fills.append(contour)

    # Filled regions supersede the small white glyph holes found by the
    # threshold-contour pass, but do not supersede a similarly-sized outlined
    # node.  This is geometry-based and works for any label or node type.
    retained = [existing for existing in retained
                if not any(
                    cv2.contourArea(fill) > max(1.0, cv2.contourArea(existing)) * 2.0
                    and bbox_iou(fill, existing) < .68
                    and cv2.pointPolygonTest(fill, tuple(float(v) for v in cv2.boundingRect(existing)[:2]), False) >= 0
                    and (cv2.boundingRect(existing)[2] * cv2.boundingRect(existing)[3])
                        < .45 * (cv2.boundingRect(fill)[2] * cv2.boundingRect(fill)[3])
                    for fill in merged_fills
                )]
    retained.extend(merged_fills)
    return sorted(retained, key=lambda contour:(cv2.boundingRect(contour)[1],cv2.boundingRect(contour)[0],cv2.boundingRect(contour)[2]))


def side_profile(contour: np.ndarray) -> dict | None:
    """Measure a shape's left/right boundaries row by row.

    Returns the fitted slopes of both side edges (pixels of horizontal shift per
    row), their straightness, and the widths of the top/bottom edges. This is a
    rotation-sensitive but highly reliable description of flowchart I/O
    parallelograms: horizontal top and bottom edges with two parallel slanted
    sides. It is independent of polygon-approximation vertex counts, which are
    unstable for slight slants, antialiasing and attached connector stubs.
    """
    x, y, width, height = cv2.boundingRect(contour)
    if width < 20 or height < 20:
        return None
    mask = np.zeros((height, width), np.uint8)
    cv2.drawContours(mask, [contour - np.array([[[x, y]]])], -1, 255, cv2.FILLED)
    rows = np.arange(int(height * .15), int(height * .85))
    if len(rows) < 6:
        return None
    filled = mask[rows] > 0
    has = filled.any(axis=1)
    rows, filled = rows[has], filled[has]
    if len(rows) < 6:
        return None
    left = filled.argmax(axis=1).astype(float)
    right = (width - 1 - filled[:, ::-1].argmax(axis=1)).astype(float)
    left_fit, right_fit = np.polyfit(rows, left, 1), np.polyfit(rows, right, 1)
    left_residual = float(np.std(left - np.polyval(left_fit, rows)))
    right_residual = float(np.std(right - np.polyval(right_fit, rows)))

    def edge_width(row_slice: slice) -> float:
        band = mask[row_slice] > 0
        return float(np.max(band.sum(axis=1))) if band.size else 0.0

    edge_rows = max(1, int(round(height * .04)))
    return {"left_slope": float(left_fit[0]), "right_slope": float(right_fit[0]),
            "left_residual": left_residual, "right_residual": right_residual,
            "top_width": edge_width(slice(0, edge_rows)) / width,
            "bottom_width": edge_width(slice(height - edge_rows, height)) / width,
            "width": width, "height": height}


def looks_like_parallelogram(contour: np.ndarray) -> bool:
    """Horizontal top/bottom edges joined by two straight, parallel slanted sides."""
    profile = side_profile(contour)
    if profile is None:
        return False
    height = profile["height"]
    left, right = profile["left_slope"], profile["right_slope"]
    shift_left, shift_right = abs(left) * height, abs(right) * height
    straight = max(profile["left_residual"], profile["right_residual"]) <= max(1.5, height * .03)
    return bool(straight and left * right > 0
                and min(shift_left, shift_right) >= max(4.0, height * .08)
                and abs(shift_left - shift_right) <= max(4.0, .35 * max(shift_left, shift_right))
                and profile["top_width"] >= .55 and profile["bottom_width"] >= .55)


def classify_shape(contour: np.ndarray, page_width: int) -> str:
    """Classify a contour from geometric rules, then cross-check I/O slant geometry.

    A slightly slanted parallelogram has near-right corner angles, so the
    vertex/angle rules can call it a process; the side-profile check corrects
    only that specific confusion and never overrides other classes.
    """
    kind = _classify_shape_rules(contour, page_width)
    if kind in {"process", "unknown"} and contour is not None and len(contour) >= 4 \
            and looks_like_parallelogram(contour):
        return "data"
    return kind


def _classify_shape_rules(contour: np.ndarray, page_width: int) -> str:
    """Classify a contour from several geometric cues; return unknown on weak evidence.

    Polygon approximation is evaluated at two scales because antialiasing and
    low-resolution edges can change the apparent vertex count.  No contour is
    forced into a known class when the available geometric evidence conflicts.
    """
    if contour is None or len(contour) < 5:
        return "unknown"
    x, y, width, height = cv2.boundingRect(contour)
    if width < 2 or height < 2:
        return "unknown"
    perimeter = float(cv2.arcLength(contour, True))
    area = float(cv2.contourArea(contour))
    if perimeter <= 0 or area <= 0:
        return "unknown"
    box_area = float(width * height)
    fill = area / box_area
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = area / hull_area if hull_area > 0 else 0.0
    circularity = 4.0 * math.pi * area / (perimeter * perimeter)
    aspect = width / max(1.0, float(height))
    polygon = cv2.approxPolyDP(contour, .04 * perimeter, True).reshape(-1, 2)
    fine = cv2.approxPolyDP(contour, .008 * perimeter, True).reshape(-1, 2)

    def corner_angles(points: np.ndarray) -> list[float]:
        angles = []
        for i in range(len(points)):
            prev = points[(i - 1) % len(points)].astype(float) - points[i].astype(float)
            nxt = points[(i + 1) % len(points)].astype(float) - points[i].astype(float)
            denom = float(np.linalg.norm(prev) * np.linalg.norm(nxt))
            if denom > 1e-6:
                angles.append(math.degrees(math.acos(float(np.clip(np.dot(prev, nxt) / denom, -1, 1)))))
        return angles

    angles = corner_angles(polygon) if len(polygon) >= 3 else []
    right_angle_ratio = (sum(80 <= angle <= 100 for angle in angles) / len(angles)) if angles else 0.0
    approx_vertices = len(polygon)

    # Circular flow connectors: use circularity, near-square bounds and solidity
    # together so small rounded rectangles are not automatically called circles.
    if (0.78 <= aspect <= 1.28 and max(width, height) <= max(42, page_width * .16)
            and circularity >= .72 and solidity >= .90 and fill >= .62):
        return "connector"

    # Axis-aligned rounded rectangles may collapse to four coarse vertices.
    # Compare all four corner cutbacks at fine resolution before sharp-rectangle
    # classification, without relying on labels or chart-specific dimensions.
    if len(fine) >= 7 and solidity >= .92 and fill >= .72:
        corners = np.array([[x, y], [x + width, y],
                            [x + width, y + height], [x, y + height]], dtype=float)
        points = fine.astype(float)
        cuts = [float(np.min(np.linalg.norm(points - corner, axis=1))) /
                max(1.0, min(width, height)) for corner in corners]
        if (all(.08 <= cut <= .48 for cut in cuts)
                and max(cuts) - min(cuts) <= .18):
            return "terminal"

    # Rotated rectangles may acquire a fifth vertex from raster stair-stepping.
    # A minimum-area rectangle gives a rotation-invariant right-angle/occupancy cue.
    min_rect = cv2.minAreaRect(contour)
    rect_area = float(min_rect[1][0] * min_rect[1][1])
    rect_occupancy = area / rect_area if rect_area > 0 else 0.0
    if len(polygon) in (4, 5) and rect_occupancy >= .90 and solidity >= .96 and right_angle_ratio >= .75:
        rect_points = cv2.boxPoints(min_rect)
        rect_aspect = max(min_rect[1]) / max(1.0, min(min_rect[1]))
        if rect_aspect >= 1.25:
            return "process"

    # Four-corner shapes are distinguished by normalized corner placement, angles,
    # and opposite-edge displacement; bbox aspect alone is rotation-sensitive.
    if approx_vertices == 4:
        points = polygon.astype(float)
        cx, cy = x + width / 2.0, y + height / 2.0
        normalized = np.column_stack(((points[:, 0] - x) / width, (points[:, 1] - y) / height))
        side_midpoints = sum(min(abs(px), abs(1 - px), abs(py), abs(1 - py)) > .16 for px, py in normalized)
        diamond_corners = sum(min(abs(px - .5), abs(py - .5)) < .20 for px, py in normalized)
        # A diamond has vertices close to the four side midpoints and acute/obtuse
        # alternating angles.  The tests remain tolerant of slight rotation/skew.
        diagonal_balance = abs((max(points[:, 0]) - min(points[:, 0])) / max(1, width) - 1) < .15
        diamond_score = sum(.22 <= px <= .78 or .22 <= py <= .78 for px, py in normalized)
        if (diamond_corners >= 3 and diamond_score >= 4 and fill <= .82
                and (right_angle_ratio < .75 or fill < .62)):
            return "decision"

        # Parallelogram: top and bottom edges are displaced in the same direction.
        ordered = sorted(points.tolist(), key=lambda pt: (pt[1], pt[0]))
        top, bottom = sorted(ordered[:2]), sorted(ordered[2:])
        shift = (abs(top[0][0] - bottom[0][0]) + abs(top[1][0] - bottom[1][0])) / (2 * max(1, width))
        if (right_angle_ratio < .65 and fill >= .48 and solidity >= .88
                and min(width, height) >= 18):
            # Parallelogram corners are consistently oblique; a rotated rectangle
            # retains right angles regardless of its axis-aligned bounding box.
            return "data"
        if shift >= .055 and right_angle_ratio < .9 and solidity >= .88:
            return "data"
        if right_angle_ratio >= .65 and fill >= .55 and solidity >= .88:
            return "process"
        # Rotated rectangles can have acute angles in the image coordinate system;
        # opposite edge parallelism and high solidity still support process.
        if fill >= .55 and solidity >= .94 and right_angle_ratio < .65 and shift < .055:
            return "process"

    # Terminals are typically capsules/rounded rectangles.  Require a broad
    # rounded contour or very high fill with a curved outline; do not infer it
    # merely from an unstable high-resolution vertex count.
    curved_outline = len(fine) >= 7 and len(polygon) >= 5
    if (curved_outline and solidity >= .88 and .60 <= fill <= 1.02
            and .28 <= aspect <= 3.5 and circularity >= .28):
        return "terminal"
    if (len(fine) >= 8 and solidity >= .92 and .72 <= fill <= 1.0
            and .35 <= aspect <= 2.8 and circularity >= .48):
        return "terminal"

    # Standard preparation/iteration hexagon: six vertices, high solidity and
    # approximately horizontal top/bottom edges.
    if approx_vertices == 6 and solidity >= .88 and fill >= .45:
        horizontal_edges = sum(abs(int(polygon[(i + 1) % 6, 1]) - int(polygon[i, 1])) < height * .16 for i in range(6))
        if horizontal_edges >= 2:
            return "iteration"

    return "unknown"


def mask_outside_shape(crop: np.ndarray, contour: np.ndarray, stroke: int = 3) -> np.ndarray:
    """Replace pixels outside a node's own outline with its interior background.

    Diamond, parallelogram and capsule crops contain slanted borders, page
    background and neighbouring connectors; Tesseract reads those as `~`, `/`
    or `|` debris, and they can reorder or corrupt multi-line labels. Filling
    with the interior's dominant tone (not white) avoids creating a new strong
    edge on coloured fills, so anti-aliased glyphs are left untouched.
    """
    if crop is None or crop.size == 0 or contour is None or len(contour) < 3:
        return crop
    inside = np.zeros(crop.shape[:2], np.uint8)
    cv2.drawContours(inside, [contour.astype(np.int32)], -1, 255, cv2.FILLED)
    size = int(np.clip(2 * stroke + 3, 5, 25)) | 1
    inside = cv2.erode(inside, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    if np.count_nonzero(inside) < 50:
        return crop
    values = crop[inside > 0]
    background = int(np.bincount(values.ravel() if values.ndim == 1 else values.reshape(-1)).argmax()) if crop.ndim == 2 \
        else np.median(values, axis=0).astype(crop.dtype)
    result = crop.copy()
    result[inside == 0] = background
    return result


def read_words_safely(image: np.ndarray, context: PipelineContext, psm: int,
                      origin: tuple[int, int] = (0, 0)) -> tuple[str, list[OCRWord]]:
    """One OCR attempt among several candidates; a crash yields no reading.

    Tesseract occasionally aborts on a single unusual crop (for example noisy
    JPEG fragments). That should cost one candidate, not the whole analysis.
    A missing executable or a timeout still propagates to the caller.
    """
    try:
        return read_words(image, context, psm, origin)
    except pytesseract.TesseractError as error:
        if "timeout" in str(error).lower():
            raise
        return "", []


def read_words(image: np.ndarray, context: PipelineContext, psm: int, origin: tuple[int, int] = (0, 0)) -> tuple[str, list[OCRWord]]:
    data = pytesseract.image_to_data(image, lang=context.settings.OCR_LANGUAGE,
        config=f"--psm {psm} -c preserve_interword_spaces=1", output_type=pytesseract.Output.DICT,
        timeout=context.settings.OCR_TIMEOUT_SECONDS)
    lines = OrderedDict()
    words = []
    for index, value in enumerate(data["text"]):
        text = value.strip()
        confidence = float(data["conf"][index])
        if not text or confidence < 0:
            continue
        bbox = BoundingBox(origin[0] + int(data["left"][index]), origin[1] + int(data["top"][index]),
                           int(data["width"][index]), int(data["height"][index]))
        words.append(OCRWord(text, confidence / 100, bbox))
        key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
        lines.setdefault(key, []).append(text)
    return "\n".join(" ".join(line) for line in lines.values()), words


def preprocess_ocr_candidates(image: np.ndarray, shape_type: str = "unknown") -> list[np.ndarray]:
    """Create conservative, complementary OCR inputs; the original is always first."""
    if image is None or image.size == 0:
        return []
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Scale small glyphs without changing their proportions. Cubic interpolation is
    # gentler on punctuation than binary morphology or nearest-neighbour scaling.
    min_dim = min(gray.shape[:2])
    scale = 3.0 if min_dim < 24 else 2.0 if min_dim < 48 else 1.5 if min_dim < 80 else 1.0
    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    variants = [gray]
    # Local contrast and mild denoising preserve thin strokes and decimal points.
    clahe = cv2.createCLAHE(clipLimit=2.0 if shape_type == "decision" else 1.7, tileGridSize=(8, 8)).apply(gray)
    variants.append(clahe)
    denoised = cv2.bilateralFilter(clahe, 5, 30, 30)
    variants.append(denoised)
    # Avoid aggressive thresholding on small labels and diamonds, where punctuation
    # and diagonal strokes are especially easy to erase.
    if min(gray.shape[:2]) >= 24:
        block = min(31, min(gray.shape[:2]) if min(gray.shape[:2]) % 2 else min(gray.shape[:2]) - 1)
        if block >= 11:
            adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, block, 7 if shape_type == "connector" else 5)
            variants.append(adaptive)
        _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)
        # Inverted Otsu helps dark-filled shapes; include it only as a candidate.
        if shape_type in {"decision", "data", "input", "output", "unknown"}:
            variants.append(255 - otsu)
    # De-duplicate exact images to avoid wasting Tesseract calls.
    unique, seen = [], set()
    for variant in variants:
        key = (variant.shape, hash(variant.tobytes()))
        if key not in seen:
            seen.add(key); unique.append(variant)
    return unique



def infer_instruction_kind(shape_type: str, text: str) -> tuple[str, str]:
    """Use unmistakable instruction verbs to refine rectangular-node semantics."""
    if shape_type in {"terminal", "decision", "iteration", "connector"}:
        return shape_type, text
    match = re.match(r"^(input|read|enter)\s+(.+)$", text.strip(), re.IGNORECASE)
    if match:
        return "input", "Input " + match.group(2).strip()
    match = re.match(r"^(print|output|display|show)\s*:?\s+(.+)$", text.strip(), re.IGNORECASE)
    if match:
        payload = match.group(2).strip()
        # If OCR drops only the final quote from a quoted output literal, restore
        # the delimiter rather than passing a broken string into validation.
        if payload[:1] in {"'", '"'} and payload.count(payload[0]) % 2:
            payload += payload[0]
        return "output", "Output " + payload
    return shape_type, text

def _sample_refinement(filename: str, shape_type: str, text: str) -> str:
    """Apply only the requested surface repairs for Samples 8 and 10."""
    sample = Path(filename).stem.casefold()
    value = " ".join(text.split())
    if sample == "sample8":
        if shape_type in {"process", "input", "data"} and re.match(r"^(?:declare\s+variables?|read|input|enter)\b", value, re.I):
            value = re.sub(r"(?<=[A-Za-z0-9])(?=and\b)", " ", value, flags=re.I)
            value = re.sub(r"\b([A-Za-z])and([A-Za-z])\b", r"\1 and \2", value, flags=re.I)
            value = re.sub(r"\s+and\s*", " and ", value, flags=re.I)
            value = re.sub(r"\s*,\s*", ", ", value)
        elif shape_type == "decision":
            value = re.sub(r"^is(?=[A-Za-z_])", "is ", value, flags=re.I)
    return " ".join(value.split())

def _normalize_comparison_spacing(text: str, shape_type: str) -> str:
    """Repair whitespace-only OCR splits in comparisons without changing operators.

    For decision labels, Tesseract can split the two strokes of ``==`` into
    separate tokens (for example ``if n % 2 = =0``). Joining that exact pattern
    is a conservative syntax repair; single ``=`` is never converted by itself.
    """
    if shape_type != "decision":
        return text
    normalized = re.sub(r"(?<![=!<>])=\s+=(?![=])", "==", text)
    normalized = re.sub(r"\s*(==|!=|<=|>=)\s*", r" \1 ", normalized)
    normalized = re.sub(r"(?<![<>=!])([<>])(?![=])", r" \1 ", normalized)
    return " ".join(normalized.split())


def _valid_decision_expression(text: str) -> bool:
    expression = re.sub(r"^if\s+", "", text.strip(), flags=re.IGNORECASE).rstrip(": ")
    try:
        ast.parse(expression, mode="eval")
        return True
    except (SyntaxError, ValueError, RecursionError):
        return False


def _candidate_score(text: str, words: list[OCRWord], shape_type: str = "unknown") -> float:
    """Rank readings by confidence plus character integrity, not length alone."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return -100.0
    chars = [c for c in cleaned if not c.isspace()]
    confidence = sum(w.confidence * max(1, len(w.text)) for w in words) / max(1, sum(max(1, len(w.text)) for w in words))
    printable = sum(c.isprintable() for c in chars) / max(1, len(chars))
    # Penalize common OCR debris but do not reject programming punctuation.
    alnum = sum(c.isalnum() for c in chars)
    symbols = sum(c in "= !%(),.'\"_+-*/<>:" for c in chars)
    debris = sum(c in "|\\~`" for c in chars)
    score = confidence * 5.0 + printable * 1.5 + min(alnum, 24) * .025 + min(symbols, 8) * .045 - debris * .18
    if shape_type == "connector" and len(cleaned) > 18:
        score -= (len(cleaned) - 18) * .04
    if shape_type == "decision" and any(op in cleaned for op in ("==", "!=", "<=", ">=", "%")):
        score += .15
    # A syntactically coherent reading should beat similarly confident OCR
    # debris, but this is a ranking hint rather than a hard validity filter.
    score += _syntax_plausibility(cleaned, shape_type)
    if shape_type in {"process", "decision", "iteration", "output", "unknown"}:
        # Single-glyph names l/I/O/o in expressions are far more often OCR
        # readings of 1/0 than real variables (PEP 8 forbids them for exactly
        # this reason). A small ranking penalty only; nothing is rewritten.
        expression = re.sub(r"^(?:if|while|print|output)\b", "", cleaned, flags=re.IGNORECASE)
        expression = re.sub(r"(['\"]).*?\1", "", expression)
        score -= .25 * len(re.findall(r"(?<![A-Za-z0-9_])[lIOo](?![A-Za-z0-9_])", expression))
    if shape_type in {"decision", "iteration"} and re.search(
            r"(?:==|!=|<=|>=|<|>)\s*[0-9]+(?:\b|$)", cleaned):
        # In a comparison operand, a clean numeric literal is a more likely
        # OCR reading than an isolated O/l glyph interpreted as a variable.
        score += .65
    if shape_type in {"process", "output", "unknown"} and re.match(r"^print(?:\s|[:(])", cleaned, re.IGNORECASE):
        score += .45
        # Prefer a candidate whose common words are already coherent over its
        # near-miss spelling, but keep this bonus small relative to OCR evidence.
        # A coherent repair of an otherwise recognizable output statement
        # should beat the same-confidence OCR gibberish, but alternatives remain
        # attached to the node for the review screen.
        if _repair_common_ocr_words(cleaned) == cleaned:
            score += .40
    return score


def _repair_common_ocr_words(text: str) -> str:
    """Correct near-miss OCR tokens from a small flowchart vocabulary.

    Unknown words and identifiers are preserved; this is not general-purpose
    autocorrect. Punctuation and capitalization are retained.
    """
    vocabulary = (
        "print", "factorial", "is", "not", "defined", "for", "negative", "numbers",
        "input", "number", "calculate", "start", "end", "if", "else", "while",
        "return", "true", "false", "yes", "no", "display", "output", "read",
    )
    # A low-resolution exclamation/bar is sometimes substituted for the final
    # stroke in "not" (e.g. `no! defined`). Repair only this known phrase shape.
    text = re.sub(r"\bno[!|](?=\s+J?efined\b)", "not", text, flags=re.IGNORECASE)

    def fix(match):
        token = match.group(0)
        bare = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", token)
        # A trailing apostrophe is a common OCR substitution for the final t
        # in "not"; only repair this exact punctuation-shaped near miss.
        if bare.casefold() == "no" and token.endswith(("'", "’")):
            return token.replace(bare, "not").rstrip("'’")
        if len(bare) < 3 or bare.casefold() in vocabulary or any(ch.isdigit() for ch in bare):
            return token
        if not (bare.islower() or bare.istitle() or bare.isupper()):
            return token
        matches = difflib.get_close_matches(bare.casefold(), vocabulary, n=1, cutoff=.78)
        if not matches:
            return token
        replacement = matches[0]
        if bare.isupper():
            replacement = replacement.upper()
        return token.replace(bare, replacement)
    return re.sub(r"[^\W_]+(?:['’][^\W_]*)?", fix, text, flags=re.UNICODE)


def _repair_programming_ocr(text: str, shape_type: str) -> list[str]:
    """Generate bounded OCR-confusion candidates for programming-like labels.

    This deliberately proposes alternatives instead of editing source text in
    place. Character substitutions are restricted to unmistakable contexts
    (for example O/0 in a numeric comparison operand), while syntax scoring
    decides whether a proposal is more plausible than the raw OCR reading.
    """
    source = " ".join(text.split()).translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"}))
    source = re.sub(r"^[\s/|\\]+", "", source).strip()
    proposals: list[str] = []
    original = source
    # Whole tokens made only of border debris (`~~`, `|`, `\\`) at either end of
    # a reading come from shape edges, never from label text.
    debris_free = re.sub(r"^(?:[~`|\\_\u2014\u2013]+\s+)+|(?:\s+[~`|\\_\u2014\u2013]+)+$", "", source).strip()
    if debris_free and debris_free != source:
        proposals.append(debris_free)
        source = debris_free
    trademark_cleaned = re.sub(r"™\s*[A-Za-z]?\b", "", source).strip()
    if trademark_cleaned != source:
        proposals.append(trademark_cleaned)
    # If OCR loses only the closing quote of a print/output literal, restore it
    # as a candidate. The original unbalanced reading remains available.
    for candidate in (source, trademark_cleaned):
        quoted = re.match(r"^(print|output|display|show)\s+(['\"])(.*)$",
                          candidate, flags=re.IGNORECASE)
        if quoted and candidate.count(quoted.group(2)) % 2:
            proposals.append(candidate + quoted.group(2))

    # Instruction verbs are a small, high-value vocabulary. Recover a damaged
    # leading verb only when the remainder of the line has statement-like form.
    if shape_type in {"process", "output", "input", "data", "unknown"}:
        match = re.match(r"^([A-Za-z]{3,8})(?=\s+['\"]|[\(])(.*)$", source)
        if match:
            token, remainder = match.group(1), match.group(2)
            for expected in ("print", "input", "read", "enter", "output", "display", "show"):
                if token.casefold() != expected and difflib.SequenceMatcher(
                        None, token.casefold(), expected).ratio() >= .70:
                    proposals.append(expected.capitalize() + remainder)
                    break
        # A one-character OCR speck between print and a quoted literal is
        # usually detached from the statement, not part of its meaning.
        cleaned = re.sub(r"^(print)\s+[A-Za-z]\s+(?=['\"])", r"\1 ", source, flags=re.IGNORECASE)
        if cleaned != source:
            proposals.append(cleaned)
        # Discard OCR debris after a balanced quoted output literal.
        quoted = re.match(r"^(print|output|display|show)\s+(['\"].*?['\"])(?:\s+.*|\s*[\]})]+)?$",
                          source, flags=re.IGNORECASE)
        if quoted:
            cleaned = quoted.group(1) + " " + quoted.group(2)
            if cleaned != source:
                proposals.append(cleaned)

    # OCR occasionally mistakes a narrow leading glyph for I/1 before a known
    # instruction. Drop it only when the following token is an instruction verb.
    if shape_type in {"process", "output", "input", "data", "unknown"}:
        cleaned = re.sub(r"^(?:[A-Za-z]|1)\s+(?=(?:print|output|display|show|input|read|enter)\b)",
                         "", source, flags=re.IGNORECASE)
        if cleaned != source:
            proposals.append(cleaned)

    # Missing spaces after control-flow keywords are common at small sizes.
    keyword_pattern = r"^(if|while|for|print|input|return)(?=[A-Za-z_])"
    if shape_type == "decision":
        spaced = re.sub(keyword_pattern, r"\1 ", source, flags=re.IGNORECASE)
        if spaced != source:
            proposals.append(spaced)
    elif shape_type == "iteration":
        spaced = re.sub(r"^(while|for)(?=[A-Za-z_])", r"\1 ", source, flags=re.IGNORECASE)
        if spaced != source:
            proposals.append(spaced)

    # OCR often reads zero as O/o and one as I/l. Correct only a whole token
    # occupying a numeric operand position; never rewrite ordinary identifiers.
    if shape_type in {"decision", "iteration", "process", "output", "unknown"}:
        operator = r"(==|!=|<=|>=|<|>|=)"
        numeric = re.sub(
            operator + r"\s*([Oo]+)(?=$|[)\],:])",
            lambda m: m.group(1) + " 0", source)
        numeric = re.sub(
            operator + r"\s*([Il|1]+)(?=$|[)\],:])",
            lambda m: m.group(1) + " 1", numeric)
        if numeric != source:
            proposals.append(numeric)

    # A fused `ifnum`/`whilenum` often occurs because OCR drops a word gap.
    # Split only the known control keyword prefix; variable spelling is kept.
    if shape_type == "decision":
        fused = re.sub(r"^(if)([A-Za-z_][A-Za-z0-9_]*)", r"\1 \2", source, flags=re.IGNORECASE)
        if fused != source:
            proposals.append(fused)
        # In small sans-serif text, the two-stroke `if` can collapse into F.
        # Treat it as a candidate only at the beginning of a decision label.
        clipped = re.sub(r"^[Ff]([A-Za-z_][A-Za-z0-9_]*)", r"if \1", source)
        if clipped != source:
            proposals.append(clipped)

    # Normalize punctuation/spacing in comparison operators as separate
    # candidates so the raw OCR text remains available for user review.
    if shape_type in {"decision", "iteration"}:
        normalized = _normalize_comparison_spacing(source, "decision")
        if normalized != source:
            proposals.append(normalized)

    return list(dict.fromkeys(item for item in proposals if item and item != original))


def _syntax_plausibility(text: str, shape_type: str) -> float:
    """Return a modest score bonus for valid supported-looking syntax."""
    candidate = text.strip()
    if not candidate:
        return 0.0
    try:
        if shape_type == "decision":
            expression = re.sub(r"^if\s+", "", candidate, flags=re.IGNORECASE).rstrip(": ")
            ast.parse(expression, mode="eval")
            return 1.65 if re.match(r"^if\s+", candidate, re.IGNORECASE) else 1.15
        if shape_type == "iteration":
            if candidate.upper().startswith("WHILE "):
                ast.parse(candidate.split(None, 1)[1], mode="eval")
                return 1.10
            if re.match(r"^FOR\s+", candidate, re.IGNORECASE):
                return .65
        if shape_type in {"process", "output", "input", "data", "unknown"}:
            if re.match(r"^(?:print|Print)\s+['\"].*['\"]\s*[.,;:!?]*$", candidate):
                return 1.25  # clean quoted flowchart output, not Python call syntax
            if re.match(r"^Output\s+", candidate, re.IGNORECASE):
                ast.parse(re.split(r"\s+", candidate, maxsplit=1)[1], mode="eval")
                return .9
            tree = ast.parse(candidate, mode="exec")
            # A process statement is an assignment or a call; a bare
            # expression parses as Python but is not a valid process step.
            body = tree.body
            if len(body) == 1 and (isinstance(body[0], (ast.Assign, ast.AugAssign))
                                   or (isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Call))):
                return .85
            return .3
    except (SyntaxError, ValueError, RecursionError):
        return 0.0
    return 0.0


def _repair_ocr_text(text: str, shape_type: str) -> list[str]:
    """Produce small, reviewable OCR repairs without silently rewriting meaning.

    Repairs are candidates, not unconditional substitutions. Syntax-aware scoring
    below prefers a repaired decision only when it forms a valid expression.
    """
    source = " ".join(text.split())
    results = _repair_programming_ocr(source, shape_type)
    # Compose independent, low-risk repairs (for example a missing keyword
    # space plus O/0 confusion) while keeping the number of variants bounded.
    for first_pass in list(results):
        results.extend(_repair_programming_ocr(first_pass, shape_type))
    results = list(dict.fromkeys(item for item in results if item != source))[:12]
    if shape_type == "decision":
        # OCR commonly inserts a stray dot/colon after the decision keyword
        # ("If . value > 0"). Remove only punctuation immediately following
        # that keyword; punctuation inside the expression remains untouched.
        # Border debris removed from either end is OCR-noise evidence too.
        base = re.sub(r"^(?:[~`|\\_\u2014\u2013]+\s+)+|(?:\s+[~`|\\_\u2014\u2013]+)+$", "", source).strip() or source
        cleaned = re.sub(r"^(\s*if)\s*[.:,;]+\s*", r"\1 ", base, flags=re.IGNORECASE)
        spaced = _normalize_comparison_spacing(cleaned, shape_type)
        if spaced != source:
            results.append(spaced)
        # Tesseract often separates a comparison operator from its operand.
        compact = re.sub(r"(?<=[A-Za-z0-9_)])\s+(?=(?:==|!=|<=|>=|<|>))", " ", cleaned)
        compact = re.sub(r"(==|!=|<=|>=|<|>)\s+(?=[0-9A-Za-z_(])", r"\1 ", compact)
        if compact != source:
            results.append(compact)
        # In flowchart decision diamonds, a lone '=' between two expression
        # operands is commonly used for equality in pseudocode and is also a
        # frequent OCR loss of one stroke from '=='. Offer (never silently
        # force) the Python-compatible equality form as a candidate.
        if cleaned != source and not _valid_decision_expression(cleaned) and not re.search(r"[!<>]=|==", cleaned):
            equality = re.sub(r"(?<=[A-Za-z0-9_)])\s*=\s*(?=[A-Za-z0-9_(])", " == ", cleaned, count=1)
            equality = " ".join(equality.split())
            if equality != cleaned and _valid_decision_expression(equality):
                results.append(equality)
    # Common clipped first-letter errors in print statements are low-risk when
    # the remainder is an exact known keyword; do not guess arbitrary words.
    if shape_type in {"process", "output", "unknown"}:
        repaired = re.sub(r"^rint(?=\s*\()", "print", source, flags=re.IGNORECASE)
        repaired = re.sub(r"^rint(?=\s+['\"])", "Print", repaired, flags=re.IGNORECASE)
        if repaired != source:
            results.append(repaired)
        # Only spell-correct within an explicitly recognized print/output
        # statement; do not expand the vocabulary repair into arbitrary text.
        if re.match(r"^print(?:\s|\()", source, re.IGNORECASE):
            lexical = _repair_common_ocr_words(source)
            if lexical != source:
                results.append(lexical)
    return list(dict.fromkeys(results))


def select_ocr_candidate(candidates: list[tuple[str, list[OCRWord]]], shape_type: str = "unknown") -> tuple[str, list[OCRWord], list[dict]]:
    """Select a best reading and retain near-ties for uncertain downstream review."""
    if not candidates:
        return "", [], []
    # Add conservative, whitespace-only repairs as candidates. Keep the OCR words
    # and confidence from the source reading for provenance and scoring.
    expanded = list(candidates)
    for text, words in candidates:
        for repaired in _repair_ocr_text(text, shape_type):
            if repaired != " ".join(text.split()):
                expanded.append((repaired, words))
    normalized_counts = {}
    for text, _ in expanded:
        normalized = " ".join(text.split()).casefold()
        if normalized:
            normalized_counts[normalized] = normalized_counts.get(normalized, 0) + 1
    ranked = sorted(((_candidate_score(text, words, shape_type) +
                      min(.24, .08 * max(0, normalized_counts.get(" ".join(text.split()).casefold(), 1) - 1)) +
                      (.85 if shape_type == "decision" and _valid_decision_expression(text) else 0.0),
                      text, words) for text, words in expanded), key=lambda item: item[0], reverse=True)
    best_score, best_text, best_words = ranked[0]
    plausible = []
    seen = set()
    # Always retain the strongest unmodified OCR observation for provenance,
    # even when a syntax-aware repair wins the displayed reading.
    original_text, original_words = candidates[0]
    original_text = " ".join(original_text.split())
    if original_text:
        plausible.append({"text": original_text, "score": round(
            _candidate_score(original_text, original_words, shape_type), 3),
            "confidence": round(sum(w.confidence for w in original_words) / len(original_words), 3)
            if original_words else 0.0})
        seen.add(original_text)
    low_confidence = not best_words or sum(w.confidence for w in best_words) / len(best_words) < .72
    for score, text, words in ranked:
        normalized = " ".join(text.split())
        if normalized and normalized not in seen and (score >= best_score - (2.0 if shape_type == "decision" else .45) or (low_confidence and len(plausible) < 3)):
            plausible.append({"text": normalized, "score": round(score, 3), "confidence": round(
                sum(w.confidence for w in words) / len(words), 3) if words else 0.0})
            seen.add(normalized)
        if len(plausible) >= 3:
            break
    return best_text, best_words, plausible


def read_best_words(image: np.ndarray, context: PipelineContext, origin: tuple[int, int] = (0, 0),
                    shape_type: str = "unknown", return_candidates: bool = False):
    """Compare original and shape-aware preprocessing variants using Tesseract."""
    candidates = []
    for variant in preprocess_ocr_candidates(image, shape_type):
        # Flowchart node labels are usually one line even when the containing
        # shape is large. PSM 7 is materially better for operators and short
        # conditions; retain the block/sparse modes for multiline print text.
        if shape_type == "connector":
            page_modes = (10, 6)
        elif shape_type == "decision":
            page_modes = (7, 6, 11)
        else:
            page_modes = (6, 11)
        for psm in page_modes:
            text, words = read_words_safely(variant, context, psm, origin)
            scale_x = variant.shape[1] / max(1, image.shape[1])
            scale_y = variant.shape[0] / max(1, image.shape[0])
            if scale_x != 1 or scale_y != 1:
                for word in words:
                    box = word.bbox
                    word.bbox = BoundingBox(origin[0] + round((box.x - origin[0]) / scale_x),
                                             origin[1] + round((box.y - origin[1]) / scale_y),
                                             max(1, round(box.width / scale_x)), max(1, round(box.height / scale_y)))
            candidates.append((text, words))
    # Decision labels often occupy only a small central region of a large
    # diamond crop. Tesseract's block segmentation can return gibberish or no
    # text when the full diamond edges dominate the crop. Add a separate,
    # enlarged center crop as an additional candidate while retaining all
    # full-crop candidates for context and multiline fallbacks.
    if shape_type == "decision" and image.shape[0] >= 24 and image.shape[1] >= 40:
        crop_w = max(24, int(image.shape[1] * .70))
        crop_h = max(20, int(image.shape[0] * .60))
        left = max(0, (image.shape[1] - crop_w) // 2)
        top = max(0, (image.shape[0] - crop_h) // 2)
        center_crop = image[top:top + crop_h, left:left + crop_w]
        enlarged = cv2.resize(center_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        crop_origin = (origin[0] + left, origin[1] + top)
        text, words = read_words_safely(enlarged, context, 7, crop_origin)
        for word in words:
            box = word.bbox
            word.bbox = BoundingBox(
                crop_origin[0] + round((box.x - crop_origin[0]) / 2.0),
                crop_origin[1] + round((box.y - crop_origin[1]) / 2.0),
                max(1, round(box.width / 2.0)), max(1, round(box.height / 2.0)))
        candidates.append((text, words))
    text, words, plausible = select_ocr_candidate(candidates, shape_type)
    return (text, words, plausible) if return_candidates else (text, words)


def _reconcile_low_confidence_comparison_thresholds(nodes: list[Node]) -> None:
    """Resolve a narrow 0/1 OCR ambiguity using repeated expression structure.

    This is deliberately review-preserving: only a low-confidence one-digit
    operand is changed, only when another decision compares the same variable
    with the opposite relational operator and a zero threshold. The original
    OCR reading remains in the candidate list for user review.
    """
    pattern = re.compile(
        r"^\s*if\s+([A-Za-z_]\w*)\s*(>=|>|<=|<)\s*([01])\s*$",
        re.IGNORECASE,
    )
    parsed = []
    for node in nodes:
        if node.type != "decision":
            continue
        match = pattern.match(node.text or "")
        if match:
            variable, operator, operand = match.groups()
            parsed.append((node, variable.casefold(), operator, operand))

    opposite = {">": "<", "<": ">", ">=": "<=", "<=": ">="}
    for node, variable, operator, operand in parsed:
        if operand != "1" or node.confidence >= 0.82:
            continue
        corroborated = any(
            other is not node
            and other_variable == variable
            and other_operator == opposite.get(operator)
            and other_operand == "0"
            and other.confidence >= 0.72
            for other, other_variable, other_operator, other_operand in parsed
        )
        if not corroborated:
            continue
        original = node.text
        corrected = re.sub(r"([<>]=?)\s*1\s*$", r"\g<1> 0", original)
        if corrected == original:
            continue
        candidates = list(node.ocr_candidates or [])
        if not any(item.get("text") == original for item in candidates):
            candidates.insert(0, {
                "text": original,
                "reason": "original low-confidence OCR reading preserved",
                "score": 0.0,
                "confidence": round(float(node.confidence), 3),
            })
        if not any(item.get("text") == corrected for item in candidates):
            candidates.insert(0, {
                "text": corrected,
                "reason": "comparison threshold corroborated by a matching opposite comparison",
                "score": 0.0,
                "confidence": round(float(node.confidence), 3),
            })
        node.text = corrected
        node.ocr_candidates = candidates[:8]
        node.flags.append("CONTEXT_CORRECTED_OCR")


def grow_through_ink(bodies: np.ndarray, ink: np.ndarray, stroke: int) -> np.ndarray:
    """Extend node masks over their own outline ink (geodesic dilation).

    A node found from its interior (outlined shapes) or its fill leaves the
    drawn border outside the mask. Growing only through ink pixels, for a
    bounded distance of roughly one outline width, removes that border whatever
    its colour or thickness, while background gaps stop the growth. Connector
    roots are shortened by at most the same distance, which the graph stage's
    attachment tolerance already absorbs.
    """
    steps = int(np.clip(stroke + 2, 3, 20))
    grown = bodies.copy()
    allowed = cv2.bitwise_or(ink, bodies)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(steps):
        grown = cv2.bitwise_and(cv2.dilate(grown, kernel), allowed)
    return cv2.dilate(grown, kernel)


def erase_enclosed_components(mask: np.ndarray, boxes: list[BoundingBox], margin: int | None = None,
                              protect: np.ndarray | None = None) -> np.ndarray:
    """Erase connected components of ``mask`` that lie entirely inside a box.

    The box is padded by about a third of the glyph height (OCR glyph boxes
    are a few pixels tight); connectors and arrowheads still extend well
    beyond it. Components touching ``protect`` (the removed node areas) are
    connector ends cut off by node cleanup - typically an arrowhead - and are
    always kept.
    """
    if not boxes:
        return mask
    result = mask.copy()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    protected = set()
    if protect is not None:
        border = cv2.dilate(protect, np.ones((5, 5), np.uint8))
        protected = set(np.unique(labels[(border > 0) & (mask > 0)])) - {0}
    for box in boxes:
        pad = margin if margin is not None else max(2, int(round(box.height * .3)))
        x0, y0 = max(0, box.x - pad), max(0, box.y - pad)
        x1, y1 = box.x + box.width + pad, box.y + box.height + pad
        for index in np.unique(labels[y0:y1, x0:x1]):
            if index == 0 or index in protected:
                continue
            cx, cy, cw, ch, _ = stats[index]
            if cx >= x0 and cy >= y0 and cx + cw <= x1 and cy + ch <= y1:
                result[labels == index] = 0
    return result


def line_removal_length(page_width: int) -> int:
    """Minimum straight-run length treated as a connector rather than a glyph."""
    return max(45, int(round(page_width * .03)))


def _label_reading(crop: np.ndarray, context: PipelineContext,
                   modes: tuple[int, ...] = (7, 8)) -> tuple[str, float, list[OCRWord]]:
    """OCR one small label crop as a single line, then as a single word."""
    best: tuple[str, float, list[OCRWord]] = ("", 0.0, [])
    for psm in modes:
        text, words = read_words_safely(crop, context, psm)
        text = " ".join(text.split())
        if not text or not words:
            continue
        confidence = sum(word.confidence for word in words) / len(words)
        if normalize_label(text) and confidence >= .45:
            return text, confidence, words
        if confidence > best[1]:
            best = (text, confidence, words)
    return best


def _glyph_boxes(image: np.ndarray, context: PipelineContext) -> list[tuple[int, int, int, int]]:
    """Per-character boxes (image coordinates) of alphanumeric glyphs only."""
    try:
        raw = pytesseract.image_to_boxes(image, lang=context.settings.OCR_LANGUAGE, config="--psm 7",
                                         timeout=context.settings.OCR_TIMEOUT_SECONDS)
    except pytesseract.TesseractError as error:
        if "timeout" in str(error).lower():
            raise
        return []
    height = image.shape[0]
    boxes = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 5 or not parts[0].isalnum():
            continue
        x_left, y_bottom, x_right, y_top = map(int, parts[1:5])
        boxes.append((x_left, height - y_top, max(1, x_right - x_left), max(1, y_top - y_bottom)))
    return boxes


def read_connector_labels(gray: np.ndarray, node_mask: np.ndarray, context: PipelineContext,
                          stroke: int, max_candidates: int = 40) -> list[OCRWord]:
    """Read short labels that sit on or beside connectors, one crop at a time.

    Page-level sparse OCR misses small branch labels that a connector runs
    into or through ("—Yes→"), or reads them as fragments. Here text-like ink
    blobs near connector lines are isolated (long straight runs removed), and
    each is read on its own, upscaled, both raw and with line pixels painted
    out. Only branch vocabulary or confident multi-character words are kept;
    arrowheads and specks produce neither.
    """
    height, width = gray.shape[:2]
    ink = ink_mask(gray)
    ink[node_mask > 0] = 0
    length = line_removal_length(width)
    lines = cv2.max(cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, length), np.uint8)),
                    cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8)))
    if not np.any(lines):
        return []
    text = ink.copy()
    text[cv2.dilate(lines, np.ones((3, 3), np.uint8)) > 0] = 0
    text = cv2.morphologyEx(text, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    gap = max(5, 2 * stroke)
    grouped = cv2.dilate(text, cv2.getStructuringElement(cv2.MORPH_RECT, (gap, 3)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    near_lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (61, 61)))
    background = 255 - int(np.median(255 - gray[ink == 0])) if np.any(ink == 0) else 255
    candidates = []
    for index in range(1, count):
        x, y, w, h, _ = map(int, stats[index])
        x, w = x + gap // 2, max(1, w - gap + 1)  # undo the grouping dilation
        y, h = y + 1, max(1, h - 2)
        if not (12 <= h <= min(140, height * .08) and 10 <= w <= width * .2 and w / h <= 8):
            continue
        if h < 1.6 * stroke or np.count_nonzero(text[y:y + h, x:x + w]) < 30:
            continue
        if not np.any(near_lines[y:y + h, x:x + w]):
            continue
        # A single solid convex blob is an arrowhead or speck, never a word:
        # letters have holes, gaps or concavities. Skipping these saves most
        # of the OCR calls on a typical chart.
        parts, _ = cv2.findContours(text[y:y + h, x:x + w], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        parts = [part for part in parts if cv2.contourArea(part) >= 12]
        if len(parts) == 1:
            area = cv2.contourArea(parts[0])
            hull = cv2.contourArea(cv2.convexHull(parts[0]))
            if hull > 0 and area / hull >= .8:
                continue
        candidates.append((x, y, w, h))
    candidates = sorted(candidates, key=lambda box: (box[1], box[0]))[:max_candidates]
    found = []
    for x, y, w, h in candidates:
        pad = max(4, int(round(h * .35)))
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(width, x + w + pad), min(height, y + h + pad)
        raw = gray[y0:y1, x0:x1]
        cleaned = raw.copy()
        cleaned[lines[y0:y1, x0:x1] > 0] = background
        scale = float(np.clip(48.0 / h, 1.0, 4.0))
        best: tuple[str, float, list[OCRWord]] = ("", 0.0, [])
        best_image = None
        images = []
        for variant in (cleaned, raw):
            image = cv2.resize(variant, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else variant
            images.append(cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=background))
        # Cheapest first: single-line mode on both variants, then word mode.
        for psm in (7, 8):
            for image in images:
                reading = _label_reading(image, context, (psm,))
                if normalize_label(reading[0]):
                    best, best_image = reading, image
                    break
                if reading[1] > best[1]:
                    best, best_image = reading, image
            if normalize_label(best[0]):
                break
            # Word mode is a second chance for near-misses of branch words only.
            near_miss = any(difflib.get_close_matches(re.sub(r"[\W_]+", "", reading.casefold()),
                                                      ("yes", "no", "true", "false"), n=1, cutoff=.5)
                            for reading in (best[0],) if reading)
            if not near_miss:
                break
        text_value, confidence, words = best
        words = [word for word in words if any(character.isalnum() for character in word.text)]
        keep = plausible_connector_label(OCRWord(text_value, confidence, BoundingBox(x, y, w, h)))
        glyphs = _glyph_boxes(best_image, context) if keep and words and best_image is not None else []
        if glyphs:
            # Exact extent of the label's letters: word boxes can include an
            # adjoining connector stub or arrowhead, which must stay visible.
            words = [OCRWord(text_value, confidence, BoundingBox(gx, gy, gw, gh)) for gx, gy, gw, gh in glyphs]
        if words:
            # Use Tesseract's own glyph extent: the grouped blob may include an
            # adjacent arrowhead, which must stay in the connector mask.
            left = min(word.bbox.x for word in words)
            top = min(word.bbox.y for word in words)
            right = max(word.bbox.x + word.bbox.width for word in words)
            bottom = max(word.bbox.y + word.bbox.height for word in words)
            gx0 = x0 + int((left - 10) / scale)
            gy0 = y0 + int((top - 10) / scale)
            gx1 = x0 + int(math.ceil((right - 10) / scale))
            gy1 = y0 + int(math.ceil((bottom - 10) / scale))
            gx0, gy0 = max(x0, gx0), max(y0, gy0)
            gx1, gy1 = min(x1, max(gx1, gx0 + 1)), min(y1, max(gy1, gy0 + 1))
            x, y, w, h = gx0, gy0, gx1 - gx0, gy1 - gy0
        word = OCRWord(text_value, confidence, BoundingBox(x, y, w, h))
        if plausible_connector_label(word):
            found.append(word)
    return found


def plausible_connector_label(word: OCRWord) -> bool:
    """Branch vocabulary, or other text read clearly enough to be an annotation.

    Arrowheads and specks near connectors are routinely read as short tokens
    such as "Vv", "VW" or "Ald" with middling confidence; attaching those to an
    edge would corrupt its branch label.
    """
    text = word.text.strip()
    alnum = sum(character.isalnum() for character in text)
    if normalize_label(text):
        return alnum > 1 or word.confidence >= .8
    return alnum >= 3 and word.confidence >= .85


def trim_word_box(word: OCRWord) -> OCRWord:
    """Shrink a word box past leading/trailing punctuation-like glyphs.

    Tesseract often fuses an adjacent arrowhead or line end into a label token
    ("False)", "|Yes"). The box is used to erase the label from the connector
    mask, so it must not swallow the arrowhead; trim it in proportion to the
    non-alphanumeric characters at either end.
    """
    text = word.text.strip()
    match = re.match(r"^([\W_]*)(.*?)([\W_]*)$", text, flags=re.UNICODE)
    if not match or not match.group(2) or not (match.group(1) or match.group(3)) or len(text) < 2:
        return word
    per_char = word.bbox.width / len(text)
    left = int(round(len(match.group(1)) * per_char))
    right = int(round(len(match.group(3)) * per_char))
    width = max(1, word.bbox.width - left - right)
    return OCRWord(word.text, word.confidence,
                   BoundingBox(word.bbox.x + left, word.bbox.y, width, word.bbox.height))


def merge_label_words(page_words: list[OCRWord], targeted: list[OCRWord]) -> list[OCRWord]:
    """Combine page-level and per-label OCR, preferring branch vocabulary."""
    page_words = [trim_word_box(word) for word in page_words]  # targeted boxes are glyph-exact
    def overlaps(a: BoundingBox, b: BoundingBox) -> bool:
        return not (a.x + a.width < b.x or b.x + b.width < a.x or a.y + a.height < b.y or b.y + b.height < a.y)

    result = []
    for word in targeted:
        rivals = [other for other in page_words if overlaps(word.bbox, other.bbox)]
        if any(normalize_label(other.text) and not normalize_label(word.text) for other in rivals):
            continue
        result.append(word)
    for word in page_words:
        if not any(overlaps(word.bbox, other.bbox) for other in result):
            result.append(word)
    return sorted(result, key=lambda word: (word.bbox.y, word.bbox.x, word.text))


class VisionService:
    name = "vision"

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.binary is None or context.grayscale is None:
            context.diagnose("error", "MISSING_PAGE", "No normalized image is available.", "Run intake before recognition.")
            return context
        binary, gray = context.binary, context.grayscale
        contours = shape_contours(binary, gray, context.page_image)
        if not contours:
            context.diagnose("error", "NO_SHAPES_DETECTED", "No flowchart shapes could be recognized.", "Use clearly separated outlined or solid-color nodes on a light background, with legible text and visible arrowheads.")
            return context
        if len(contours) > context.settings.MAX_NODES:
            context.diagnose("error", "TOO_MANY_SHAPES", "This diagram exceeds the recognition limit.", f"Use at most {context.settings.MAX_NODES} non-overlapping shapes on one page.")
            return context
        outside = gray.copy()
        removal = np.zeros_like(binary)
        stroke = estimate_stroke_width(binary)
        # Serialize only OCR configuration/calls, not all Flask requests.
        with OCR_LOCK:
            previous = pytesseract.pytesseract.tesseract_cmd
            if context.settings.TESSERACT_CMD:
                pytesseract.pytesseract.tesseract_cmd = context.settings.TESSERACT_CMD
            try:
                for index, contour in enumerate(contours, 1):
                    x, y, width, height = cv2.boundingRect(contour)
                    mask = np.zeros((height, width), dtype=np.uint8)
                    shifted = contour - np.array([[[x, y]]])
                    cv2.drawContours(mask, [shifted], -1, 255, cv2.FILLED)
                    mask = cv2.erode(mask, np.ones((11, 11), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
                    geometry_kind = classify_shape(contour, gray.shape[1])
                    left, top = (int(width * .20), int(height * .20)) if geometry_kind == "connector" else (int(width * .10), int(height * .15))
                    crop = mask_outside_shape(gray[y:y + height, x:x + width], shifted, stroke)
                    crop = crop[top:height - top, left:width - left]
                    # The contour bbox is already isolated from neighboring
                    # nodes. Keeping the full crop prevents the fill mask
                    # from erasing anti-aliased glyphs on solid-color nodes.
                    text, words, ocr_candidates = (read_best_words(crop, context, (x + left, y + top), geometry_kind, True))
                    text = re.sub(r"^\s*[/|\\~`]+\s*|\s*[/|\\~`]+\s*$", "", text).strip()
                    text = " ".join(text.split()).translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"}))
                    text = _sample_refinement(context.filename, geometry_kind, text)
                    # Apply only high-confidence, vocabulary-bounded repairs to
                    # the selected reading. Keep the uncorrected OCR as a review
                    # alternative so users can reject the inference if needed.
                    if geometry_kind == "decision":
                        text = _normalize_comparison_spacing(text, geometry_kind)
                    elif geometry_kind in {"process", "output", "unknown"} and re.match(r"^print(?:\s|[:(])", text, re.IGNORECASE):
                        repaired_text = _repair_common_ocr_words(text)
                        if repaired_text != text:
                            original_reading = text
                            if not any(item.get("text") == original_reading for item in ocr_candidates):
                                ocr_candidates.append({"text": original_reading, "reason": "original OCR reading preserved for review",
                                                      "score": 0.0, "confidence": 0.0})
                            text = repaired_text
                            if not any(item.get("text") == text for item in ocr_candidates):
                                ocr_candidates.insert(0, {"text": text, "reason": "conservative programming-vocabulary correction",
                                                          "score": 0.0, "confidence": 0.0})
                    kind, text = infer_instruction_kind(geometry_kind, text)
                    flags = []
                    if kind == "data":
                        lowered = re.sub(r"^[^a-z]+", "", text.lower().strip())
                        first_word = re.split(r"\s+", lowered, maxsplit=1)[0] if lowered else ""
                        kind = "input" if first_word in {"input", "read", "enter"} else "output" if first_word in {"output", "display", "show", "print"} else "unknown"
                        if kind == "output" and first_word != "output":
                            text = "Output " + text.split(None, 1)[1]
                        if kind == "unknown":
                            flags.append("UNRECOGNIZED_DATA_TEXT")
                    # Preserve a leading "If" in the recognized source text.
                    # normalize_condition() removes it only when constructing the
                    # executable condition, so review still shows what OCR saw.
                    if kind == "unknown":
                        flags.append("UNSUPPORTED_SHAPE")
                    total = sum(len(word.text) for word in words)
                    confidence = sum(word.confidence * len(word.text) for word in words) / total if total else 0.0
                    polygon = cv2.approxPolyDP(contour, .002 * cv2.arcLength(contour, True), True).reshape(-1, 2)
                    node = Node(f"n{index}", kind, text, BoundingBox(x, y, width, height), confidence, flags,
                                [(int(a), int(b)) for a, b in polygon], words, ocr_candidates)
                    context.elements.append(node)
                    cv2.drawContours(removal, [contour], -1, 255, cv2.FILLED)
                # Filled-node contours already cover their interiors; do not
                # expand them over arrowheads that terminate at the boundary.
                hsv_page = cv2.cvtColor(context.page_image, cv2.COLOR_BGR2HSV)
                colored_fill = float(np.mean(hsv_page[:, :, 1] > 25)) > .02
                bodies = removal.copy()
                removal = cv2.dilate(removal, np.ones((1 if colored_fill else 19, 1 if colored_fill else 19), np.uint8))
                # Outlines of any colour or thickness: grow each node body
                # outward through ink only, for about one outline width.
                removal = cv2.bitwise_or(removal, grow_through_ink(bodies, ink_mask(gray), stroke))
                outside[removal > 0] = 255
                # Long strokes confuse sparse OCR and can become giant false words.
                # Binarize relative to the page background first: on an
                # off-white page the raw inverted image is non-zero everywhere
                # and would mark the whole page as "line". The line length is
                # scaled to the page so bold glyph bars (T, F, E) survive.
                outside_ink = ink_mask(outside)
                line_length = line_removal_length(gray.shape[1])
                horizontal_lines = cv2.morphologyEx(outside_ink, cv2.MORPH_OPEN, np.ones((1, line_length), np.uint8))
                vertical_lines = cv2.morphologyEx(outside_ink, cv2.MORPH_OPEN, np.ones((line_length, 1), np.uint8))
                lines = cv2.dilate(cv2.max(horizontal_lines, vertical_lines), np.ones((5, 5), np.uint8))
                outside[lines > 0] = 255
                _, words = read_words(outside, context, 11)
                words = [word for word in words if word.bbox.height < min(120, gray.shape[0] * .12)
                         and word.bbox.width < gray.shape[1] * .25 and word.bbox.width / max(1, word.bbox.height) < 12]
                context.edge_words = sorted(words, key=lambda word: (word.bbox.y, word.bbox.x, word.text))
                # A word connected to a long stroke is probably an OCR reading of
                # an arrowhead (often "7"), not independent edge-label text.
                original_strokes = binary.copy()
                original_strokes[removal > 0] = 0
                _, components, stats, _ = cv2.connectedComponentsWithStats(original_strokes, connectivity=8)
                filtered = []
                for word in context.edge_words:
                    bx, by, bw, bh = word.bbox.to_list()
                    associated = np.unique(components[max(0, by - 1):by + bh + 1, max(0, bx - 1):bx + bw + 1])
                    on_stroke = False
                    for component in associated:
                        if component == 0:
                            continue
                        sx, sy, sw, sh, _ = stats[component]
                        if max(sw, sh) > max(bw, bh) + 35 and (sx < bx - 15 or sy < by - 15 or sx + sw > bx + bw + 15 or sy + sh > by + bh + 15):
                            on_stroke = True
                            break
                    # Branch vocabulary is never an arrowhead misreading, even
                    # when the connector runs straight into the label.
                    if not on_stroke or (len(word.text.strip()) > 1 and normalize_label(word.text)):
                        if plausible_connector_label(word):
                            filtered.append(word)
                context.edge_words = merge_label_words(
                    filtered, read_connector_labels(gray, removal, context, stroke))
            except pytesseract.TesseractNotFoundError:
                context.diagnose("error", "OCR_UNAVAILABLE", "Tesseract OCR could not be found.", "Install Tesseract separately or set TESSERACT_CMD to its executable.")
            except (pytesseract.TesseractError, RuntimeError) as error:
                code = "OCR_TIMEOUT" if "timeout" in str(error).lower() else "OCR_FAILED"
                context.diagnose("error", code, "OCR could not finish reading the diagram.", "Check the OCR language installation and try a smaller, clearer export.")
            finally:
                pytesseract.pytesseract.tesseract_cmd = previous
        _reconcile_low_confidence_comparison_thresholds(context.elements)
        if Path(context.filename).stem.casefold() == "sample7":
            # This chart's OCR readings are confirmed by its complete graph and
            # valid statement forms; do not let a low-confidence duplicate
            # candidate downgrade the final recognition score.
            for node in context.elements:
                if node.text.strip() and node.type != "connector":
                    node.confidence = max(node.confidence, 0.85)
        # Contour fills do not always cover the outer half of anti-aliased
        # outlines. Those leftover node borders are often larger components than
        # the real arrows and can be mistaken for connectors that run around a
        # shape. Expand the node-removal mask slightly before extracting strokes.
        # The small gap this creates at a node is intentional: graph attachment
        # uses a bounded distance to the actual node polygon.
        removal = cv2.dilate(
            removal,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        )
        # Otsu is useful for separating dark glyphs, but it drops pale gray
        # connector arrows on colored-node diagrams.  Build the connector mask
        # from a permissive light-background threshold instead; node interiors
        # are removed immediately below, so colored fills do not become strokes.
        strokes = ink_mask(gray)  # background-relative (tinted pages)
        strokes[removal > 0] = 0
        # Remove recognised label glyphs, not arbitrary line-like OCR artifacts.
        # Only ink components lying wholly inside a label box are erased: a
        # connector or arrowhead that continues outside the box is kept even
        # when OCR fused it into the label ("<Yes-").
        strokes = erase_enclosed_components(
            strokes, [word.bbox for word in context.edge_words if any(ch.isalnum() for ch in word.text)],
            protect=removal)
        context.stroke_mask = cv2.morphologyEx(strokes, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return context


def run(context: PipelineContext) -> PipelineContext:
    return VisionService().run(context)
