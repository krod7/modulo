"""Unit tests for scale-aware shape, connector, arrowhead and label handling.

These are OCR-free: they draw synthetic geometry and exercise each new stage
directly, so they run in milliseconds and pin down the behaviour that fixed
samples 1, 7, 8 and 10 without depending on those images.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.config import Settings
from backend.models import BoundingBox, Node, OCRWord, PipelineContext
from backend.notation import normalize_condition, terminal_role
from backend.services import graph_service as gs
from backend.services.connector_geometry import (connector_stroke_width, endpoint_has_arrowhead, ink_mask,
                                                 stroke_exits)
from backend.services.vision_service import (classify_shape, erase_enclosed_components, estimate_stroke_width,
                                             filled_region_contours, looks_like_parallelogram,
                                             plausible_connector_label, read_words_safely, trim_word_box)


def _points(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    return np.column_stack((xs, ys))


def _arrow(mask, start, end, width=4, head=24, style="filled"):
    cv2.line(mask, start, end, 255, width)
    angle = np.arctan2(end[1] - start[1], end[0] - start[0])
    left = (int(end[0] - head * np.cos(angle - .45)), int(end[1] - head * np.sin(angle - .45)))
    right = (int(end[0] - head * np.cos(angle + .45)), int(end[1] - head * np.sin(angle + .45)))
    if style == "filled":
        cv2.fillPoly(mask, [np.array([end, left, right], np.int32)], 255)
    else:
        cv2.polylines(mask, [np.array([left, end, right], np.int32)], False, 255, width)


def _node(node_id, kind, x, y, w, h, text="x = 1"):
    polygon = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return Node(node_id, kind, text, BoundingBox(x, y, w, h), polygon=polygon)


# ---------------------------------------------------------------- stroke width
def test_stroke_width_estimates_follow_the_drawing():
    mask = np.zeros((400, 400), np.uint8)
    for x in (50, 150, 250):
        cv2.line(mask, (x, 20), (x, 380), 255, 6)
    assert abs(estimate_stroke_width(mask) - 6) <= 1


def test_connector_stroke_width_ignores_page_frame():
    mask = np.zeros((600, 600), np.uint8)
    mask[:, 585:600] = 255  # a thick screenshot edge touching the border
    for y in (100, 250, 400):
        cv2.line(mask, (100, y), (400, y), 255, 5)
    assert connector_stroke_width(mask) <= 8  # the lines (~5-7 px), not the 15 px frame


# ---------------------------------------------------------------- filled shapes
def test_thick_same_colour_connectors_do_not_weld_filled_nodes():
    page = np.full((700, 900, 3), 250, np.uint8)
    purple, cyan = (230, 40, 140), (210, 200, 0)
    cv2.rectangle(page, (300, 60), (600, 160), purple, -1)
    cv2.rectangle(page, (300, 300), (600, 400), purple, -1)
    cv2.line(page, (450, 160), (450, 300), cyan, 9)
    cv2.line(page, (100, 230), (800, 230), cyan, 9)  # long thick crossing line
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours = filled_region_contours(binary, gray, page)
    boxes = sorted(cv2.boundingRect(c) for c in contours)
    assert len(boxes) == 2, boxes
    assert all(w >= 280 and h >= 90 for _, _, w, h in boxes)


def test_filled_rectangles_keep_sharp_corners_for_classification():
    page = np.full((500, 700, 3), 255, np.uint8)
    cv2.rectangle(page, (200, 150), (500, 300), (240, 220, 170), -1)
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours = filled_region_contours(binary, gray, page)
    assert len(contours) == 1
    assert classify_shape(contours[0], page.shape[1]) == "process"


def test_bold_coloured_text_is_not_a_filled_node():
    page = np.full((400, 600, 3), 255, np.uint8)
    cv2.putText(page, "True", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 4, (30, 160, 30), 14)
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    assert filled_region_contours(binary, gray, page) == []


def test_slightly_slanted_parallelogram_is_data_and_rectangle_is_not():
    image = np.zeros((200, 700), np.uint8)
    cv2.fillPoly(image, [np.array([[60, 40], [620, 40], [600, 140], [40, 140]], np.int32)], 255)
    contour = max(cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    assert looks_like_parallelogram(contour)
    assert classify_shape(contour, 2000) == "data"
    rect = np.zeros((200, 700), np.uint8)
    cv2.rectangle(rect, (40, 40), (620, 140), 255, -1)
    contour = max(cv2.findContours(rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    assert not looks_like_parallelogram(contour)


# ---------------------------------------------------------------- arrowheads
@pytest.mark.parametrize("style", ["filled", "open"])
def test_arrowhead_found_at_tip_but_not_at_tail_of_short_connector(style):
    mask = np.zeros((200, 200), np.uint8)
    _arrow(mask, (100, 40), (100, 110), width=4, head=26, style=style)
    points = _points(mask)
    assert endpoint_has_arrowhead(points, (100, 110), 4)
    assert not endpoint_has_arrowhead(points, (100, 40), 4)


def test_t_junction_and_corner_are_not_arrowheads():
    junction = np.zeros((300, 400), np.uint8)
    cv2.line(junction, (200, 20), (200, 250), 255, 4)
    cv2.line(junction, (40, 70), (360, 70), 255, 4)  # crossing bar near the end
    assert not endpoint_has_arrowhead(_points(junction), (200, 20), 4)
    corner = np.zeros((300, 400), np.uint8)
    cv2.line(corner, (200, 20), (200, 60), 255, 4)
    cv2.line(corner, (200, 60), (380, 60), 255, 4)  # elbow turning sideways
    assert not endpoint_has_arrowhead(_points(corner), (200, 20), 4)


def test_arrowhead_recovered_from_raw_ink_when_cleanup_cut_it():
    raw = np.zeros((300, 200), np.uint8)
    _arrow(raw, (100, 30), (100, 200), width=4, head=30)
    cleaned = raw.copy()
    cleaned[175:, :] = 0  # node cleanup erased most of the arrowhead
    assert endpoint_has_arrowhead(_points(cleaned), (100, 174), 4, head_mask=raw)


# ---------------------------------------------------------------- contacts
def test_line_running_beside_a_node_is_not_attached():
    nodes = [_node("a", "process", 300, 300, 200, 100)]
    mask = np.zeros((800, 800), np.uint8)
    cv2.line(mask, (240, 100), (240, 700), 255, 4)  # 60px beside, never ending there
    ports = gs.contacts(_points(mask), nodes, mask.shape, 100, stroke_width=4, touch_distance=30)
    assert ports == []
    legacy = gs.contacts(_points(mask), nodes, mask.shape, 100)
    assert legacy, "the legacy proximity test attaches it, which is the bug being fixed"


def test_stroke_exits_counts_only_continuing_pieces():
    band = np.zeros((50, 50), np.uint8)
    band[20:30, 10:40] = 255
    ring = np.zeros_like(band)
    ring[10:20, 22:28] = 255  # continuing shaft
    ring[18:20, 10:13] = 255  # barb poking in
    outer = np.zeros_like(band)
    outer[10:14, 22:28] = 255
    assert stroke_exits(band, ring) == 2
    assert stroke_exits(band, ring, outer) == 1


def _graph_context(nodes, mask, gray=None, words=()):
    context = PipelineContext(settings=Settings())
    context.elements = nodes
    context.stroke_mask = mask
    context.grayscale = gray if gray is not None else 255 - mask
    context.edge_words = list(words)
    return context


def test_merging_connectors_become_one_edge_per_source():
    nodes = [_node("n1", "output", 100, 100, 200, 80, "Output 1"),
             _node("n2", "output", 500, 100, 200, 80, "Output 2"),
             _node("n3", "terminal", 300, 450, 200, 80, "End")]
    mask = np.zeros((700, 800), np.uint8)
    cv2.line(mask, (200, 190), (200, 330), 255, 4)
    cv2.line(mask, (600, 190), (600, 330), 255, 4)
    cv2.line(mask, (200, 330), (600, 330), 255, 4)
    _arrow(mask, (400, 330), (400, 440), width=4, head=26)
    context = gs.GraphService().run(_graph_context(nodes, mask))
    edges = {(edge.source, edge.target) for edge in context.graph.edges}
    assert edges == {("n1", "n3"), ("n2", "n3")}
    assert not any(d.severity == "error" for d in context.diagnostics)


def test_redundant_label_fragment_does_not_raise_ambiguity_error():
    nodes = [_node("n1", "process", 300, 100, 200, 80), _node("n2", "process", 300, 400, 200, 80)]
    mask = np.zeros((600, 800), np.uint8)
    _arrow(mask, (400, 190), (400, 390), width=4, head=26)
    cv2.line(mask, (470, 200), (470, 380), 255, 4)  # stray glyph stem, no arrowhead
    context = gs.GraphService().run(_graph_context(nodes, mask))
    assert {(e.source, e.target) for e in context.graph.edges} == {("n1", "n2")}
    assert not any(d.severity == "error" for d in context.diagnostics)


# ---------------------------------------------------------------- labels
def test_label_on_line_and_label_at_corner_are_bridged():
    nodes = []
    mask = np.zeros((300, 400), np.uint8)
    cv2.line(mask, (20, 100), (150, 100), 255, 4)
    cv2.line(mask, (230, 100), (380, 100), 255, 4)
    word = OCRWord("Yes", .95, BoundingBox(160, 88, 60, 24))
    bridged = gs.bridge_through_labels(mask, [word], nodes, 4)
    assert cv2.connectedComponents(bridged)[0] - 1 == 1
    corner = np.zeros((300, 400), np.uint8)
    cv2.line(corner, (20, 200), (150, 200), 255, 4)   # enters from the left
    cv2.line(corner, (190, 180), (190, 20), 255, 4)   # leaves through the top
    word = OCRWord("No", .95, BoundingBox(162, 186, 40, 26))
    bridged = gs.bridge_through_labels(corner, [word], nodes, 4)
    assert cv2.connectedComponents(bridged)[0] - 1 == 1


def test_label_erasure_keeps_lines_and_node_touching_arrowheads():
    mask = np.zeros((200, 400), np.uint8)
    cv2.putText(mask, "Yes", (150, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.line(mask, (20, 150), (380, 150), 255, 3)  # line passing through the box
    cv2.fillPoly(mask, [np.array([[60, 85], [90, 75], [90, 95]], np.int32)], 255)  # arrowhead cut off at a node
    node_area = np.zeros_like(mask)
    node_area[:, :60] = 255
    boxes = [BoundingBox(55, 70, 160, 100)]
    cleaned = erase_enclosed_components(mask, boxes, protect=node_area)
    assert cleaned[150, 200] == 255           # crossing line kept
    assert cleaned[85, 80] == 255             # node-touching arrowhead kept
    assert not np.any(cleaned[85:112, 150:215])  # glyphs erased


def test_label_word_filters_and_trimming():
    assert plausible_connector_label(OCRWord("False", .7, BoundingBox(0, 0, 50, 20)))
    assert not plausible_connector_label(OCRWord("Vv", .76, BoundingBox(0, 0, 20, 20)))
    assert not plausible_connector_label(OCRWord("Ald", .77, BoundingBox(0, 0, 20, 20)))
    trimmed = trim_word_box(OCRWord("False)", .9, BoundingBox(100, 0, 120, 20)))
    assert trimmed.bbox.x == 100 and trimmed.bbox.width == 100


def test_ink_mask_ignores_tinted_background():
    gray = np.full((100, 100), 249, np.uint8)
    gray[40:60, 40:60] = 120
    mask = ink_mask(gray)
    assert mask[0, 0] == 0 and mask[50, 50] == 255


def test_single_ocr_crash_is_contained(monkeypatch):
    import pytesseract
    from backend.services import vision_service

    def crash(*args, **kwargs):
        raise pytesseract.TesseractError(-8, "")
    monkeypatch.setattr(vision_service, "read_words", crash)
    assert read_words_safely(np.zeros((10, 10), np.uint8), PipelineContext(settings=Settings()), 7) == ("", [])


# ---------------------------------------------------------------- notation
def test_terminal_synonyms_and_question_form_conditions():
    assert terminal_role("Stop") == "end" and terminal_role("END.") == "end"
    assert terminal_role("Begin") == "start" and terminal_role("Return x") is None
    assert normalize_condition("is a > b?") == "a > b"
    assert normalize_condition("Does S = 0?") == "S == 0"
    assert normalize_condition("if Number%2=0") == "Number%2 == 0"
    assert normalize_condition("x <= 3") == "x <= 3"
    # Natural-language conditions are left untouched for the validator to flag.
    assert normalize_condition("Is success less than 75?") == "Is success less than 75?"
