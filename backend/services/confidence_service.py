"""Deterministic, component-level confidence scoring for flowchart analysis.

The scale is element-normalized: each diagnostic lowers the score of the
element it describes (a node's text, a connector) rather than subtracting a
fixed number of points from the whole chart, so a large, correct diagram is not
graded lower than a small one merely because it raises more routine warnings.
See README Section 12.1 for the rationale and the measured calibration.
"""
from __future__ import annotations

from typing import Iterable

from backend.models import Diagnostic, FlowGraph, Node

# Connector/graph reconstruction carries the most weight because it caused 24 of
# the 28 blocking failures in the recorded test-chart runs; shape classification
# is near-perfect whenever analysis gets that far, so it is weighted lightly.
CONFIDENCE_WEIGHTS = {"ocr": 0.40, "shape": 0.10, "graph": 0.50}
VALID_SHAPES = {"terminal", "process", "decision", "input", "output", "iteration", "connector"}

# Highest overall score an analysis with a blocking (error-level) diagnostic can
# receive: it cannot proceed to code generation, so it is never above "Low".
BLOCKED_CAP = 69
# Highest overall score when a statement reads a name nothing in the chart
# defines (UNDEFINED_VARIABLE): usually a misread that would still produce wrong
# or failing code, so the grade is at most "Moderate" (check flagged elements).
REVIEW_CAP = 79
# Tesseract confidence at or above RELIABLE counts as fully reliable; at or
# below UNRELIABLE it counts as unusable; linear in between. Tesseract often
# reports 60-75% for correctly read short arithmetic, so the floor is kept low.
OCR_RELIABLE = 0.90
OCR_UNRELIABLE = 0.40
# Share of a component score taken from its weakest element (rest: average).
WEAKEST_SHARE = 0.35

# Diagnostics that specifically describe a component's reliability. Validation and
# logic errors remain visible, but are not allowed to masquerade as OCR quality.
OCR_CODES = {
    "LOW_OCR_CONFIDENCE", "PROGRAMMING_OCR_AMBIGUITY", "UNREADABLE_TEXT",
    "OCR_FAILED", "OCR_TIMEOUT", "OCR_UNAVAILABLE", "UNSUPPORTED_STATEMENT",
    "UNDEFINED_VARIABLE",
}
SHAPE_CODES = {"UNSUPPORTED_SHAPE", "NO_SHAPES_DETECTED", "TOO_MANY_SHAPES"}
GRAPH_CODES = {
    "AMBIGUOUS_CONNECTOR", "AMBIGUOUS_ARROW_DIRECTION", "AMBIGUOUS_EDGE_LABEL",
    "AMBIGUOUS_CONNECTOR_PAIR", "DANGLING_ARROW", "INFERRED_ARROW_DIRECTION",
    "INFERRED_TERMINAL_DIRECTION", "INFERRED_VISIBLE_GAP_CONNECTION",
    "INFERRED_ARROW_TO_CONNECTOR", "INFERRED_BRANCH_LABELS", "INFERRED_DECISION_BRANCH",
    "INFERRED_BRANCH_TO_END", "REDUNDANT_CONNECTOR_FRAGMENT", "RESOLVED_CONNECTOR_AMBIGUITY",
    "MISSING_CONNECTORS", "INVALID_OUT_DEGREE", "END_HAS_OUTGOING_ARROW",
    "START_HAS_INCOMING_ARROW", "INVALID_BRANCH_LABELS", "DECISION_MISSING_BRANCH",
    "UNPAIRED_CONNECTOR", "NO_START_NODE", "MULTIPLE_START_NODES", "NO_END_NODE",
    "UNREACHABLE_NODE", "NO_PATH_TO_END",
}

# Fraction of one element's reliability removed by a warning. Errors always
# remove all of it. Unlisted warnings use DEFAULT_WARNING_WEIGHT.
DEFAULT_WARNING_WEIGHT = 0.30
WARNING_WEIGHTS = {
    # Raised on nearly every chart, including every end-to-end correct one;
    # it is a review prompt, not evidence of a misread.
    "PROGRAMMING_OCR_AMBIGUITY": 0.02,
    "LOW_OCR_CONFIDENCE": 0.20,
    # A name nothing in the chart defines: the strongest sign of a misread
    # statement among correctly structured charts.
    "UNDEFINED_VARIABLE": 0.60,
    # Start/End roles fix the direction, so these inferences are low risk.
    "INFERRED_TERMINAL_DIRECTION": 0.05,
    "RESOLVED_CONNECTOR_AMBIGUITY": 0.05,
    "REDUNDANT_CONNECTOR_FRAGMENT": 0.10,
    "DANGLING_ARROW": 0.50,
}


def _clamp(value: float) -> int:
    return max(0, min(100, round(float(value))))


def _unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def confidence_grade(score: float) -> tuple[str, str]:
    """Return the user-facing grade required by the product scale."""
    score = float(score)
    if score >= 90:
        return "Excellent", "Excellent"
    if score >= 80:
        return "Good", "Good"
    if score >= 70:
        return "Moderate", "Moderate"
    if score >= 50:
        return "Low", "Low"
    return "Very Low", "Very Low"


def _weight(item: Diagnostic) -> float:
    if item.severity == "error":
        return 1.0
    return WARNING_WEIGHTS.get(item.code, DEFAULT_WARNING_WEIGHT)


def _has_error(diagnostics: Iterable[Diagnostic], codes: set[str]) -> bool:
    return any(item.severity == "error" and item.code in codes for item in diagnostics)


def _blend(values: list[float]) -> float:
    """Mostly the average, with a smaller share from the weakest element."""
    average = sum(values) / len(values)
    return 100.0 * ((1.0 - WEAKEST_SHARE) * average + WEAKEST_SHARE * min(values))


def _ocr_reliability(confidence: float) -> float:
    return _unit((float(confidence) - OCR_UNRELIABLE) / (OCR_RELIABLE - OCR_UNRELIABLE))


def score_ocr(nodes: list[Node], diagnostics: list[Diagnostic]) -> int:
    if not nodes:
        return 0
    node_ids = {node.id for node in nodes}
    penalty = {node.id: 0.0 for node in nodes}
    label_penalty: dict[str, float] = {}
    page_penalty = 0.0
    for item in diagnostics:
        if item.code not in OCR_CODES:
            continue
        hits = [element for element in item.element_ids if element in node_ids]
        for element in hits:
            penalty[element] += _weight(item)
        others = [element for element in item.element_ids if element not in node_ids]
        for element in others:  # branch labels are OCR'd text on a connector
            label_penalty[element] = label_penalty.get(element, 0.0) + _weight(item)
        if not item.element_ids:
            page_penalty += _weight(item)
    # Node confidence is the existing OCR confidence, derived from the selected
    # candidate's Tesseract word confidence and preserved OCR provenance.
    values = [_ocr_reliability(node.confidence) * _unit(1.0 - penalty[node.id] - page_penalty)
              for node in nodes]
    values += [_unit(1.0 - value - page_penalty) for value in label_penalty.values()]
    score = _blend(values)
    if _has_error(diagnostics, OCR_CODES):
        score = min(score, BLOCKED_CAP)
    return _clamp(score)


def score_shape(nodes: list[Node], diagnostics: list[Diagnostic]) -> int:
    if not nodes:
        return 0
    flagged = {element for item in diagnostics if item.code in SHAPE_CODES for element in item.element_ids}
    # A missing polygon is not itself a failure: some supported contours are
    # represented by bounding boxes. Unknown/unsupported classifications are.
    values = [float(node.type in VALID_SHAPES and node.id not in flagged) for node in nodes]
    score = _blend(values)
    if _has_error(diagnostics, SHAPE_CODES):
        score = min(score, BLOCKED_CAP)
    return _clamp(score)


def score_graph(graph: FlowGraph | None, diagnostics: list[Diagnostic]) -> int:
    if graph is None:
        return 0
    if not graph.nodes:
        return 0
    if not graph.edges:
        base = 100.0 if len(graph.nodes) <= 1 else 35.0
        reliability = 1.0
    else:
        node_ids = {node.id for node in graph.nodes}
        valid_attachments = sum(
            edge.source in node_ids and edge.target in node_ids and bool(edge.points)
            for edge in graph.edges
        )
        base = valid_attachments / len(graph.edges) * 100.0
        # Each connector diagnostic describes (about) one connector, so the
        # penalty is the share of connectors affected, weighted by risk.
        flagged = sum(_weight(item) for item in diagnostics if item.code in GRAPH_CODES)
        reliability = _unit(1.0 - flagged / len(graph.edges))
    score = base * reliability
    if _has_error(diagnostics, GRAPH_CODES):
        score = min(score, BLOCKED_CAP)
    return _clamp(score)


def calculate_confidence(nodes: list[Node], graph: FlowGraph | None,
                         diagnostics: list[Diagnostic]) -> dict:
    """Calculate all confidence values from the current interpreted graph."""
    ocr = score_ocr(nodes, diagnostics)
    shape = score_shape(nodes, diagnostics)
    graph_score = score_graph(graph, diagnostics)
    weighted = _clamp(ocr * CONFIDENCE_WEIGHTS["ocr"] +
                      shape * CONFIDENCE_WEIGHTS["shape"] +
                      graph_score * CONFIDENCE_WEIGHTS["graph"])
    critical = min(ocr, shape, graph_score)
    blocking = sum(1 for item in diagnostics if item.severity == "error")
    overall, cap = weighted, None
    if overall > REVIEW_CAP and any(item.code == "UNDEFINED_VARIABLE" for item in diagnostics):
        overall, cap = REVIEW_CAP, "undefined_variable"
    if blocking and overall > BLOCKED_CAP:
        overall, cap = BLOCKED_CAP, "blocking_error"
    grade, label = confidence_grade(overall)
    return {
        "overall": overall,
        "grade": grade,
        "label": label,
        "ocr": ocr,
        "shape": shape,
        "graph": graph_score,
        "critical": critical,
        "weighted": weighted,
        "cap": cap,
        "blocking_errors": blocking,
        "weights": {key: int(round(value * 100)) for key, value in CONFIDENCE_WEIGHTS.items()},
    }
