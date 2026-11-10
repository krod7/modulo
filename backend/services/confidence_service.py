"""Deterministic, component-level confidence scoring for flowchart analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from backend.models import Diagnostic, FlowGraph, Node

CONFIDENCE_WEIGHTS = {"ocr": 0.40, "shape": 0.20, "graph": 0.40}
VALID_SHAPES = {"terminal", "process", "decision", "input", "output", "iteration", "connector"}

# Diagnostics that specifically describe a component's reliability. Validation and
# logic errors remain visible, but are not allowed to masquerade as OCR quality.
OCR_CODES = {
    "LOW_OCR_CONFIDENCE", "PROGRAMMING_OCR_AMBIGUITY", "UNREADABLE_TEXT",
    "OCR_FAILED", "OCR_TIMEOUT", "OCR_UNAVAILABLE", "UNSUPPORTED_STATEMENT",
}
SHAPE_CODES = {"UNSUPPORTED_SHAPE", "NO_SHAPES_DETECTED", "TOO_MANY_SHAPES"}
GRAPH_CODES = {
    "AMBIGUOUS_CONNECTOR", "AMBIGUOUS_ARROW_DIRECTION", "AMBIGUOUS_EDGE_LABEL",
    "AMBIGUOUS_CONNECTOR_PAIR", "DANGLING_ARROW", "INFERRED_ARROW_DIRECTION",
    "INFERRED_TERMINAL_DIRECTION", "INFERRED_VISIBLE_GAP_CONNECTION",
    "INFERRED_ARROW_TO_CONNECTOR", "REDUNDANT_CONNECTOR_FRAGMENT",
    "MISSING_CONNECTORS", "INVALID_OUT_DEGREE", "END_HAS_OUTGOING_ARROW",
    "START_HAS_INCOMING_ARROW", "INVALID_BRANCH_LABELS", "DECISION_MISSING_BRANCH",
    "UNPAIRED_CONNECTOR", "NO_START_NODE", "MULTIPLE_START_NODES", "NO_END_NODE",
    "UNREACHABLE_NODE", "NO_PATH_TO_END",
}


def _clamp(value: float) -> int:
    return max(0, min(100, round(float(value))))


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


def _diagnostic_penalty(diagnostics: Iterable[Diagnostic], codes: set[str]) -> float:
    penalty = 0.0
    for item in diagnostics:
        if item.code not in codes:
            continue
        penalty += 16.0 if item.severity == "error" else 7.0
    return penalty


def score_ocr(nodes: list[Node], diagnostics: list[Diagnostic]) -> int:
    if not nodes:
        return 0
    # Node confidence is the existing OCR confidence, derived from the selected
    # candidate's Tesseract word confidence and preserved OCR provenance.
    values = [max(0.0, min(1.0, float(node.confidence))) for node in nodes]
    score = sum(values) / len(values) * 100.0
    return _clamp(score - _diagnostic_penalty(diagnostics, OCR_CODES))


def score_shape(nodes: list[Node], diagnostics: list[Diagnostic]) -> int:
    if not nodes:
        return 0
    classified = sum(node.type in VALID_SHAPES for node in nodes) / len(nodes) * 100.0
    # A missing polygon is not itself a failure: some supported contours are
    # represented by bounding boxes. Unknown/unsupported classifications are.
    return _clamp(classified - _diagnostic_penalty(diagnostics, SHAPE_CODES))


def score_graph(graph: FlowGraph | None, diagnostics: list[Diagnostic]) -> int:
    if graph is None:
        return 0
    if not graph.nodes:
        return 0
    if not graph.edges:
        base = 100.0 if len(graph.nodes) <= 1 else 35.0
    else:
        valid_attachments = sum(
            edge.source in {node.id for node in graph.nodes}
            and edge.target in {node.id for node in graph.nodes}
            and bool(edge.points)
            for edge in graph.edges
        )
        base = valid_attachments / len(graph.edges) * 100.0
    return _clamp(base - _diagnostic_penalty(diagnostics, GRAPH_CODES))


def calculate_confidence(nodes: list[Node], graph: FlowGraph | None,
                         diagnostics: list[Diagnostic]) -> dict:
    """Calculate all confidence values from the current interpreted graph."""
    ocr = score_ocr(nodes, diagnostics)
    shape = score_shape(nodes, diagnostics)
    graph_score = score_graph(graph, diagnostics)
    overall = _clamp(ocr * CONFIDENCE_WEIGHTS["ocr"] +
                     shape * CONFIDENCE_WEIGHTS["shape"] +
                     graph_score * CONFIDENCE_WEIGHTS["graph"])
    critical = min(ocr, shape, graph_score)
    grade, label = confidence_grade(overall)
    return {
        "overall": overall,
        "grade": grade,
        "label": label,
        "ocr": ocr,
        "shape": shape,
        "graph": graph_score,
        "critical": critical,
        "weights": {key: int(value * 100) for key, value in CONFIDENCE_WEIGHTS.items()},
    }
