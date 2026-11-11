"""Fast, OCR-free tests for the recognition confidence scale (README Section 12.1)."""
from __future__ import annotations

from backend.models import BoundingBox, Diagnostic, Edge, FlowGraph, Node
from backend.services.confidence_service import (
    BLOCKED_CAP, CONFIDENCE_WEIGHTS, CRITICAL_MARGIN, calculate_confidence, confidence_grade,
)


def chain(length: int, confidence: float = 0.96, kinds: list[str] | None = None) -> FlowGraph:
    kinds = kinds or (["terminal"] + ["process"] * (length - 2) + ["terminal"])
    nodes = [Node(f"n{i}", kinds[i - 1], "x = 1", BoundingBox(0, i * 100, 80, 40), confidence)
             for i in range(1, length + 1)]
    edges = [Edge(f"e{i}", f"n{i}", f"n{i + 1}", points=[(0, i), (0, i + 1)]) for i in range(1, length)]
    return FlowGraph(nodes=nodes, edges=edges)


def warn(code: str, *ids: str) -> Diagnostic:
    return Diagnostic("warning", code, code, "", list(ids))


def score(graph: FlowGraph, diagnostics: list[Diagnostic]) -> dict:
    return calculate_confidence(graph.nodes, graph, diagnostics)


def test_weights_sum_to_one_and_are_reported():
    assert abs(sum(CONFIDENCE_WEIGHTS.values()) - 1.0) < 1e-9
    assert score(chain(4), [])["weights"] == {"ocr": 40, "shape": 10, "graph": 50}


def test_clean_chart_is_excellent():
    result = score(chain(6), [])
    assert result["overall"] >= 95 and result["grade"] == "Excellent" and result["cap"] is None


def test_grade_bands():
    assert [confidence_grade(value)[0] for value in (90, 89, 80, 79, 70, 69, 50, 49)] == [
        "Excellent", "Good", "Good", "Moderate", "Moderate", "Low", "Low", "Very Low"]


def test_routine_warnings_do_not_scale_with_chart_size():
    # One ambiguity review warning on every node, and both terminal connectors
    # inferred: the pattern every verified correct chart shows.
    def routine(length):
        graph = chain(length)
        diagnostics = [warn("PROGRAMMING_OCR_AMBIGUITY", node.id) for node in graph.nodes]
        diagnostics += [warn("INFERRED_TERMINAL_DIRECTION", "n1", "n2"),
                        warn("INFERRED_TERMINAL_DIRECTION", f"n{length - 1}", f"n{length}")]
        return score(graph, diagnostics)
    small, large = routine(4), routine(14)
    assert small["grade"] == large["grade"] == "Excellent"
    assert large["overall"] >= small["overall"]


def test_one_unreliable_node_limits_the_overall_score():
    graph = chain(10)
    graph.nodes[4].confidence = 0.50
    result = score(graph, [warn("LOW_OCR_CONFIDENCE", "n5")])
    assert result["ocr"] < 60
    assert result["overall"] <= result["critical"] + CRITICAL_MARGIN
    assert result["grade"] in {"Low", "Very Low"}
    assert result["cap"] == "weakest_component"


def test_blocking_error_caps_overall_even_when_components_are_strong():
    graph = chain(8)
    result = score(graph, [Diagnostic("error", "UNSTRUCTURED_FLOW", "", "", ["n3"])])
    assert result["weighted"] > BLOCKED_CAP
    assert result["overall"] == BLOCKED_CAP and result["grade"] == "Low"
    assert result["cap"] == "blocking_error" and result["blocking_errors"] == 1


def test_component_error_caps_that_component():
    graph = chain(6)
    result = score(graph, [Diagnostic("error", "AMBIGUOUS_ARROW_DIRECTION", "", "", ["n2", "n3"])])
    assert result["graph"] <= BLOCKED_CAP


def test_unsupported_shape_is_visible_in_shape_and_overall():
    graph = chain(6, kinds=["terminal", "process", "unknown", "process", "process", "terminal"])
    result = score(graph, [Diagnostic("error", "UNSUPPORTED_SHAPE", "", "", ["n3"])])
    assert result["shape"] < 50
    assert result["overall"] <= result["shape"] + CRITICAL_MARGIN


def test_branch_label_ocr_warning_affects_label_not_every_node():
    graph = chain(6)
    labelled = score(graph, [warn("LOW_OCR_CONFIDENCE", "e2")])
    page_wide = score(graph, [Diagnostic("warning", "LOW_OCR_CONFIDENCE", "", "", [])])
    assert labelled["ocr"] > page_wide["ocr"]


def test_inferred_connectors_cost_more_than_terminal_inference():
    graph = chain(6)
    terminal = score(graph, [warn("INFERRED_TERMINAL_DIRECTION", "n1", "n2")])
    inferred = score(graph, [warn("INFERRED_ARROW_DIRECTION", "n2", "n3")])
    dangling = score(graph, [warn("DANGLING_ARROW", "n3")])
    assert terminal["graph"] > inferred["graph"] > dangling["graph"]
