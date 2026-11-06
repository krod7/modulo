"""Regression tests for conservative recovery of incomplete decision branches."""
import numpy as np

from backend.models import BoundingBox, Edge, FlowGraph, Node, PipelineContext
from backend.services.graph_service import (
    infer_high_confidence_missing_edges,
    infer_obvious_missing_branch_to_end,
)


def _node(node_id, kind, text, box):
    return Node(node_id, kind, text, BoundingBox(*box))


def test_missing_vertical_sibling_branch_requires_visible_stroke_and_preserves_end_recovery():
    import cv2

    decision = _node("decision", "decision", "number % 2 == 0", (100, 100, 200, 100))
    lateral = _node("lateral", "output", 'Output "odd"', (350, 120, 100, 60))
    lower = _node("lower", "output", 'Output "even"', (150, 270, 100, 60))
    end = _node("end", "terminal", "End", (150, 380, 100, 60))
    nodes = [decision, lateral, lower, end]
    graph = FlowGraph(nodes=nodes, edges=[
        Edge("e1", "decision", "lateral", "False"),
        Edge("e2", "lower", "end"),
    ])
    stroke_mask = np.zeros((500, 600), dtype=np.uint8)
    # Visible downward branch from the decision's bottom apex.
    cv2.line(stroke_mask, (200, 200), (200, 270), 255, 3)
    # Visible orthogonal continuation from the lateral output to End.
    route = [(450, 150), (468, 150), (468, 410), (250, 410)]
    for first, second in zip(route, route[1:]):
        cv2.line(stroke_mask, first, second, 255, 3)
    context = PipelineContext(elements=nodes, graph=graph, stroke_mask=stroke_mask)

    recovered = infer_high_confidence_missing_edges(context, graph)
    inferred = [edge for edge in graph.edges if edge.source == "decision" and edge.target == "lower"]
    assert recovered == 1
    assert len(inferred) == 1
    assert inferred[0].points[0] == (200, 200)
    assert inferred[0].points[-1] == (200, 270)

    assert infer_obvious_missing_branch_to_end(context, graph)
    assert any(edge.source == "lateral" and edge.target == "end" for edge in graph.edges)
    assert any(item.code == "INFERRED_DECISION_BRANCH" for item in context.diagnostics)
    assert any(item.code == "INFERRED_BRANCH_TO_END" for item in context.diagnostics)


def test_missing_vertical_sibling_branch_is_not_inferred_from_geometry_alone():
    decision = _node("decision", "decision", "number % 2 == 0", (100, 100, 200, 100))
    lateral = _node("lateral", "output", 'Output "odd"', (350, 120, 100, 60))
    lower = _node("lower", "output", 'Output "even"', (150, 270, 100, 60))
    end = _node("end", "terminal", "End", (150, 380, 100, 60))
    nodes = [decision, lateral, lower, end]
    graph = FlowGraph(nodes=nodes, edges=[
        Edge("e1", "decision", "lateral", "False"),
        Edge("e2", "lower", "end"),
    ])
    context = PipelineContext(
        elements=nodes, graph=graph,
        stroke_mask=np.zeros((500, 600), dtype=np.uint8),
    )

    assert infer_high_confidence_missing_edges(context, graph) == 0
    assert not any(edge.source == "decision" and edge.target == "lower" for edge in graph.edges)


def test_missing_sibling_branch_is_not_guessed_when_multiple_end_connected_candidates_exist():
    decision = _node("decision", "decision", "x > 0", (100, 100, 200, 100))
    known = _node("known", "output", 'Output "known"', (350, 120, 100, 60))
    candidate_a = _node("candidate_a", "output", 'Output "a"', (120, 260, 100, 60))
    candidate_b = _node("candidate_b", "process", "value = 1", (180, 270, 100, 60))
    end = _node("end", "terminal", "End", (150, 400, 100, 60))
    nodes = [decision, known, candidate_a, candidate_b, end]
    graph = FlowGraph(nodes=nodes, edges=[
        Edge("e1", "decision", "known", "No"),
        Edge("e2", "candidate_a", "end"),
        Edge("e3", "candidate_b", "end"),
    ])
    context = PipelineContext(
        elements=nodes,
        graph=graph,
        stroke_mask=np.zeros((520, 600), dtype=np.uint8),
    )

    assert infer_high_confidence_missing_edges(context, graph) == 0
    assert not any(edge.source == "decision" and edge.target != "known" for edge in graph.edges)
