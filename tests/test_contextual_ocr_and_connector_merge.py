import cv2
import numpy as np
import pytest

from backend.models import BoundingBox, Edge, FlowGraph, Node, PipelineContext
from backend.services.vision_service import _reconcile_low_confidence_comparison_thresholds
from backend.services.graph_service import infer_dangling_arrow_to_existing_edge


def test_low_confidence_zero_one_comparison_is_contextually_reconciled_and_preserved():
    uncertain = Node("n1", "decision", "if count > 1", BoundingBox(0, 0, 100, 60), .74)
    corroborating = Node("n2", "decision", "if count < 0", BoundingBox(0, 100, 100, 60), .90)
    _reconcile_low_confidence_comparison_thresholds([uncertain, corroborating])
    assert uncertain.text == "if count > 0"
    assert any(item["text"] == "if count > 1" for item in uncertain.ocr_candidates)
    assert "CONTEXT_CORRECTED_OCR" in uncertain.flags


def test_clear_or_unpaired_numeric_comparisons_are_not_rewritten():
    clear = Node("n1", "decision", "if count > 1", BoundingBox(0, 0, 100, 60), .95)
    unpaired = Node("n2", "decision", "if total > 1", BoundingBox(0, 100, 100, 60), .70)
    _reconcile_low_confidence_comparison_thresholds([clear, unpaired])
    assert clear.text == "if count > 1"
    assert unpaired.text == "if total > 1"


def test_arrowhead_pointing_into_existing_connector_recovers_merge_edge():
    # Draw a right-angle branch that ends in a left-facing arrowhead just short
    # of an already traced vertical connector.
    mask = np.zeros((360, 360), dtype=np.uint8)
    cv2.line(mask, (220, 120), (220, 180), 255, 4)
    cv2.line(mask, (220, 180), (120, 180), 255, 4)
    cv2.line(mask, (120, 180), (100, 168), 255, 4)
    cv2.line(mask, (120, 180), (100, 192), 255, 4)
    ys, xs = np.where(mask > 0)
    points = np.column_stack((xs, ys)).astype(np.int32)

    source = Node("n1", "output", "Output result", BoundingBox(200, 80, 40, 40))
    target = Node("n2", "decision", "if ready", BoundingBox(70, 280, 40, 40))
    upstream = Node("n3", "decision", "if valid", BoundingBox(70, 100, 40, 40))
    receiving = Edge("e1", "n3", "n2", points=[(90, 140), (90, 280)])
    graph = FlowGraph(nodes=[source, target, upstream], edges=[receiving])
    context = PipelineContext(graph=graph)

    count = infer_dangling_arrow_to_existing_edge(context, graph, [(source, points)], mask.shape)
    assert count == 1
    assert any(edge.source == "n1" and edge.target == "n2" for edge in graph.edges)
    assert any(item.code == "INFERRED_ARROW_TO_CONNECTOR" for item in context.diagnostics)


def test_node_outline_cleanup_preserves_arrowheads_near_shape_boundaries():
    from backend.models import BoundingBox, Node
    from backend.services.graph_service import suppress_node_outline_artifacts

    mask = np.zeros((240, 1000), dtype=np.uint8)
    node = Node(
        "n1", "decision", "ready?", BoundingBox(50, 40, 50, 40),
        polygon=[(50, 40), (100, 40), (100, 80), (50, 80)],
    )
    # A left-facing arrow enters the right edge. Its tip and V are close to
    # the node boundary, as they commonly are after rasterization.
    cv2.line(mask, (150, 60), (106, 60), 255, 3)
    cv2.line(mask, (106, 60), (116, 50), 255, 3)
    cv2.line(mask, (106, 60), (116, 70), 255, 3)
    # Include the node outline so the cleanup still has border artifacts to
    # remove while preserving the nearby external arrowhead.
    cv2.rectangle(mask, (50, 40), (100, 80), 255, 2)

    cleaned = suppress_node_outline_artifacts(mask, [node])
    assert cleaned[60, 106] > 0
    assert np.count_nonzero(cleaned[40:81, 50:101]) < np.count_nonzero(mask[40:81, 50:101])


@pytest.mark.skipif(
    __import__("os").getenv("FLOWCHART_RUN_PIPELINE") != "1",
    reason="set FLOWCHART_RUN_PIPELINE=1 to run OCR-dependent flowchart regressions",
)
def test_countdown_loop_preserves_both_loop_directions_and_exit_branch():
    from pathlib import Path
    from backend.config import Settings
    from backend.pipeline import analyze

    fixture = Path(__file__).parents[1] / "samples" / "countdown.png"
    result = analyze(fixture.read_bytes(), filename=fixture.name, settings=Settings())
    actual = {(edge.source, edge.target, edge.label) for edge in result.graph.edges}
    assert actual == {
        ("n1", "n2", None),
        ("n2", "n3", None),
        ("n3", "n4", "Yes"),
        ("n4", "n3", None),
        ("n3", "n5", "No"),
    }
    assert not any(
        item.severity == "error" and item.code in {
            "AMBIGUOUS_ARROW_DIRECTION", "AMBIGUOUS_CONNECTOR", "DANGLING_ARROW"
        }
        for item in result.diagnostics
    )
