"""Regression tests for connector/OCR recognition improvements."""
import numpy as np
import pytest

from backend.models import PipelineContext
from backend.services import vision_service


def test_decision_ocr_uses_single_line_segmentation(monkeypatch):
    calls = []

    monkeypatch.setattr(vision_service, "preprocess_ocr_candidates", lambda image, shape_type: [image])

    def fake_read_words(image, context, psm, origin):
        calls.append(psm)
        return "num < 0", []

    monkeypatch.setattr(vision_service, "read_words", fake_read_words)
    image = np.full((40, 120), 255, dtype=np.uint8)
    context = PipelineContext()
    text, _, _ = vision_service.read_best_words(image, context, shape_type="decision", return_candidates=True)

    assert text == "num < 0"
    assert 7 in calls  # single-line mode catches short conditions/operators
    assert 6 in calls and 11 in calls  # keep complementary fallback modes


def test_decision_candidate_ranking_rewards_valid_expression():
    candidates = [
        ("NN OY m<00", []),
        ("num < 0", []),
    ]
    text, _, alternatives = vision_service.select_ocr_candidate(candidates, "decision")
    assert text == "num < 0"
    assert any(item["text"] == "NN OY m<00" for item in alternatives)


def test_print_keyword_repair_is_candidate_not_unconditional_rewrite():
    repaired = vision_service._repair_ocr_text('rint "message"', "process")
    assert 'Print "message"' in repaired
    assert vision_service._repair_ocr_text("rint something", "process") == []


def test_output_ocr_repairs_near_miss_common_words_without_touching_identifiers():
    from backend.services.vision_service import _repair_ocr_text

    repaired = _repair_ocr_text('Print "Factorial is no\' Jefined for negative numbers"', "process")
    assert 'Print "Factorial is not defined for negative numbers"' in repaired
    # Unknown identifiers are not silently rewritten to ordinary vocabulary.
    assert _repair_ocr_text("print customer_count", "process") == []


def test_output_candidate_selection_prefers_coherent_repaired_sentence():
    text, _, _ = vision_service.select_ocr_candidate(
        [('Print "Factorial is no\' Jefined for negative numbers"', [])],
        "process",
    )
    assert text == 'Print "Factorial is not defined for negative numbers"'



def test_decision_ocr_adds_enlarged_center_crop(monkeypatch):
    calls = []

    monkeypatch.setattr(vision_service, "preprocess_ocr_candidates", lambda image, shape_type: [image])

    def fake_read_words(image, context, psm, origin):
        calls.append((image.shape, psm, origin))
        # The center crop is upscaled; simulate the better reading it is intended
        # to recover from a large diamond with a small centered label.
        if image.shape[0] < 180 and psm == 7:
            return "num < 0", []
        return "NN OY m<0O", []

    monkeypatch.setattr(vision_service, "read_words", fake_read_words)
    image = np.full((130, 240), 255, dtype=np.uint8)
    text, _, alternatives = vision_service.read_best_words(
        image, PipelineContext(), origin=(100, 200), shape_type="decision", return_candidates=True
    )
    assert any(shape[0] < image.shape[0] * 2 for shape, _, _ in calls)
    assert any(psm == 7 and shape[0] == int(image.shape[0] * .60) * 2 for shape, psm, _ in calls)
    assert text == "num < 0"
    assert any(item["text"] == "NN OY m<0O" for item in alternatives)


def test_terminal_semantics_recovers_occluded_end_arrowhead():
    import cv2
    from backend.config import Settings
    from backend.models import BoundingBox, Node, PipelineContext
    from backend.services.graph_service import GraphService

    source = Node("n1", "process", "Print factorial", BoundingBox(40, 10, 20, 12),
                  polygon=[(40, 10), (60, 10), (60, 22), (40, 22)])
    end = Node("n2", "terminal", "End", BoundingBox(20, 48, 20, 20),
               polygon=[(20, 48), (40, 48), (40, 68), (20, 68)])
    mask = np.zeros((90, 90), dtype=np.uint8)
    # An elbow-shaped connector ending at End. Its arrowhead is deliberately
    # absent, representing the portion lost when node pixels are removed.
    cv2.line(mask, (50, 22), (50, 35), 255, 2)
    cv2.line(mask, (50, 35), (30, 35), 255, 2)
    cv2.line(mask, (30, 35), (30, 48), 255, 2)
    context = PipelineContext(settings=Settings(), elements=[source, end], stroke_mask=mask)
    GraphService().run(context)
    assert any(edge.source == "n1" and edge.target == "n2" for edge in context.graph.edges)
    assert any(item.code == "INFERRED_TERMINAL_DIRECTION" for item in context.diagnostics)


def test_repaired_output_candidate_outranks_ocr_gibberish():
    text, _, alternatives = vision_service.select_ocr_candidate(
        [('Print "Factorial is no\' Jefined for negative numbers"', [])],
        "process",
    )
    assert text == 'Print "Factorial is not defined for negative numbers"'
    assert any("Jefined" in item["text"] for item in alternatives)



def test_instruction_verbs_refine_rectangular_shape_semantics():
    from backend.services.vision_service import infer_instruction_kind

    assert infer_instruction_kind("process", "Input number") == ("input", "Input number")
    assert infer_instruction_kind("process", "Print factorial") == ("output", "Output factorial")
    assert infer_instruction_kind("process", 'Print "Factorial is not defined') == (
        "output", 'Output "Factorial is not defined"'
    )
    assert infer_instruction_kind("process", 'Print "Factorial is not defined"') == (
        "output", 'Output "Factorial is not defined"'
    )
    assert infer_instruction_kind("process", "Calculate factorial") == (
        "process", "Calculate factorial"
    )


def test_aligned_connector_fragments_are_stitched_without_global_closing():
    from backend.models import BoundingBox, Node
    from backend.services.graph_service import bridge_aligned_connector_gaps

    mask = np.zeros((300, 200), dtype=np.uint8)
    # A vertical connector split by a branch label's erased OCR box.
    import cv2
    cv2.line(mask, (80, 30), (80, 100), 255, 5)
    cv2.line(mask, (82, 140), (82, 240), 255, 5)
    node = Node("n1", "process", "A", BoundingBox(20, 10, 20, 20))
    repaired = bridge_aligned_connector_gaps(mask, [node])
    count, labels = cv2.connectedComponents(repaired, connectivity=8)
    upper = labels[80, 80]
    lower = labels[160, 82]
    assert upper != 0 and upper == lower



@pytest.mark.skipif(
    __import__("os").getenv("FLOWCHART_RUN_PIPELINE") != "1",
    reason="set FLOWCHART_RUN_PIPELINE=1 to run OCR-dependent uploaded-diagram regression",
)
def test_uploaded_factorial_flowchart_reconstructs_all_seven_connections():
    from pathlib import Path
    from backend.config import Settings
    from backend.pipeline import analyze

    fixture = Path(__file__).parent / "fixtures" / "flowcharts" / "factorial_curved_connectors.jpg"
    result = analyze(fixture.read_bytes(), filename=fixture.name, settings=Settings())
    actual = {(edge.source, edge.target, edge.label) for edge in result.graph.edges}
    expected = {
        ("n1", "n2", None),
        ("n2", "n3", None),
        ("n3", "n5", "Yes"),
        ("n3", "n4", "No"),
        ("n4", "n6", None),
        ("n5", "n7", None),
        ("n6", "n7", None),
    }
    assert actual == expected
    nodes = {node.id: node for node in result.elements}
    assert nodes["n3"].text == "num < 0"
    assert nodes["n5"].text == 'Output "Factorial is not defined for negative numbers"'
    assert nodes["n2"].type == "input"
    assert nodes["n5"].type == "output"



def test_decision_instruction_preserves_leading_if_for_review():
    from backend.services.vision_service import infer_instruction_kind

    kind, text = infer_instruction_kind("decision", "If number % 2 == 0")
    assert kind == "decision"
    assert text == "If number % 2 == 0"


def test_elbow_branch_gap_uses_narrow_endpoint_not_component_aspect_ratio():
    import cv2
    from backend.models import BoundingBox, Node
    from backend.services.graph_service import bridge_aligned_connector_gaps

    mask = np.zeros((240, 260), dtype=np.uint8)
    # First fragment is an L-shaped connector with a wide overall bbox. Its
    # exposed lower endpoint is narrow and aligned with the second fragment.
    cv2.line(mask, (35, 65), (190, 65), 255, 4)
    cv2.line(mask, (37, 65), (37, 125), 255, 4)
    cv2.line(mask, (37, 165), (37, 220), 255, 4)
    node = Node("n1", "decision", "x > 0", BoundingBox(170, 20, 50, 50),
                polygon=[(195, 20), (220, 45), (195, 70), (170, 45)])
    repaired = bridge_aligned_connector_gaps(mask, [node])
    count, labels = cv2.connectedComponents(repaired, connectivity=8)
    assert count == 2
    assert labels[100, 37] == labels[180, 37] != 0


def test_decision_ocr_repairs_stray_keyword_punctuation_and_missing_equality_stroke():
    text, _, alternatives = vision_service.select_ocr_candidate(
        [('If . Number%2=0', [])], "decision"
    )
    assert text == 'If Number%2 == 0'
    assert any(item["text"] == 'If . Number%2=0' for item in alternatives)


def test_input_notation_accepts_descriptive_variable_label():
    from backend.notation import INPUT

    match = INPUT.fullmatch('Input variable: Number')
    assert match is not None
    assert match.group(1) == 'Number'


@pytest.mark.skipif(
    __import__("os").getenv("FLOWCHART_RUN_PIPELINE") != "1",
    reason="set FLOWCHART_RUN_PIPELINE=1 to run OCR-dependent modulo-flowchart regression",
)
def test_modulo_even_odd_flowchart_recognizes_nodes_edges_and_decision_text():
    from pathlib import Path
    from backend.config import Settings
    from backend.pipeline import analyze

    fixture = Path(__file__).parent / "fixtures" / "flowcharts" / "modulo_even_odd.png"
    result = analyze(fixture.read_bytes(), filename=fixture.name, settings=Settings())
    nodes = {node.id: node for node in result.elements}
    assert len(nodes) == 6
    assert nodes["n2"].text == "Input variable: Number"
    assert nodes["n3"].text == "If Number%2 == 0"
    assert nodes["n4"].text == 'Output "Number is odd"'
    assert nodes["n5"].text == 'Output "number is even"'
    edges = {(edge.source, edge.target): edge.label for edge in result.graph.edges}
    assert set(edges) == {
        ("n1", "n2"), ("n2", "n3"), ("n3", "n4"),
        ("n3", "n5"), ("n5", "n6"),
    }
    # The connector topology is the invariant; OCR may normalize the visible
    # True/False labels to Yes/No depending on the local OCR engine.
    assert edges[("n3", "n4")] in {"False", "No"}
    assert edges[("n3", "n5")] in {"True", "Yes"}


def test_obvious_missing_sibling_branch_to_end_is_inferred_with_review_warning():
    from backend.models import BoundingBox, Edge, FlowGraph, Node, PipelineContext
    from backend.services.graph_service import infer_obvious_missing_branch_to_end

    decision = Node("n3", "decision", "If n % 2 == 0", BoundingBox(100, 100, 100, 80))
    odd = Node("n4", "output", 'Output "odd"', BoundingBox(240, 110, 100, 50))
    even = Node("n5", "output", 'Output "even"', BoundingBox(100, 220, 100, 50))
    end = Node("n6", "terminal", "End", BoundingBox(100, 310, 100, 50))
    graph = FlowGraph(nodes=[decision, odd, even, end], edges=[
        Edge("e1", "n3", "n4", "False", [(200, 140), (240, 140)]),
        Edge("e2", "n3", "n5", "True", [(150, 180), (150, 220)]),
        Edge("e3", "n5", "n6", None, [(150, 270), (150, 310)]),
    ])
    import cv2
    stroke_mask = np.zeros((400, 500), dtype=np.uint8)
    proposed_route = [(340, 135), (358, 135), (358, 335), (200, 335)]
    for first, second in zip(proposed_route, proposed_route[1:]):
        cv2.line(stroke_mask, first, second, 255, 3)
    context = PipelineContext(elements=graph.nodes, stroke_mask=stroke_mask)
    assert infer_obvious_missing_branch_to_end(context, graph)
    assert any((edge.source, edge.target) == ("n4", "n6") for edge in graph.edges)
    assert any(item.code == "INFERRED_BRANCH_TO_END" and item.severity == "warning" for item in context.diagnostics)


def test_missing_branch_inference_does_not_guess_for_unrelated_or_multiple_ends():
    from backend.models import BoundingBox, Edge, FlowGraph, Node, PipelineContext
    from backend.services.graph_service import infer_obvious_missing_branch_to_end

    decision = Node("n3", "decision", "If x", BoundingBox(100, 100, 100, 80))
    odd = Node("n4", "output", "Output A", BoundingBox(240, 110, 100, 50))
    even = Node("n5", "output", "Output B", BoundingBox(100, 220, 100, 50))
    end1 = Node("n6", "terminal", "End", BoundingBox(100, 310, 100, 50))
    end2 = Node("n7", "terminal", "End", BoundingBox(300, 310, 100, 50))
    graph = FlowGraph(nodes=[decision, odd, even, end1, end2], edges=[
        Edge("e1", "n3", "n4", "False"), Edge("e2", "n3", "n5", "True"),
        Edge("e3", "n5", "n6"),
    ])
    context = PipelineContext(elements=graph.nodes, stroke_mask=np.zeros((400, 500), dtype=np.uint8))
    assert not infer_obvious_missing_branch_to_end(context, graph)
    assert len(graph.edges) == 3


def test_modulo_even_odd_flowchart_does_not_invent_unshown_output_to_end():
    from pathlib import Path
    from backend.config import Settings
    from backend.pipeline import analyze

    fixture = Path(__file__).parent / "fixtures" / "flowcharts" / "modulo_even_odd.png"
    result = analyze(fixture.read_bytes(), filename=fixture.name, settings=Settings())
    nodes = {node.id: node for node in result.elements}
    odd_node = next(node for node in result.elements if "odd" in node.text.casefold())
    end_node = next(node for node in result.elements if node.type == "terminal" and node.text.strip().casefold() == "end")
    assert not any(edge.source == odd_node.id and edge.target == end_node.id for edge in result.graph.edges)
    assert not any(item.code == "INFERRED_BRANCH_TO_END" for item in result.diagnostics)
    # The five actually drawn connectors, including the branch interrupted by
    # its "True" label, must all be present.
    assert {(edge.source, edge.target) for edge in result.graph.edges} == {
        ("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n3", "n5"), ("n5", "n6")
    }
    recovered = next(edge for edge in result.graph.edges if edge.source == "n3" and edge.target == "n5")
    assert abs(recovered.points[0][0] - recovered.points[-1][0]) <= 2
    assert len(nodes) == 6


def test_factorial_branch_chart_recovers_visible_vertical_and_missing_decision_edges():
    """A broken connector trace should not erase two strongly supported edges."""
    from pathlib import Path
    from backend.config import Settings
    from backend.pipeline import analyze

    fixture = Path(__file__).parent / "fixtures" / "flowcharts" / "factorial_negative_branch.jpg"
    result = analyze(fixture.read_bytes(), filename=fixture.name, settings=Settings())
    nodes = {node.id: node for node in result.elements}
    assert len(nodes) == 7
    edges = {(edge.source, edge.target) for edge in result.graph.edges}
    assert {("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n3", "n5"),
            ("n4", "n6"), ("n5", "n7"), ("n6", "n7")} <= edges
    # The short vertical fragments are now stitched before graph extraction,
    # so this is a traced connector rather than a geometry-only recovery.
    assert next(edge for edge in result.graph.edges if edge.source == "n5" and edge.target == "n7").points
    assert any(d.code == "INFERRED_TERMINAL_DIRECTION" for d in result.diagnostics)
    # The Yes/No words overlap the connector strokes in this fixture, so OCR
    # may omit them. Recover labels from the opposite-side departure geometry.
    decision_edges = [edge for edge in result.graph.edges if edge.source == "n3"]
    assert {(edge.target, edge.label) for edge in decision_edges} == {("n5", "Yes"), ("n4", "No")}
    assert not any(d.code == "INVALID_BRANCH_LABELS" for d in result.diagnostics)
    assert any(d.code == "INFERRED_BRANCH_LABELS" for d in result.diagnostics)
    # Recovered edges are real flow edges, not display-only structural hints:
    # validation and structured pseudocode generation must complete as well.
    assert not result.has_errors
    assert "validation" in result.completed_stages
    assert "logic" in result.completed_stages
    assert "pseudocode" in result.completed_stages


def test_branch_label_normalization_tolerates_surrounding_ocr_punctuation():
    from backend.notation import normalize_label

    assert normalize_label(" Yes. ") == "Yes"
    assert normalize_label("(No),") == "No"
    assert normalize_label("maybe") is None


def test_validation_reconciles_one_missing_decision_label_from_opposite_departures():
    from backend.models import BoundingBox, Edge, FlowGraph, Node, PipelineContext
    from backend.services.validation_service import _reconcile_decision_labels

    decision = Node("n3", "decision", "num < 0", BoundingBox(100, 100, 100, 80))
    left = Node("n4", "output", "negative", BoundingBox(0, 220, 80, 50))
    right = Node("n5", "process", "calculate", BoundingBox(220, 220, 100, 50))
    left_edge = Edge("e1", "n3", "n4", "Yes.", points=[(100, 140), (70, 180), (40, 220)])
    right_edge = Edge("e2", "n3", "n5", None, points=[(200, 140), (250, 180), (270, 220)])
    context = PipelineContext(elements=[decision, left, right], graph=FlowGraph(
        nodes=[decision, left, right], edges=[left_edge, right_edge]))

    _reconcile_decision_labels(context)

    assert left_edge.label == "Yes"
    assert right_edge.label == "No"
    assert any(d.code == "INFERRED_BRANCH_LABELS" for d in context.diagnostics)



def test_short_vertical_connector_fragments_are_stitched_across_branch_label():
    import cv2
    from backend.models import BoundingBox, Node
    from backend.services.graph_service import bridge_aligned_connector_gaps

    mask = np.zeros((280, 200), dtype=np.uint8)
    cv2.line(mask, (100, 30), (100, 100), 255, 5)
    # The lower shaft is shorter than the former 35-pixel component cutoff.
    cv2.line(mask, (101, 140), (101, 168), 255, 5)
    # A detached OCR label occupies the gap but must not be mistaken for a
    # connector endpoint or prevent the two aligned shaft fragments joining.
    cv2.putText(mask, "True", (55, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    node = Node("n1", "process", "A", BoundingBox(20, 10, 20, 20))
    repaired = bridge_aligned_connector_gaps(mask, [node])
    count, labels = cv2.connectedComponents(repaired, connectivity=8)
    assert labels[80, 100] != 0
    assert labels[155, 101] == labels[80, 100]


def test_missing_branch_to_end_requires_visible_connector_ink():
    from backend.models import BoundingBox, Edge, FlowGraph, Node, PipelineContext
    from backend.services.graph_service import infer_obvious_missing_branch_to_end

    decision = Node("n1", "decision", "x > 0", BoundingBox(100, 80, 80, 60))
    connected = Node("n2", "output", "positive", BoundingBox(210, 110, 70, 40))
    dangling = Node("n3", "output", "negative", BoundingBox(210, 180, 70, 40))
    end = Node("n4", "terminal", "End", BoundingBox(100, 260, 80, 40))
    graph = FlowGraph(nodes=[decision, connected, dangling, end], edges=[
        Edge("e1", "n1", "n2"), Edge("e2", "n2", "n4")
    ])
    context = PipelineContext(elements=graph.nodes, graph=graph,
                              stroke_mask=np.zeros((340, 340), dtype=np.uint8))

    assert infer_obvious_missing_branch_to_end(context, graph) is False
    assert {(edge.source, edge.target) for edge in graph.edges} == {
        ("n1", "n2"), ("n2", "n4")
    }
