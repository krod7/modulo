"""Regression coverage for branch merge selection and structured IR."""
import shutil
from pathlib import Path

import pytest

from backend.models import BoundingBox, Edge, FlowGraph, Node, PipelineContext
from backend.services.logic_service import LogicService


def _node(node_id, kind, text):
    return Node(node_id, kind, text, BoundingBox(0, 0, 20, 20))


def test_nested_decision_uses_nearest_inner_merge_not_outer_end_region():
    nodes = [
        _node("start", "terminal", "Start"),
        _node("outer", "decision", "outer_flag == 1"),
        _node("inner", "decision", "inner_flag == 1"),
        _node("left", "process", "left_value = 1"),
        _node("right", "process", "right_value = 1"),
        _node("inner_merge", "process", "merged_value = 1"),
        _node("outer_else", "process", "fallback_value = 1"),
        _node("outer_merge", "process", "finished_value = 1"),
        _node("end", "terminal", "End"),
    ]
    edges = [
        Edge("e1", "start", "outer"),
        Edge("e2", "outer", "inner", "Yes"),
        Edge("e3", "outer", "outer_else", "No"),
        Edge("e4", "inner", "left", "Yes"),
        Edge("e5", "inner", "right", "No"),
        Edge("e6", "left", "inner_merge"),
        Edge("e7", "right", "inner_merge"),
        Edge("e8", "inner_merge", "outer_merge"),
        Edge("e9", "outer_else", "outer_merge"),
        Edge("e10", "outer_merge", "end"),
    ]
    context = PipelineContext(elements=nodes, graph=FlowGraph(
        nodes=nodes, edges=edges, start_id="start", end_ids=["end"]
    ))

    LogicService().run(context)

    assert not context.has_errors
    assert context.ir is not None
    outer_if = next(item for item in context.ir.children if item.kind == "If")
    inner_if = next(item for item in outer_if.children if item.kind == "If")
    assert [item.source_ids[0] for item in inner_if.children] == ["left"]
    assert [item.source_ids[0] for item in inner_if.otherwise] == ["right"]
    assert [item.source_ids[0] for item in outer_if.otherwise] == ["outer_else"]
    assert [item.source_ids[0] for item in context.ir.children[-2:]] == ["outer_merge", "end"]


def test_sequential_decisions_do_not_absorb_their_shared_continuation():
    nodes = [
        _node("start", "terminal", "Start"),
        _node("first", "decision", "a == 1"),
        _node("a_yes", "process", "a_value = 1"),
        _node("a_no", "process", "a_value = 0"),
        _node("between", "process", "middle_value = 1"),
        _node("second", "decision", "b == 1"),
        _node("b_yes", "process", "b_value = 1"),
        _node("b_no", "process", "b_value = 0"),
        _node("end", "terminal", "End"),
    ]
    edges = [
        Edge("e1", "start", "first"),
        Edge("e2", "first", "a_yes", "Yes"), Edge("e3", "first", "a_no", "No"),
        Edge("e4", "a_yes", "between"), Edge("e5", "a_no", "between"),
        Edge("e6", "between", "second"),
        Edge("e7", "second", "b_yes", "Yes"), Edge("e8", "second", "b_no", "No"),
        Edge("e9", "b_yes", "end"), Edge("e10", "b_no", "end"),
    ]
    context = PipelineContext(elements=nodes, graph=FlowGraph(
        nodes=nodes, edges=edges, start_id="start", end_ids=["end"]
    ))

    LogicService().run(context)

    assert not context.has_errors
    decisions = [item for item in context.ir.children if item.kind == "If"]
    assert len(decisions) == 2
    assert decisions[0].condition == "a == 1"
    assert decisions[1].condition == "b == 1"
    assert any(item.source_ids == ["between"] for item in context.ir.children)


def test_true_false_parity_branches_render_as_if_else_not_sequential_outputs():
    from backend.services.pseudocode_service import PseudocodeService

    nodes = [
        _node("start", "terminal", "Start"),
        _node("input", "input", "Input number"),
        _node("check", "decision", "number % 2 == 0"),
        _node("odd", "output", 'Output "Number is odd"'),
        _node("even", "output", 'Output "number is even"'),
        _node("end", "terminal", "End"),
    ]
    edges = [
        Edge("e1", "start", "input"),
        Edge("e2", "input", "check"),
        Edge("e3", "check", "odd", "False"),
        Edge("e4", "check", "even", "True"),
        Edge("e5", "odd", "end"),
        Edge("e6", "even", "end"),
    ]
    context = PipelineContext(elements=nodes, graph=FlowGraph(
        nodes=nodes, edges=edges, start_id="start", end_ids=["end"]
    ))

    LogicService().run(context)
    PseudocodeService().run(context)

    rendered = [line.text for line in context.pseudocode]
    assert "IF number % 2 == 0" in rendered
    assert "ELSE" in rendered
    assert rendered.index('OUTPUT "number is even"') < rendered.index("ELSE")
    assert rendered.index('OUTPUT "Number is odd"') > rendered.index("ELSE")
    assert rendered.count('OUTPUT "Number is odd"') == 1
    assert rendered.count('OUTPUT "number is even"') == 1


# --- Branches that finish on an output shape without an arrow to End -----------

def _analyze_graph(nodes, edges):
    """Run validation, logic and pseudocode over a hand-built graph."""
    from backend.config import Settings
    from backend.services.pseudocode_service import PseudocodeService
    from backend.services.validation_service import ValidationService

    context = PipelineContext(
        elements=nodes, settings=Settings(),
        graph=FlowGraph(nodes=nodes, edges=edges),
    )
    for service in (ValidationService(), LogicService(), PseudocodeService()):
        if context.has_errors:
            break
        service.run(context)
    return context


def _parity_nodes(odd_kind="output", odd_text='Output "Number is odd"'):
    return [
        _node("start", "terminal", "Start"),
        _node("input", "input", "Input Number"),
        _node("check", "decision", "Number % 2 == 0"),
        _node("odd", odd_kind, odd_text),
        _node("even", "output", 'Output "number is even"'),
        _node("end", "terminal", "End"),
    ]


def _parity_edges():
    # The odd output is intentionally NOT connected to End.
    return [
        Edge("e1", "start", "input"),
        Edge("e2", "input", "check"),
        Edge("e3", "check", "odd", "False"),
        Edge("e4", "check", "even", "True"),
        Edge("e5", "even", "end"),
    ]


def test_output_finishing_a_decision_branch_is_valid_without_inventing_a_connector():
    nodes, edges = _parity_nodes(), _parity_edges()
    context = _analyze_graph(nodes, edges)

    assert not context.has_errors, [d.code for d in context.diagnostics]
    assert context.ir is not None
    # No artificial connector: the graph still holds exactly the drawn arrows.
    assert {(e.source, e.target) for e in context.graph.edges} == {
        ("start", "input"), ("input", "check"), ("check", "odd"),
        ("check", "even"), ("even", "end"),
    }
    assert not any(e.source == "odd" for e in context.graph.edges)
    # The reviewer is told about it, and the warning is attached to the shape.
    notes = [d for d in context.diagnostics if d.code == "BRANCH_ENDS_WITHOUT_END"]
    assert len(notes) == 1 and notes[0].severity == "warning" and notes[0].element_ids == ["odd"]


def test_branch_finishing_on_output_renders_if_else_then_a_single_end():
    context = _analyze_graph(_parity_nodes(), _parity_edges())

    rendered = [line.text for line in context.pseudocode]
    assert rendered == [
        "START", "INPUT Number", "IF Number % 2 == 0",
        'OUTPUT "number is even"', "ELSE", 'OUTPUT "Number is odd"',
        "END IF", "END",
    ]
    indents = {line.text: line.indent for line in context.pseudocode}
    assert indents['OUTPUT "Number is odd"'] == indents['OUTPUT "number is even"'] == 1
    assert indents["END"] == 0  # End follows the IF; it is not buried in one branch


def test_branch_terminal_output_works_whichever_side_is_unconnected():
    nodes = _parity_nodes()
    edges = [
        Edge("e1", "start", "input"), Edge("e2", "input", "check"),
        Edge("e3", "check", "odd", "False"), Edge("e4", "check", "even", "True"),
        Edge("e5", "odd", "end"),  # now the *True* output is the unfinished one
    ]
    context = _analyze_graph(nodes, edges)

    assert not context.has_errors, [d.code for d in context.diagnostics]
    rendered = [line.text for line in context.pseudocode]
    assert rendered.index('OUTPUT "number is even"') < rendered.index("ELSE") < rendered.index('OUTPUT "Number is odd"')
    assert rendered[-1] == "END"
    assert [d.element_ids for d in context.diagnostics if d.code == "BRANCH_ENDS_WITHOUT_END"] == [["even"]]


def test_branches_that_both_reach_end_are_unchanged_and_raise_no_branch_note():
    nodes = _parity_nodes()
    edges = _parity_edges() + [Edge("e6", "odd", "end")]
    context = _analyze_graph(nodes, edges)

    assert not context.has_errors
    assert not any(d.code == "BRANCH_ENDS_WITHOUT_END" for d in context.diagnostics)
    assert [line.text for line in context.pseudocode].count("END") == 1


def test_non_output_shape_without_outgoing_arrow_is_still_an_error():
    context = _analyze_graph(_parity_nodes("process", "odd_value = 1"), _parity_edges())

    codes = {d.code for d in context.diagnostics if d.severity == "error"}
    assert "INVALID_OUT_DEGREE" in codes
    assert "NO_PATH_TO_END" in codes
    assert context.ir is None


def test_output_outside_any_branch_still_needs_its_arrow():
    nodes = [
        _node("start", "terminal", "Start"),
        _node("show", "output", 'Output "hello"'),
        _node("end", "terminal", "End"),
    ]
    edges = [Edge("e1", "start", "show")]  # no arrow from the output, no decision upstream
    context = _analyze_graph(nodes, edges)

    codes = {d.code for d in context.diagnostics if d.severity == "error"}
    assert "INVALID_OUT_DEGREE" in codes
    assert not any(d.code == "BRANCH_ENDS_WITHOUT_END" for d in context.diagnostics)


def test_all_paths_unfinished_is_still_rejected_when_end_is_unreachable():
    nodes = _parity_nodes()
    edges = [e for e in _parity_edges() if e.id != "e5"]  # neither output reaches End
    context = _analyze_graph(nodes, edges)

    assert any(d.code == "UNREACHABLE_NODE" and "end" in d.element_ids for d in context.diagnostics)
    assert context.ir is None


def test_loop_exit_branch_may_finish_on_an_output():
    nodes = [
        _node("start", "terminal", "Start"),
        _node("loop", "iteration", "WHILE count < 3"),
        _node("body", "process", "count = count + 1"),
        _node("done", "output", 'Output "done"'),
        _node("end", "terminal", "End"),
    ]
    edges = [
        Edge("e1", "start", "loop"),
        Edge("e2", "loop", "body", "Yes"), Edge("e3", "body", "loop"),
        Edge("e4", "loop", "done", "No"),
    ]
    context = _analyze_graph(nodes, edges)
    # The end terminal is unreachable here, so this chart stays invalid; the point
    # is that a loop-exit output never crashes or hides the genuine problem.
    assert any(d.code == "UNREACHABLE_NODE" for d in context.diagnostics)


def test_with_several_end_terminals_an_unfinished_branch_is_not_joined_to_either():
    nodes = [
        _node("start", "terminal", "Start"),
        _node("first", "decision", "a == 1"),
        _node("second", "decision", "b == 1"),
        _node("a_out", "output", 'Output "a"'),
        _node("b_out", "output", 'Output "b"'),
        _node("end_a", "terminal", "End"),
        _node("end_b", "terminal", "End"),
    ]
    edges = [
        Edge("e1", "start", "first"),
        Edge("e2", "first", "a_out", "Yes"), Edge("e3", "first", "second", "No"),
        Edge("e4", "a_out", "end_a"),
        Edge("e5", "second", "b_out", "Yes"), Edge("e6", "second", "end_b", "No"),
        # b_out is unfinished; two Ends make any automatic join a guess.
    ]
    context = _analyze_graph(nodes, edges)

    assert not any(d.code == "INVALID_OUT_DEGREE" for d in context.diagnostics)
    assert not any(e.source == "b_out" for e in context.graph.edges)
    # Structuring either succeeds or reports a clear structure problem; it never
    # raises and never silently drops the unfinished output.
    if context.ir is not None:
        assert any('"b"' in line.text for line in context.pseudocode)
    else:
        assert any(d.code == "UNSTRUCTURED_FLOW" for d in context.diagnostics)


def test_logic_service_alone_does_not_mutate_the_graph_for_unfinished_branch():
    nodes, edges = _parity_nodes(), _parity_edges()
    context = PipelineContext(elements=nodes, graph=FlowGraph(
        nodes=nodes, edges=edges, start_id="start", end_ids=["end"]))
    before = [(e.id, e.source, e.target) for e in context.graph.edges]

    LogicService().run(context)

    assert not context.has_errors
    assert [(e.id, e.source, e.target) for e in context.graph.edges] == before
    assert [c.kind for c in context.ir.children] == ["Start", "Input", "If", "End"]
    branch = context.ir.children[2]
    assert [c.source_ids[0] for c in branch.children] == ["even"]
    assert [c.source_ids[0] for c in branch.otherwise] == ["odd"]


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract OCR is required for the fixture pipeline test")
def test_modulo_even_odd_fixture_completes_every_pipeline_stage():
    from backend.config import Settings
    from backend.pipeline import analyze, generate

    fixture = Path(__file__).parent / "fixtures" / "flowcharts" / "modulo_even_odd.png"
    context = analyze(fixture.read_bytes(), filename=fixture.name, settings=Settings())

    assert not context.has_errors, [(d.code, d.element_ids) for d in context.diagnostics if d.severity == "error"]
    assert context.completed_stages == ["intake", "vision", "graph", "validation", "logic", "pseudocode"]
    odd = next(n for n in context.graph.nodes if "odd" in n.text.casefold())
    end = next(n for n in context.graph.nodes if n.type == "terminal" and n.text.strip().casefold() == "end")
    # Only the five drawn connectors exist; nothing links odd -> End.
    assert len(context.graph.edges) == 5
    assert not any(e.source == odd.id and e.target == end.id for e in context.graph.edges)
    assert any(d.code == "BRANCH_ENDS_WITHOUT_END" and d.element_ids == [odd.id] for d in context.diagnostics)

    rendered = [line.text for line in context.pseudocode]
    assert rendered[0] == "START" and rendered[-1] == "END"
    assert rendered.count("ELSE") == 1 and rendered.count("END IF") == 1
    assert rendered.index('OUTPUT "number is even"') < rendered.index("ELSE") < rendered.index('OUTPUT "Number is odd"')

    # The result is usable downstream: approval, Python generation and tracing.
    context.approved = True
    generate(context)
    assert not context.has_errors
    assert context.completed_stages[-2:] == ["codegen", "trace"]
