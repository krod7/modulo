"""OCR-free regressions for loop-header detection, loop-body exits, loop-back
repair, and unquoted one-word outputs (validation -> logic -> codegen)."""
from backend.config import Settings
from backend.models import BoundingBox, Edge, FlowGraph, Node, PipelineContext
from backend.services.codegen_service import CodegenService
from backend.services.graph_service import repair_semantic_loop_back_edges
from backend.services.logic_service import LogicService
from backend.services.pseudocode_service import PseudocodeService
from backend.services.validation_service import ValidationService


def _node(node_id, kind, text, y=0):
    return Node(node_id, kind, text, BoundingBox(100, y, 120, 40), confidence=0.95)


def _convert(nodes, edges):
    graph = FlowGraph(nodes=nodes, edges=[Edge(f"e{i}", *edge) for i, edge in enumerate(edges, 1)])
    context = PipelineContext(elements=nodes, graph=graph, settings=Settings())
    for service in (ValidationService(), LogicService(), PseudocodeService()):
        if context.has_errors:
            break
        context = service.run(context)
    if not context.has_errors:
        context.approved = True
        context = CodegenService().run(context)
    return context


def _types(context):
    return {node.id: node.type for node in context.graph.nodes}


def _code(context):
    return "\n".join("    " * line.indent + line.text for line in context.code)


def _errors(context):
    return [item.code for item in context.diagnostics if item.severity == "error"]


def test_decision_inside_loop_body_stays_a_decision():
    nodes = [_node("s", "terminal", "Start"), _node("i0", "process", "i = 1"),
             _node("h", "decision", "i <= 10?"), _node("d", "decision", "i % 2 == 0?"),
             _node("add", "process", "total = total + i"), _node("inc", "process", "i = i + 1"),
             _node("t0", "process", "total = 0"), _node("o", "output", "Output total"),
             _node("e", "terminal", "End")]
    context = _convert(nodes, [("s", "t0"), ("t0", "i0"), ("i0", "h"), ("h", "d", "Yes"), ("h", "o", "No"),
                               ("d", "add", "Yes"), ("d", "inc", "No"), ("add", "inc"), ("inc", "h"),
                               ("o", "e")])
    assert not _errors(context), context.diagnostics
    assert _types(context)["h"] == "iteration" and _types(context)["d"] == "decision"
    code = _code(context)
    assert "while i <= 10:" in code and "if i % 2 == 0:" in code
    assert "while i % 2" not in code


def test_outer_loop_with_nested_loop_is_still_a_loop():
    nodes = [_node("s", "terminal", "Start"), _node("i0", "process", "i = 1"),
             _node("outer", "decision", "i <= 3?"), _node("j0", "process", "j = 1"),
             _node("inner", "decision", "j <= 3?"), _node("p", "output", "Output i * j"),
             _node("jinc", "process", "j = j + 1"), _node("iinc", "process", "i = i + 1"),
             _node("e", "terminal", "End")]
    context = _convert(nodes, [("s", "i0"), ("i0", "outer"), ("outer", "j0", "Yes"), ("outer", "e", "No"),
                               ("j0", "inner"), ("inner", "p", "Yes"), ("inner", "iinc", "No"),
                               ("p", "jinc"), ("jinc", "inner"), ("iinc", "outer")])
    assert not _errors(context), context.diagnostics
    assert _types(context)["outer"] == "iteration" and _types(context)["inner"] == "iteration"


def test_post_test_loop_with_nested_decision_is_still_a_loop():
    nodes = [_node("s", "terminal", "Start"), _node("n0", "process", "n = 0"),
             _node("body", "process", "n = n + 1"), _node("odd", "decision", "n % 2 == 1?"),
             _node("show", "output", "Output n"), _node("again", "decision", "n < 5?"),
             _node("e", "terminal", "End")]
    context = _convert(nodes, [("s", "n0"), ("n0", "body"), ("body", "odd"), ("odd", "show", "Yes"),
                               ("odd", "again", "No"), ("show", "again"), ("again", "body", "Yes"),
                               ("again", "e", "No")])
    assert _types(context)["again"] == "iteration"
    assert _types(context)["odd"] == "decision"
    # The structurer cannot yet lower this shape; it must report it, not crash.
    assert context.code or "UNSTRUCTURED_FLOW" in _errors(context)


def test_loop_body_exiting_to_end_is_reported_not_silently_dropped():
    nodes = [_node("s", "terminal", "Start"), _node("a0", "process", "attempts = 0"),
             _node("h", "decision", "attempts < 3?"), _node("pw", "input", "Input password"),
             _node("ok", "decision", "password == 1234?"), _node("yes", "output", 'Output "Granted"'),
             _node("inc", "process", "attempts = attempts + 1"), _node("lock", "output", 'Output "Locked"'),
             _node("e", "terminal", "End")]
    context = _convert(nodes, [("s", "a0"), ("a0", "h"), ("h", "pw", "Yes"), ("h", "lock", "No"),
                               ("pw", "ok"), ("ok", "yes", "Yes"), ("ok", "inc", "No"), ("inc", "h"),
                               ("yes", "e"), ("lock", "e")])
    assert "UNSTRUCTURED_FLOW" in _errors(context)
    assert not context.code


def test_undefined_one_word_output_is_printed_as_text():
    nodes = [_node("s", "terminal", "Start"), _node("in", "input", "Input age"),
             _node("d", "decision", "age >= 18?"), _node("o", "output", "Output Adult"),
             _node("e", "terminal", "End")]
    context = _convert(nodes, [("s", "in"), ("in", "d"), ("d", "o", "Yes"), ("d", "e", "No"), ("o", "e")])
    assert not _errors(context), context.diagnostics
    assert "print('Adult')" in _code(context)
    assert any(item.code == "OUTPUT_TEXT_ASSUMED" and item.element_ids == ["o"] for item in context.diagnostics)


def test_defined_one_word_output_is_still_a_variable():
    nodes = [_node("s", "terminal", "Start"), _node("in", "input", "Input Total"),
             _node("o", "output", "Output total"), _node("e", "terminal", "End")]
    context = _convert(nodes, [("s", "in"), ("in", "o"), ("o", "e")])
    assert not _errors(context), context.diagnostics
    assert "print(Total)" in _code(context)
    assert not any(item.code == "OUTPUT_TEXT_ASSUMED" for item in context.diagnostics)


def test_loop_back_repair_keeps_a_step_that_already_flows_back_to_the_header():
    nodes = [_node("s", "terminal", "Start", 0), _node("h", "decision", "i <= n?", 100),
             _node("d", "decision", "score >= 60?", 200), _node("p", "process", "passed = passed + 1", 300),
             _node("inc", "process", "i = i + 1", 400), _node("e", "terminal", "End", 500)]
    edges = [Edge("e1", "s", "h"), Edge("e2", "h", "d", "Yes"), Edge("e3", "h", "e", "No"),
             Edge("e4", "d", "p", "Yes"), Edge("e5", "d", "inc", "No"), Edge("e6", "p", "inc"),
             Edge("e7", "inc", "h")]
    graph = FlowGraph(nodes=nodes, edges=edges)
    context = PipelineContext(elements=nodes, graph=graph, settings=Settings())
    repair_semantic_loop_back_edges(context, graph)
    assert ("p", "inc") in {(edge.source, edge.target) for edge in graph.edges}
    assert ("p", "h") not in {(edge.source, edge.target) for edge in graph.edges}


def test_misread_name_is_flagged_as_undefined_variable():
    nodes = [_node("s", "terminal", "Start"), _node("i0", "process", "i = 1"),
             _node("h", "decision", "i <= 5?"), _node("inc", "process", "i=itl"),
             _node("e", "terminal", "End")]
    context = _convert(nodes, [("s", "i0"), ("i0", "h"), ("h", "inc", "Yes"), ("h", "e", "No"), ("inc", "h")])
    flagged = [item for item in context.diagnostics if item.code == "UNDEFINED_VARIABLE"]
    assert [item.element_ids for item in flagged] == [["inc"]] and "'itl'" in flagged[0].message
