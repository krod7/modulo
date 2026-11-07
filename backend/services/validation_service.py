"""Validate graph structure and the supported safe notation, never execute text."""
from collections import defaultdict
import networkx as nx
from backend.models import PipelineContext
from backend.notation import terminal_role, NotationError, normalize_label, validate_node_text
from backend.services.programming_ocr_validation import ProgrammingOCRValidationService

def _reconcile_decision_labels(context: PipelineContext) -> None:
    """Canonicalize OCR labels and recover only strongly supported missing pairs.

    This runs immediately before validation as a last, conservative normalization
    layer. It does not replace explicit Yes/No labels. If OCR captured one branch
    label but missed the other, the missing counterpart is inferred only when the
    two connector paths visibly depart from opposite sides of the same decision
    and both lead downward to recognized nodes.
    """
    graph = context.graph
    if graph is None:
        return
    nodes = {node.id: node for node in graph.nodes}
    for decision in graph.nodes:
        if decision.type not in ("decision", "iteration"):
            continue
        branches = [edge for edge in graph.edges if edge.source == decision.id]
        if len(branches) != 2:
            continue

        # Normalize punctuation/case from OCR, e.g. "Yes." and "No,".
        for edge in branches:
            canonical = normalize_label(edge.label)
            if canonical:
                edge.label = canonical

        labels = [normalize_label(edge.label) for edge in branches]
        if sorted(label or "" for label in labels) == ["No", "Yes"]:
            continue

        # Only infer an omitted label, never overwrite two explicit conflicting
        # labels or silently reinterpret arbitrary branch text.
        missing = [i for i, edge in enumerate(branches) if not (edge.label or "").strip()]
        known = [i for i, label in enumerate(labels) if label in ("Yes", "No")]
        if len(missing) != 1 or len(known) != 1:
            if any((edge.label or "").strip() for edge in branches):
                continue

        center_x, center_y = decision.bbox.center
        departures = {}
        for i, edge in enumerate(branches):
            target = nodes.get(edge.target)
            if target is None or not edge.points:
                continue
            sx, sy = edge.points[0]
            dx = sx - center_x
            if target.bbox.center[1] > center_y and abs(dx) >= max(12, decision.bbox.width * 0.22):
                departures["left" if dx < 0 else "right"] = i

        if set(departures) != {"left", "right"}:
            continue

        if len(missing) == 1 and len(known) == 1:
            missing_index = missing[0]
            known_label = labels[known[0]]
            branches[missing_index].label = "No" if known_label == "Yes" else "Yes"
        else:
            branches[departures["left"]].label = "Yes"
            branches[departures["right"]].label = "No"
        context.diagnose(
            "warning", "INFERRED_BRANCH_LABELS",
            "A missing decision label was inferred from opposite-side connector departures; confirm it during review.",
            "Verify the Yes/No labels during review.",
            [decision.id] + [edge.id for edge in branches],
        )


def branch_terminal_outputs(graph) -> set[str]:
    """Return output shapes that legitimately finish a decision branch.

    A flowchart branch may end on its last action (for example, ``Print "odd"``)
    without a further arrow to End.  Such a shape qualifies only when it is an
    output, has no outgoing connector, is itself reachable, and sits downstream
    of a decision/iteration, so a real branch ends there.  Every other
    non-terminal shape keeps the strict one-outgoing-arrow rule, and a missing
    connector is never fabricated: the graph is left untouched, and later stages
    treat the shape as the end of that branch.
    """
    if graph is None:
        return set()
    nodes = {node.id: node for node in graph.nodes}
    directed = nx.DiGraph()
    directed.add_nodes_from(nodes)
    for edge in graph.edges:
        if edge.source in nodes and edge.target in nodes:
            directed.add_edge(edge.source, edge.target)
    result = set()
    for node in graph.nodes:
        if node.type != "output" or directed.out_degree(node.id) or not directed.in_degree(node.id):
            continue
        if any(nodes[ancestor].type in ("decision", "iteration") for ancestor in nx.ancestors(directed, node.id)):
            result.add(node.id)
    return result


class ValidationService:
    name = "validation"

    def run(self, context: PipelineContext) -> PipelineContext:
        ProgrammingOCRValidationService().run(context)
        graph = context.graph
        if graph is None:
            context.diagnose("error", "MISSING_GRAPH", "No graph is available to validate.", "Run graph construction before validation.")
            return context
        _reconcile_decision_labels(context)
        nodes = {node.id: node for node in graph.nodes}
        directed = nx.DiGraph()
        directed.add_nodes_from(nodes)
        incoming, outgoing = defaultdict(list), defaultdict(list)
        for edge in graph.edges:
            if edge.source not in nodes or edge.target not in nodes:
                context.diagnose("error", "DANGLING_ARROW", "An arrow refers to a missing shape.", "Connect every arrow to recognized shapes.", [edge.id])
                continue
            incoming[edge.target].append(edge)
            outgoing[edge.source].append(edge)
            directed.add_edge(edge.source, edge.target)
        branch_terminals = branch_terminal_outputs(graph)
        starts = [node.id for node in graph.nodes if node.type == "terminal" and terminal_role(node.text) == "start"]
        ends = [node.id for node in graph.nodes if node.type == "terminal" and terminal_role(node.text) == "end"]
        graph.start_id = starts[0] if len(starts) == 1 else None
        graph.end_ids = ends
        if not starts:
            context.diagnose("error", "NO_START_NODE", "No Start terminal was recognized.", "Add one rounded rectangle containing only Start.")
        elif len(starts) > 1:
            context.diagnose("error", "MULTIPLE_START_NODES", "More than one Start terminal was recognized.", "Keep exactly one Start terminal on the page.", starts)
        if not ends:
            context.diagnose("error", "NO_END_NODE", "No End terminal was recognized.", "Add at least one rounded rectangle containing only End.")
        pairs = defaultdict(list)
        for node in graph.nodes:
            if node.type == "connector":
                pairs[node.text.strip().casefold()].append(node.id)
            if node.confidence * 100 < context.settings.OCR_MIN_CONFIDENCE or any(word.confidence * 100 < context.settings.OCR_MIN_CONFIDENCE for word in node.words):
                node.flags.append("LOW_OCR_CONFIDENCE")
                context.diagnose("warning", "LOW_OCR_CONFIDENCE", f"Text in {node.id} may have been misread.", "Compare it with your source; use larger text or a higher-resolution export if needed.", [node.id])
            try:
                validate_node_text(node.type, node.text)
            except NotationError as error:
                code = "UNREADABLE_TEXT" if not node.text.strip() else "UNSUPPORTED_SHAPE" if node.type not in ("terminal", "process", "input", "output", "decision", "iteration", "connector") else "UNSUPPORTED_STATEMENT"
                context.diagnose("error", code, f"{node.id}: {error}", "Use the notation guide and correct the source diagram, then re-upload.", [node.id])
            branches = outgoing[node.id]
            is_end = node.id in ends
            if is_end:
                if branches:
                    context.diagnose("error", "END_HAS_OUTGOING_ARROW", "An End terminal has an outgoing arrow.", "End terminates a path and must not send flow onward.", [node.id])
            elif node.type in ("decision", "iteration"):
                if len(branches) != 2:
                    context.diagnose("error", "DECISION_MISSING_BRANCH", f"{node.id} needs exactly two outgoing branches.", "Draw two outgoing arrows labeled Yes and No.", [node.id] + [edge.id for edge in branches])
                elif sorted(normalize_label(edge.label) or "" for edge in branches) != ["No", "Yes"]:
                    context.diagnose("error", "INVALID_BRANCH_LABELS", f"{node.id} does not have one Yes and one No branch.", "Label branches Yes/No, True/False, or Y/N. Move labels near their arrows.", [node.id] + [edge.id for edge in branches])
            elif node.id in branch_terminals:
                context.diagnose(
                    "warning", "BRANCH_ENDS_WITHOUT_END",
                    f"{node.id} finishes its branch but no arrow leaves it, so the branch was treated as ending the flowchart there.",
                    "If this path should continue to End, draw the missing arrow and re-upload; otherwise no change is needed.",
                    [node.id],
                )
            elif len(branches) != 1:
                context.diagnose("error", "INVALID_OUT_DEGREE", f"{node.id} needs one outgoing arrow.", "Only a decision or iteration may split flow; all other non-End shapes have one next step.", [node.id])
            if node.id in starts and incoming[node.id]:
                context.diagnose("error", "START_HAS_INCOMING_ARROW", "Start has an incoming arrow.", "Start begins the program; loop back to an iteration, not Start.", [node.id])
        for label in sorted(pairs):
            if len(pairs[label]) != 2:
                context.diagnose("error", "UNPAIRED_CONNECTOR", "A connector label does not occur exactly twice.", "Use two circles with the same label: one entry and one exit.", pairs[label])
        if graph.start_id:
            reachable = nx.descendants(directed, graph.start_id) | {graph.start_id}
            unreachable = [node.id for node in graph.nodes if node.id not in reachable]
            if unreachable:
                context.diagnose("error", "UNREACHABLE_NODE", "Some shapes cannot be reached from Start.", "Connect these shapes to the main flow or remove them.", unreachable)
            can_end = set(ends)
            # A branch that legitimately finishes on an output completes the flow
            # just as an End terminal does, along with everything leading to it.
            for end in [*ends, *branch_terminals]:
                can_end.add(end)
                can_end.update(nx.ancestors(directed, end))
            dead = [node.id for node in graph.nodes if node.id in reachable and node.id not in can_end]
            if dead:
                context.diagnose("error", "NO_PATH_TO_END", "Some reachable paths cannot reach an End terminal.", "Provide a loop exit or connect the unfinished path to End.", dead)
            # Validate reducible cycles without implementing the later IR recovery stage.
            loops = []
            for component in nx.strongly_connected_components(directed.subgraph(reachable)):
                cyclic = len(component) > 1 or any(directed.has_edge(node, node) for node in component)
                if not cyclic:
                    continue
                ids = [node.id for node in graph.nodes if node.id in component]
                entries = {target for source, target in directed.edges if target in component and source not in component}
                headers = [node_id for node_id in ids if nodes[node_id].type == "iteration"]
                if len(entries) != 1 or not headers or next(iter(entries)) not in headers:
                    context.diagnose("error", "UNSTRUCTURED_FLOW", "A cycle has multiple entries or no iteration header at its entry.", "Express loops with a WHILE/FOR hexagon and a single entry into the loop.", ids)
                else:
                    loops.append(component)
            for node in graph.nodes:
                if node.type == "process" and node.text.strip() in ("break", "continue") and not any(node.id in component for component in loops):
                    # Break may not lie in the cyclic SCC, so test dominance by an iteration too.
                    dominators = nx.immediate_dominators(directed, graph.start_id)
                    ancestor, inside = node.id, False
                    seen = set()
                    while ancestor in dominators and ancestor not in seen:
                        seen.add(ancestor)
                        if nodes[ancestor].type == "iteration":
                            yes = [edge.target for edge in outgoing[ancestor] if normalize_label(edge.label) == "Yes"]
                            exits = {edge.target for edge in outgoing[ancestor] if normalize_label(edge.label) == "No"}
                            body_graph = directed.subgraph(set(directed.nodes) - exits - {ancestor})
                            inside = any(body in body_graph and node.id in body_graph and
                                         (node.id == body or nx.has_path(body_graph, body, node.id)) for body in yes)
                            if inside:
                                break
                        ancestor = dominators[ancestor]
                    if not inside:
                        context.diagnose("error", "LOOP_CONTROL_OUTSIDE_LOOP", f"{node.text.strip()} is not inside a recognized loop body.", "Use break/continue only on the Yes/body path of an iteration.", [node.id])
        return context

def run(context: PipelineContext) -> PipelineContext:
    return ValidationService().run(context)
