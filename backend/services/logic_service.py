"""Recover the supported structured control flow into one deterministic IR."""
from __future__ import annotations
import ast
import networkx as nx
from backend.models import Edge, FlowGraph, IRNode, Node, PipelineContext
from backend.notation import FOR, INPUT, normalize_condition, normalize_label
from backend.services.validation_service import branch_terminal_outputs

class LogicService:
    name = "logic"

    def run(self, context: PipelineContext) -> PipelineContext:
        graph = context.graph
        if graph is None or graph.start_id is None or context.has_errors:
            return context
        nodes = {node.id: node for node in graph.nodes}
        outgoing = {node_id: [] for node_id in nodes}
        for edge in graph.edges:
            if not edge.structural:
                outgoing.setdefault(edge.source, []).append(edge)
        # An output that finishes a decision branch has no drawn arrow.  Join it to
        # the single End *for structuring only*: this lets both branches reconverge
        # at End (IF/ELSE ... END) without adding a connector to graph.edges.  With
        # several End terminals there is no unambiguous join, so it stays a plain
        # branch exit.
        if len(graph.end_ids) == 1:
            for node_id in sorted(branch_terminal_outputs(graph)):
                if not outgoing.get(node_id):
                    outgoing[node_id] = [Edge(f"implicit_{node_id}_end", node_id, graph.end_ids[0])]
        for edge_list in outgoing.values():
            edge_list.sort(key=lambda edge: (normalize_label(edge.label) or "", edge.id))
        reachable = nx.DiGraph()
        reachable.add_nodes_from(nodes)
        reachable.add_edges_from((edge.source, edge.target) for edges in outgoing.values() for edge in edges)
        try:
            dominators = nx.immediate_dominators(reachable, graph.start_id)
        except (nx.NetworkXError, nx.NetworkXNoPath) as error:
            context.diagnose("error", "UNSTRUCTURED_FLOW", "The graph cannot be structured from Start.", "Connect every path using sequences, decisions, and loops.", [graph.start_id])
            return context
        post = self._postdominators(reachable, set(graph.end_ids))
        emitted: set[str] = set()
        try:
            root_children = self._sequence(graph.start_id, None, nodes, outgoing, reachable, dominators, post, emitted, context)
            context.ir = IRNode("Program", source_ids=[], children=root_children)
            if emitted - set(nodes):
                raise ValueError("IR referenced an unknown node")
            missing = [node.id for node in graph.nodes if node.type not in ("terminal", "connector") and node.id not in emitted]
            if missing:
                context.diagnose("error", "UNSTRUCTURED_FLOW", "Some flowchart elements could not be placed in structured control flow.", "Use one-entry if/else and loop regions with visible merge points.", missing)
                context.ir = None
        except StructureError as error:
            context.diagnose("error", "UNSTRUCTURED_FLOW", str(error), "Use sequences, two-way decisions, and single-entry loops; avoid jumps and crossing regions.", error.ids)
            context.ir = None
        return context

    def _postdominators(self, graph: nx.DiGraph, ends: set[str]) -> dict[str, set[str]]:
        """Return the nodes that every path from each node must visit.

        A synthetic exit joins all declared End terminals.  The fixed-point
        calculation runs over the original graph's successors (not its reverse),
        which is essential for finding real branch reconvergence points.
        """
        sink = "__synthetic_end__"
        working = graph.copy()
        working.add_node(sink)
        for end in ends:
            if end in working:
                working.add_edge(end, sink)

        universe = set(working.nodes)
        result = {node: ({sink} if node == sink else set(universe)) for node in working}
        changed = True
        while changed:
            changed = False
            # Reverse lexical order is deterministic; convergence does not rely
            # on this ordering, including for cyclic flow graphs.
            for node in sorted(universe - {sink}, reverse=True):
                successors = list(working.successors(node))
                if successors:
                    common = set.intersection(*(result[target] for target in successors))
                    updated = {node} | common
                else:
                    updated = {node}
                if updated != result[node]:
                    result[node] = updated
                    changed = True
        return result

    def _merge(self, node_id, edges, post, stop, ids):
        """Find the nearest common postdominator of the branch destinations.

        Reachability alone is insufficient: two branches may both eventually
        reach End while first reconverging at an earlier node.  Choosing that
        earliest mandatory merge keeps nested and sequential decisions scoped
        to their own regions.  ``stop`` remains the outer-region fallback.
        """
        if len(edges) < 2:
            return stop

        def distances(start):
            found = {start: 0}
            queue = [start]
            while queue:
                current = queue.pop(0)
                for target in self._forward.get(current, ()):
                    if target not in found:
                        found[target] = found[current] + 1
                        queue.append(target)
            return found

        reach = [distances(edge.target) for edge in edges]
        common_postdominators = set.intersection(
            *(post.get(edge.target, set()) for edge in edges)
        )
        candidates = common_postdominators.intersection(*(set(item) for item in reach))
        candidates.discard(node_id)
        if not candidates:
            return stop
        # Prefer the merge reached soonest by the slowest branch. Sum and node
        # id provide stable tie-breakers for diagrams with equal-length routes.
        return min(
            candidates,
            key=lambda item: (
                max(mapping[item] for mapping in reach),
                sum(mapping[item] for mapping in reach),
                item,
            ),
        )

    def _sequence(self, current, stop, nodes, outgoing, graph, dominators, post, emitted, context):
        result = []
        cursor = current
        local_seen = set()
        while cursor is not None and cursor != stop:
            if cursor in local_seen:
                raise StructureError("A sequence contains a cycle without an iteration header.", list(local_seen))
            local_seen.add(cursor)
            node = nodes[cursor]
            if node.type in ("terminal", "connector"):
                emitted.add(cursor)
                if node.type == "terminal" and node.text.strip().casefold() == "start":
                    result.append(IRNode("Start", [cursor]))
                elif node.type == "terminal" and node.text.strip().casefold() == "end":
                    result.append(IRNode("End", [cursor]))
                cursor = outgoing[cursor][0].target if outgoing.get(cursor) else None
                continue
            if node.type in ("decision", "iteration"):
                edges = outgoing.get(cursor, [])
                yes = next((edge for edge in edges if normalize_label(edge.label) == "Yes"), None)
                no = next((edge for edge in edges if normalize_label(edge.label) == "No"), None)
                if not yes or not no:
                    raise StructureError(f"{cursor} has no Yes/No pair.", [cursor])
                self._forward={node_id: [edge.target for edge in edge_list] for node_id,edge_list in outgoing.items()}
                merge = self._merge(cursor, [yes, no], post, stop, [cursor])
                if node.type == "iteration":
                    loop_body = self._loop_body(cursor, yes.target, nodes, outgoing, graph, post, emitted, context)
                    condition = node.text.split(None, 1)[1].strip() if node.text.upper().startswith("WHILE ") else ""
                    variable, expression = "", ""
                    if node.text.upper().startswith("FOR "):
                        match = FOR.fullmatch(node.text.strip())
                        if match: variable, expression = match.group(1), match.group(2)
                    result.append(IRNode("While" if condition else "For", [cursor], condition=condition,
                                         variable=variable, expression=expression, children=loop_body))
                    emitted.add(cursor)
                    cursor = no.target
                    continue
                then = self._sequence(yes.target, merge, nodes, outgoing, graph, dominators, post, emitted, context)
                otherwise = self._sequence(no.target, merge, nodes, outgoing, graph, dominators, post, emitted, context)
                result.append(IRNode("If", [cursor], condition=normalize_condition(node.text), children=then, otherwise=otherwise))
                emitted.add(cursor); cursor = merge
                continue
            result.append(self._leaf(node)); emitted.add(cursor)
            edges = outgoing.get(cursor, [])
            cursor = edges[0].target if edges else None
        return result

    def _loop_body(self, header, start, nodes, outgoing, graph, post, emitted, context):
        result=[]; cursor=start; seen=set()
        while cursor != header:
            if cursor in seen or cursor not in nodes:
                raise StructureError("A loop body has multiple entries or cannot return to its header.", [header, cursor])
            seen.add(cursor); node=nodes[cursor]
            if node.type in ("decision", "iteration"):
                raise StructureError("Nested control-flow regions are not reducible in this graph.", [header, cursor])
            result.append(self._leaf(node)); emitted.add(cursor)
            edges=outgoing.get(cursor, [])
            back=next((edge for edge in edges if edge.target == header), None)
            if back: cursor=header; continue
            if len(edges)!=1: raise StructureError("A loop body has an ambiguous exit.", [header,cursor])
            cursor=edges[0].target
        return result

    def _leaf(self, node: Node) -> IRNode:
        text=node.text.strip()
        if node.type == "input":
            match=INPUT.fullmatch(text)
            return IRNode("Input", [node.id], text=text, variable=match.group(1) if match else "", expression=match.group(2) if match and match.group(2) else "str")
        if node.type == "output": return IRNode("Output", [node.id], text=text, expression=text.split(None,1)[1].strip())
        if node.type == "process": return IRNode("Process", [node.id], text=text)
        raise StructureError(f"Unsupported node {node.id} in structured flow.", [node.id])

class StructureError(ValueError):
    def __init__(self, message, ids): super().__init__(message); self.ids=ids

def run(context: PipelineContext) -> PipelineContext: return LogicService().run(context)
