"""The approved review text is rendered from IR and never reparsed."""
import re
from backend.models import IRNode, PipelineContext, SourceLine

class PseudocodeService:
    name = "pseudocode"
    def run(self, context: PipelineContext) -> PipelineContext:
        if context.ir is None or context.has_errors: return context
        lines=[]
        self._render(context.ir, lines, 0)
        context.pseudocode=[SourceLine(index+1, indent, text, ids) for index,(indent,text,ids) in enumerate(lines)]
        return context
    def _render(self,node,lines,indent):
        for child in node.children:
            ids=child.source_ids
            if child.kind=="Start": lines.append((indent,"START",ids))
            elif child.kind=="End": lines.append((indent,"END",ids))
            elif child.kind=="Process":
                declaration = re.fullmatch(r"declare\s+variables?\s+(.+)", child.text.strip(), re.IGNORECASE)
                if declaration:
                    variables = re.sub(r"\s+and\s+", ", ", declaration.group(1).strip(), flags=re.IGNORECASE)
                    variables = re.sub(r"\s*,\s*", ", ", variables)
                    lines.append((indent, f"DECLARE {variables}", ids))
                else:
                    lines.append((indent,f"SET {child.text}",ids))
            elif child.kind=="Input": lines.append((indent,f"INPUT {child.variable}",ids))
            elif child.kind=="Output": lines.append((indent,f"OUTPUT {child.expression}",ids))
            elif child.kind=="If":
                lines.append((indent,f"IF {child.condition}",ids)); self._render(IRNode("Block",children=child.children),lines,indent+1)
                if child.otherwise: lines.append((indent,"ELSE",ids)); self._render(IRNode("Block",children=child.otherwise),lines,indent+1)
                lines.append((indent,"END IF",ids))
            elif child.kind in ("While","For"):
                header=f"WHILE {child.condition}" if child.kind=="While" else f"FOR {child.variable} IN {child.expression}"
                lines.append((indent,header,ids)); self._render(IRNode("Block",children=child.children),lines,indent+1); lines.append((indent,"END WHILE" if child.kind=="While" else "END FOR",ids))
            else: self._render(child,lines,indent)

    def _walk(self, node):
        for child in node.children:
            yield child
            yield from self._walk(child)
            if child.otherwise:
                yield from self._walk(IRNode("Block", children=child.otherwise))

def run(context): return PseudocodeService().run(context)
