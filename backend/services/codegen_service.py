"""Deterministic Python renderer with case-insensitive flowchart names and input inference.

Generated source is parsed and compiled, and variable references are normalized to the
spelling used by their declaration so OCR/capitalization differences do not create
runtime NameErrors (for example, Input Number followed by `number % 2 == 0`).
"""
from __future__ import annotations
import ast
import re
from backend.models import IRNode, PipelineContext, SourceLine
from backend.notation import DECLARE, FACTORIAL_COMMAND, input_names

_NUMERIC_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)

class _CanonicalNames(ast.NodeTransformer):
    def __init__(self, names: dict[str, str]):
        self.names = names
    def visit_Name(self, node):
        canonical = self.names.get(node.id.casefold())
        if canonical:
            node.id = canonical
        return node

class CodegenService:
    name = "codegen"

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.ir is None or context.has_errors or not context.approved:
            return context
        names, input_types = self._analyze(context.ir)
        if any(child.kind == "Process" and FACTORIAL_COMMAND.fullmatch(child.text.strip())
               for child in self._walk_ir(context.ir)) and not getattr(self, "_factorial_operand", None):
            context.diagnose("error", "AMBIGUOUS_FACTORIAL_INPUT",
                             "Calculate factorial needs an explicit input variable when the flowchart has multiple inputs.",
                             "Write 'Calculate factorial of name' or use a single input variable.")
            return context
        body=[]
        self._render(context.ir, body, 1, names, input_types)
        body=[SourceLine(1,0,"def main():",[])]+[SourceLine(index+2,indent,text,ids) for index,(indent,text,ids) in enumerate(body)]
        body += [SourceLine(len(body)+1,0,"",[]),SourceLine(len(body)+2,0,'if __name__ == "__main__":',[]),SourceLine(len(body)+3,1,"main()",[])]
        source="\n".join("    "*line.indent+line.text for line in body)+"\n"
        try:
            tree = ast.parse(source)
            compile(tree,"<modulo-generated>","exec")
            self._check_references(tree)
        except (SyntaxError, ValueError, NameError) as error:
            context.diagnose("error","SYNTAX_CHECK_FAILED",f"Generated Python failed its validation check: {error}","This is an implementation defect; preserve the flowchart and report the generated structure.")
            return context
        context.code=body
        return context

    def _analyze(self, root: IRNode):
        names: dict[str,str] = {}
        input_types: dict[str,str] = {}
        explicit_input_types: set[str] = set()
        expressions: list[str] = []
        factorial_input: str | None = None
        input_variables: list[str] = []
        def define(name):
            if name:
                names.setdefault(name.casefold(), name)
        def walk(node):
            nonlocal factorial_input
            for child in node.children:
                if child.kind == "Input":
                    variables = input_names(child.variable)
                    for variable in variables:
                        define(variable)
                        input_variables.append(variable)
                        input_types[variable.casefold()] = child.expression or "str"
                    if child.text and re.search(r":\s*(?:int|float|str)\s*$", child.text, re.IGNORECASE):
                        explicit_input_types.update(variable.casefold() for variable in variables)
                elif child.kind == "For":
                    define(child.variable); expressions.append(child.expression)
                if child.kind in ("If", "While"):
                    expressions.append(child.condition)
                elif child.kind == "Output":
                    expressions.append(child.expression)
                elif child.kind == "Process":
                    expressions.append(child.text)
                    match = FACTORIAL_COMMAND.fullmatch(child.text.strip())
                    if match and match.group(1):
                        factorial_input = match.group(1)
                walk(child)
                if child.otherwise:
                    walk(IRNode("Block", children=child.otherwise))
        walk(root)
        # Infer integer input from arithmetic/comparisons with numeric literals, but
        # never override an explicit :int/:float/:str annotation from the diagram.
        for expression in expressions:
            try:
                tree = ast.parse(expression, mode="exec" if "=" in expression and "==" not in expression else "eval")
            except (SyntaxError, ValueError):
                try: tree = ast.parse(expression, mode="eval")
                except (SyntaxError, ValueError): continue
            numeric_context = any(isinstance(n, (ast.BinOp, ast.Compare)) for n in ast.walk(tree))
            if not numeric_context:
                continue
            numeric_names = {item.id.casefold() for item in ast.walk(tree) if isinstance(item, ast.Name)}
            if "number" in input_types and "num" in numeric_names and "num" not in input_types:
                numeric_names.add("number")
            for key in numeric_names.intersection(input_types):
                if key not in explicit_input_types and input_types[key] == "str":
                    input_types[key] = "int"
        # A command like "Calculate factorial" operates on the sole input when no
        # operand is written. This is deterministic and avoids guessing with several inputs.
        if factorial_input:
            names.setdefault(factorial_input.casefold(), factorial_input)
        self._factorial_operand = factorial_input or (input_variables[0] if len(input_variables) == 1 else None)
        return names, input_types

    def _normalize(self, text: str, names: dict[str,str], mode="eval") -> str:
        try:
            tree = ast.parse(text, mode=mode)
            canonical_names = dict(names)
            # Common abbreviation in educational diagrams: "num" often refers to
            # the input labelled "number". Apply only when unambiguous.
            if "number" in names and "num" not in names:
                canonical_names["num"] = names["number"]
            tree = _CanonicalNames(canonical_names).visit(tree)
            ast.fix_missing_locations(tree)
            return ast.unparse(tree).strip()
        except (SyntaxError, ValueError):
            return text

    def _render(self,node,lines,indent,names,input_types):
        for child in node.children:
            ids=child.source_ids
            if child.kind=="Process":
                factorial_match = FACTORIAL_COMMAND.fullmatch(child.text.strip())
                if factorial_match:
                    operand = factorial_match.group(1) or getattr(self, "_factorial_operand", None)
                    if operand:
                        operand = names.get(operand.casefold(), operand)
                        lines.append((indent, "factorial = 1", ids))
                        lines.append((indent, f"for _factorial_index in range(1, {operand} + 1):", ids))
                        lines.append((indent + 1, "factorial *= _factorial_index", ids))
                    else:
                        # Validation should have caught this ambiguous operation; keep
                        # code generation defensive rather than silently choosing input.
                        raise ValueError("Calculate factorial needs an explicit operand or exactly one input.")
                elif DECLARE.fullmatch(child.text.strip()):
                    pass
                else:
                    lines.append((indent,self._normalize(child.text,names,"exec"),ids))
            elif child.kind=="Input":
                for raw_variable in input_names(child.variable):
                    variable=names.get(raw_variable.casefold(),raw_variable)
                    input_type=input_types.get(raw_variable.casefold(),child.expression or "str")
                    lines.append((indent,f"{variable} = {input_type}(input())",ids))
            elif child.kind=="Output":
                expression=self._normalize(child.expression,names)
                lines.append((indent,f"print({expression})",ids))
            elif child.kind=="If":
                condition=self._normalize(child.condition,names)
                lines.append((indent,f"if {condition}:",ids)); self._render_block(child.children,lines,indent+1,names,input_types)
                if child.otherwise: lines.append((indent,"else:",ids)); self._render_block(child.otherwise,lines,indent+1,names,input_types)
            elif child.kind=="While":
                condition=self._normalize(child.condition,names)
                lines.append((indent,f"while {condition}:",ids)); self._render_block(child.children,lines,indent+1,names,input_types)
            elif child.kind=="For":
                variable=names.get(child.variable.casefold(),child.variable)
                expression=self._normalize(child.expression,names)
                lines.append((indent,f"for {variable} in {expression}:",ids)); self._render_block(child.children,lines,indent+1,names,input_types)
            elif child.kind in ("Start","End"):
                continue

    def _walk_ir(self, root):
        for child in root.children:
            yield child
            yield from self._walk_ir(child)
            if child.otherwise:
                yield from self._walk_ir(IRNode("Block", children=child.otherwise))

    def _render_block(self,children,lines,indent,names,input_types):
        self._render(IRNode("Block",children=children),lines,indent,names,input_types)

    def _check_references(self, tree: ast.AST):
        """Catch obvious unresolved local names before presenting code to the user."""
        defined = {"print", "input", "int", "float", "str", "len", "range", "abs", "min", "max", "round", "__name__"}
        defined.update(n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store): defined.add(node.id)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in defined:
                raise NameError(f"Generated code refers to undefined variable '{node.id}'.")

def run(context): return CodegenService().run(context)
