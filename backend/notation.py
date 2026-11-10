"""Single source of truth for the deterministic flowchart language."""
import ast
import keyword
import re
SHAPE_TYPES = ("terminal", "process", "input", "output", "decision", "iteration", "connector")
BRANCH_LABELS = {"yes": "Yes", "true": "Yes", "y": "Yes", "no": "No", "false": "No", "n": "No"}
SAFE_BUILTINS = ("print", "input", "int", "float", "str", "len", "range", "abs", "min", "max", "round")
INPUT_TYPES = ("int", "float", "str")
ALLOWED_AST_TYPES = (ast.Expression, ast.Module, ast.Assign, ast.AugAssign, ast.Expr, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call, ast.List, ast.Tuple,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.UAdd, ast.USub, ast.Not,
    ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
INPUT = re.compile(r"^Input\s+(?:variable\s*:\s*)?((?:[A-Za-z][A-Za-z0-9_]*\s*(?:,\s*(?:and\s+)?|\s+and\s+))*(?:[A-Za-z][A-Za-z0-9_]*))(?::(int|float|str))?$", re.IGNORECASE)
FOR = re.compile(r"^FOR\s+([A-Za-z][A-Za-z0-9_]*)\s+IN\s+(.+)$", re.IGNORECASE)
# Small, explicit vocabulary of safe high-level operations that can be lowered to Python.
FACTORIAL_COMMAND = re.compile(r"^calculate\s+(?:the\s+)?factorial(?:\s+of\s+([A-Za-z][A-Za-z0-9_]*))?$", re.IGNORECASE)
DECLARE = re.compile(r"^declare\s+variables?\s+(.+)$", re.IGNORECASE)

def is_supported_natural_operation(text: str) -> bool:
    """Whether a process label names a deterministic operation we implement."""
    return bool(FACTORIAL_COMMAND.fullmatch(text.strip()) or DECLARE.fullmatch(text.strip()))

class NotationError(ValueError):
    pass

def normalize_label(text: str | None) -> str | None:
    """Normalize common branch-label OCR variants without guessing arbitrary text.

    OCR frequently includes nearby punctuation (for example ``Yes.`` or ``No,``)
    when a label is printed close to a connector. Strip only surrounding
    punctuation/whitespace; keep the accepted vocabulary deliberately small.
    """
    value = (text or "").strip().casefold()
    value = re.sub(r"^[\W_]+|[\W_]+$", "", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value)
    return BRANCH_LABELS.get(value)

START_WORDS = ("start", "begin")
END_WORDS = ("end", "stop", "exit", "finish", "halt")


def terminal_role(text: str | None) -> str | None:
    """Return "start"/"end" for a terminal label, accepting common synonyms.

    Flowchart tools and textbooks use Begin/Start and End/Stop/Exit/Finish
    interchangeably; surrounding punctuation from OCR is ignored. Anything else
    (e.g. "Return result") is not treated as a terminal role.
    """
    value = re.sub(r"^[\W_]+|[\W_]+$", "", (text or "").strip().casefold())
    if value in START_WORDS:
        return "start"
    if value in END_WORDS:
        return "end"
    return None


def _parses(expression: str) -> bool:
    try:
        ast.parse(expression, mode="eval")
        return True
    except (SyntaxError, ValueError, RecursionError):
        return False


def normalize_condition(text: str) -> str:
    """Accept natural flowchart decision labels such as `if x % 2 == 0`.

    A diamond represents a condition, not a Python compound-statement block, so
    strip an optional leading `if` and trailing colon before parsing/rendering.
    Question-form wrappers (`is a > b?`, `Does S = 0?`) are removed only when
    what remains is a valid expression, and because an assignment cannot occur
    in a condition, a lone `=` is read as the equality pseudocode intends - but
    again only when that turns an unparseable condition into a valid one. The
    recognised node text itself is never rewritten.
    """
    condition = text.strip()
    if re.match(r"^if\s+", condition, re.IGNORECASE):
        condition = re.sub(r"^if\s+", "", condition, count=1, flags=re.IGNORECASE).strip()
    if condition.endswith(":"):
        condition = condition[:-1].rstrip()
    if _parses(condition):
        return condition
    candidate = condition.rstrip("?").rstrip()
    candidate = re.sub(r"^(?:is|does|do|are)\s+(?=\S)", "", candidate, count=1, flags=re.IGNORECASE)
    if not _parses(candidate):
        equality = re.sub(r"(?<![=!<>])=(?!=)", " == ", candidate)
        candidate = " ".join(equality.split()) if _parses(equality) else candidate
    return candidate if _parses(candidate) else condition

def safe_name(name: str) -> bool:
    # Flowchart authors commonly use `max` as an accumulator name. It is safe
    # as a local assignment target even though it shadows a builtin in Python.
    return bool(NAME.fullmatch(name)) and "__" not in name and not keyword.iskeyword(name)

def input_names(value: str) -> list[str]:
    """Split a readable input list without accepting arbitrary expressions."""
    return [name for name in re.split(r"\s*(?:,|\band\b)\s*", value, flags=re.IGNORECASE) if name]

def validate_ast(text: str, mode: str = "eval") -> ast.AST:
    try:
        tree = ast.parse(text, mode=mode)
    except (SyntaxError, ValueError, RecursionError) as error:
        raise NotationError("The text is not a valid supported Python expression or statement.") from error
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_AST_TYPES:
            raise NotationError(f"{type(node).__name__} is unsupported; imports, attributes, indexing, and arbitrary calls are forbidden.")
        if isinstance(node, ast.Name):
            if not NAME.fullmatch(node.id) or "__" in node.id:
                raise NotationError("Use ordinary variable names without double underscores.")
            if isinstance(node.ctx, ast.Store) and not safe_name(node.id):
                raise NotationError("Assignments cannot replace built-ins or reserved names.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_BUILTINS or node.keywords:
                raise NotationError("Only documented built-ins with positional arguments are supported.")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, str, bool, type(None))):
            raise NotationError("Only numbers, strings, booleans, and None literals are supported.")
    if mode == "exec":
        if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.Assign, ast.AugAssign, ast.Expr)):
            raise NotationError("Use exactly one assignment or allowed built-in call per process.")
        statement = tree.body[0]
        if isinstance(statement, ast.Expr) and not isinstance(statement.value, ast.Call):
            raise NotationError("A process needs an assignment or allowed built-in call.")
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target] if isinstance(statement, ast.AugAssign) else []
        if any(not isinstance(target, ast.Name) for target in targets):
            raise NotationError("Assign to a simple variable, not a tuple, attribute, or subscript.")
    return tree

def validate_node_text(kind: str, text: str) -> None:
    text = text.strip()
    if not text:
        raise NotationError("The shape has no readable text.")
    if kind == "terminal":
        if terminal_role(text) is None:
            raise NotationError("A terminal must contain only Start or End (Begin, Stop, Exit and Finish are also accepted).")
    elif kind == "connector":
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,7}", text):
            raise NotationError("A connector needs a short matching letter/number label.")
    elif kind == "input":
        match = INPUT.fullmatch(text)
        if not match or not all(safe_name(name) for name in input_names(match.group(1))):
            raise NotationError("Use Input name or Input name:int (also float or str).")
    elif kind == "output":
        if not text.lower().startswith("output "):
            raise NotationError("Use Output followed by an expression.")
        expression = text.split(None, 1)[1]
        try:
            validate_ast(expression)
        except NotationError:
            # Plain-language output labels are valid flowchart content. They
            # remain untrusted text and are quoted by the IR renderer.
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:\s+[A-Za-z][A-Za-z0-9]*)*", expression):
                raise
    elif kind == "decision":
        validate_ast(normalize_condition(text))
    elif kind == "iteration":
        if text.upper().startswith("WHILE "):
            validate_ast(text.split(None, 1)[1])
        else:
            match = FOR.fullmatch(text)
            if not match or not safe_name(match.group(1)):
                raise NotationError("Use WHILE condition or FOR name IN expression.")
            validate_ast(match.group(2))
    elif kind == "process":
        if text not in ("break", "continue", "return") and not is_supported_natural_operation(text):
            validate_ast(text, "exec")
    else:
        raise NotationError("This shape is outside the supported notation.")
