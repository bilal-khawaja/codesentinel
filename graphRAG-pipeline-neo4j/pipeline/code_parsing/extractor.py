import ast
import builtins
from bisect import bisect_right
import inspect
from dataclasses import dataclass, field
from typing import List, Optional

from .language_spec import LanguageSpec
from .parser_loader import get_parser

BUILTIN_NAMES = set(dir(builtins))
MAX_RAW_CALL_CHARS = 120


@dataclass
class Symbol:
    kind: str                 # "Function", "Class" or "Variable" (methods are Functions with a parent)
    name: str
    qualname: str             # "Class.method", "outer.inner", "CONSTANT" ...
    parent: Optional[str]     # qualname of the enclosing class or function, None at module level
    start_line: int           # 1-based, includes decorators
    end_line: int
    signature: str
    docstring: str            # first docstring line, "" if none
    source: str
    bases: List[str] = field(default_factory=list)   # Class: base class expressions, e.g. "models.Base"
    calls: List[str] = field(default_factory=list)   # Function: every call, e.g. "Gateway.charge", "self.verify"
    refs: List[str] = field(default_factory=list)    # Function: non-call reads of names/attributes
    locals: List[str] = field(default_factory=list)  # Function: parameters and locally bound names


@dataclass
class CodeInfo:
    language: str
    line_count: int
    module_docstring: str = ""
    symbols: List[Symbol] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)  # "alias|module|name" ("from x import y as z" -> "z|x|y")


def extract(source: str, spec: LanguageSpec) -> CodeInfo:
    """Parse source into symbols (functions, classes, variables, nested ones included)
    and keep every call/reference as a plain dotted string. Nothing is resolved here:
    resolver.py turns these strings into edges once it can see the other files."""
    data = source.encode("utf-8")
    tree = get_parser(spec.name).parse(data)  # nodes point into the tree: it must outlive the walk
    info = CodeInfo(language=spec.name, line_count=len(source.splitlines()))
    _Walker(data, spec, info).walk(tree.root_node)
    return info


class _Walker:
    def __init__(self, data: bytes, spec: LanguageSpec, info: CodeInfo):
        self.data = data
        self.spec = spec
        self.info = info
        # Line numbers come from byte offsets: node.start_point / end_point crash
        # (access violation) in the tree-sitter build we run on, so they are never used.
        self.line_starts = [0] + [i + 1 for i, byte in enumerate(data) if byte == 10]

    def start_line(self, node) -> int:
        return bisect_right(self.line_starts, node.start_byte)

    def end_line(self, node) -> int:
        return bisect_right(self.line_starts, max(node.end_byte - 1, node.start_byte))

    def text(self, node) -> str:
        return self.data[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def walk(self, root):
        self.info.module_docstring = self._docstring(root)
        for node in root.named_children:
            definition = self._unwrap(node)
            if definition.type in self.spec.import_types:
                self.info.imports.extend(self._import_entries(definition))
            elif definition.type in self.spec.class_types:
                self._add_class(node, definition, parent=None)
            elif definition.type in self.spec.function_types:
                self._add_function(node, definition, parent=None)
            else:
                self._add_variables(node, parent=None)

    def _unwrap(self, node):
        """Decorated definitions wrap the real function/class node."""
        if node.type in self.spec.wrapper_types:
            return node.child_by_field_name("definition") or node
        return node

    def _is_function(self, node) -> bool:
        return self._unwrap(node).type in self.spec.function_types

    def _is_class(self, node) -> bool:
        return self._unwrap(node).type in self.spec.class_types

    def _qualname(self, parent, name) -> str:
        return f"{parent}.{name}" if parent else name

    def _import_entries(self, node) -> List[str]:
        """"alias|module|name" per bound name: `from x import y as z` -> "z|x|y",
        `import a.b as c` -> "c|a.b|", `import a.b` -> "a|a.b|"."""
        entries = []
        if node.type == "import_from_statement":
            module = self.text(node.child_by_field_name("module_name"))
            for name_node in node.children_by_field_name("name"):
                name, alias = self._name_and_alias(name_node)
                entries.append(f"{alias}|{module}|{name}")
            return entries
        for child in node.named_children:
            name, alias = self._name_and_alias(child)
            if alias == name:
                alias = name.split(".")[0]
            entries.append(f"{alias}|{name}|")
        return entries

    def _name_and_alias(self, node):
        if node.type == "aliased_import":
            return self.text(node.child_by_field_name("name")), self.text(node.child_by_field_name("alias"))
        name = self.text(node)
        return name, name

    def _add_class(self, outer, node, parent):
        name = self.text(node.child_by_field_name("name"))
        body = node.child_by_field_name("body")
        symbol = self._add_symbol("Class", name, parent, outer, node, body)
        superclasses = node.child_by_field_name("superclasses")
        if superclasses is not None:
            for base in superclasses.named_children:
                if base.type != "keyword_argument":
                    symbol.bases.append(self._chain_text(base))
        for child in body.named_children:
            if self._is_function(child):
                self._add_function(child, self._unwrap(child), parent=symbol.qualname)
            elif self._is_class(child):
                self._add_class(child, self._unwrap(child), parent=symbol.qualname)
            else:
                self._add_variables(child, parent=symbol.qualname)

    def _add_function(self, outer, node, parent):
        name = self.text(node.child_by_field_name("name"))
        body = node.child_by_field_name("body")
        symbol = self._add_symbol("Function", name, parent, outer, node, body)
        _FunctionScan(self, symbol, node).run()

    def _add_symbol(self, kind, name, parent, outer, node, body) -> Symbol:
        signature = " ".join(self.data[node.start_byte:body.start_byte].decode("utf-8", errors="replace").split())
        symbol = Symbol(
            kind=kind,
            name=name,
            qualname=self._qualname(parent, name),
            parent=parent,
            start_line=self.start_line(outer),
            end_line=self.end_line(outer),
            signature=signature.rstrip(":"),
            docstring=self._docstring(body),
            source=self.text(outer),
        )
        self.info.symbols.append(symbol)
        return symbol

    def _add_variables(self, node, parent):
        """Module- or class-level `NAME = value` / `NAME: type = value` statements."""
        if node.type != "expression_statement" or not node.named_children:
            return
        assignment = node.named_children[0]
        if assignment.type != "assignment":
            return
        for name in self._target_names(assignment.child_by_field_name("left")):
            self._add_variable(name, parent, node)

    def _add_variable(self, name, parent, statement):
        qualname = self._qualname(parent, name)
        if any(s.qualname == qualname for s in self.info.symbols):
            return  # re-assignment of a variable we already track
        text = self.text(statement)
        self.info.symbols.append(Symbol(
            kind="Variable",
            name=name,
            qualname=qualname,
            parent=parent,
            start_line=self.start_line(statement),
            end_line=self.end_line(statement),
            signature=" ".join(text.split())[:100],
            docstring="",
            source=text,
        ))

    def _target_names(self, target) -> List[str]:
        """Plain names bound by an assignment target; attribute/subscript targets bind nothing."""
        if target is None:
            return []
        if target.type == "identifier":
            return [self.text(target)]
        if target.type in ("pattern_list", "tuple_pattern", "list_pattern", "list_splat_pattern", "tuple", "list"):
            return [name for child in target.named_children for name in self._target_names(child)]
        return []

    def _chain_text(self, node) -> str:
        """Dotted text for names/attributes, whitespace-collapsed raw text for anything else."""
        chain = _chain(self, node, [], [])
        if chain is not None:
            return ".".join(chain)
        return " ".join(self.text(node).split())[:MAX_RAW_CALL_CHARS]

    def _docstring(self, block) -> str:
        """First line of the docstring: a string literal as first statement of the block."""
        first = block.named_children[0] if block.named_children else None
        if first is None or first.type != "expression_statement" or not first.named_children:
            return ""
        literal = first.named_children[0]
        if literal.type != "string":
            return ""
        raw = self.text(literal)
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raw = raw.strip("\"' ")
        lines = inspect.cleandoc(raw).splitlines()
        return lines[0] if lines else ""


def _chain(walker: "_Walker", node, calls: List[str], extra_nodes: list) -> Optional[List[str]]:
    """Dotted segments for `a.b.c`. A call in the middle is looked through, so
    `Gateway().charge` gives ["Gateway", "charge"], and the inner call is added to
    `calls` (its argument nodes go to `extra_nodes` so they still get scanned).
    Returns None when the expression is not a plain name chain."""
    if node.type == "identifier":
        return [walker.text(node)]
    if node.type == "attribute":
        base = _chain(walker, node.child_by_field_name("object"), calls, extra_nodes)
        if base is None:
            return None
        return base + [walker.text(node.child_by_field_name("attribute"))]
    if node.type in walker.spec.call_types:
        chain = _chain(walker, node.child_by_field_name("function"), calls, extra_nodes)
        if chain is None:
            return None
        calls.append(".".join(chain))
        extra_nodes.append(node.child_by_field_name("arguments"))
        return chain
    return None


class _FunctionScan:
    """Reads one function body: calls, name references, local names, nested
    definitions and (in methods) `self.x = ...` instance variables. Nested
    functions and classes are scanned on their own, so their calls are never
    credited to the enclosing function."""

    def __init__(self, walker: _Walker, symbol: Symbol, node):
        self.walker = walker
        self.symbol = symbol
        self.node = node
        self.locals = set()
        self.globals = set()
        self.calls: List[str] = []
        self.refs: List[str] = []

    def run(self):
        parameters = self.node.child_by_field_name("parameters")
        if parameters is not None:
            for parameter in parameters.named_children:
                self._bind_first_identifier(parameter)
        self._visit(self.node.child_by_field_name("body"))

        local_names = self.locals - self.globals
        self.symbol.locals = sorted(local_names)
        self.symbol.calls = list(dict.fromkeys(self.calls))
        self.symbol.refs = [
            ref for ref in dict.fromkeys(self.refs)
            if ref.split(".")[0] in ("self", "cls")
            or (ref.split(".")[0] not in local_names and ref.split(".")[0] not in BUILTIN_NAMES)
        ]

    def _bind_first_identifier(self, node):
        stack = [node]
        while stack:
            current = stack.pop(0)
            if current.type == "identifier":
                self.locals.add(self.walker.text(current))
                return
            stack = list(current.named_children) + stack

    def _bind_targets(self, target):
        for name in self.walker._target_names(target):
            self.locals.add(name)
        if target is not None and target.type == "attribute":
            self._record_instance_variable(target)

    def _record_instance_variable(self, target):
        """`self.x = ...` inside a method defines instance variable x on the class."""
        owner = self.symbol.parent
        object_node = target.child_by_field_name("object")
        is_method = owner is not None and any(
            s.qualname == owner and s.kind == "Class" for s in self.walker.info.symbols
        )
        if is_method and object_node.type == "identifier" and self.walker.text(object_node) == "self":
            statement = target
            while statement.parent is not None and statement.type != "expression_statement":
                statement = statement.parent
            self.walker._add_variable(self.walker.text(target.child_by_field_name("attribute")), owner, statement)

    def _visit(self, node):
        if node is None:
            return
        walker, kind = self.walker, node.type

        if walker._is_function(node) or walker._is_class(node):
            definition = walker._unwrap(node)
            self.locals.add(walker.text(definition.child_by_field_name("name")))
            if walker._is_function(node):
                walker._add_function(node, definition, parent=self.symbol.qualname)
            else:
                walker._add_class(node, definition, parent=self.symbol.qualname)
        elif kind in walker.spec.call_types:
            self._visit_call(node)
        elif kind == "attribute":
            self._visit_reference(node)
        elif kind == "identifier":
            self.refs.append(walker.text(node))
        elif kind in ("assignment", "augmented_assignment"):
            self._bind_targets(node.child_by_field_name("left"))
            for field_name in ("right", "type"):
                self._visit(node.child_by_field_name(field_name))
        elif kind in ("for_statement", "for_in_clause"):
            self._bind_targets(node.child_by_field_name("left"))
            self._visit_children(node, skip=node.child_by_field_name("left"))
        elif kind == "named_expression":
            self._bind_targets(node.child_by_field_name("name"))
            self._visit(node.child_by_field_name("value"))
        elif kind == "as_pattern":
            for alias in node.children_by_field_name("alias"):
                self._bind_first_identifier(alias)
            self._visit_children(node, skip=node.child_by_field_name("alias"))
        elif kind == "lambda":
            parameters = node.child_by_field_name("parameters")
            for parameter in (parameters.named_children if parameters is not None else []):
                self._bind_first_identifier(parameter)
            self._visit(node.child_by_field_name("body"))
        elif kind in ("import_statement", "import_from_statement"):
            self.locals.update(entry.split("|")[0] for entry in walker._import_entries(node))
        elif kind in ("global_statement", "nonlocal_statement"):
            self.globals.update(walker.text(child) for child in node.named_children)
        elif kind == "keyword_argument":
            self._visit(node.child_by_field_name("value"))
        else:
            self._visit_children(node)

    def _visit_children(self, node, skip=None):
        for child in node.named_children:
            if skip is None or child.id != skip.id:
                self._visit(child)

    def _visit_call(self, node):
        function = node.child_by_field_name("function")
        extra_nodes = []
        chain = _chain(self.walker, function, self.calls, extra_nodes)
        if chain is not None:
            self.calls.append(".".join(chain))
            for extra in extra_nodes:
                self._visit(extra)
        else:  # e.g. `handlers[0](x)`: keep the raw text, still scan what is inside
            self.calls.append(" ".join(self.walker.text(function).split())[:MAX_RAW_CALL_CHARS])
            self._visit(function.child_by_field_name("object") if function.type == "attribute" else function)
        self._visit(node.child_by_field_name("arguments"))

    def _visit_reference(self, node):
        extra_nodes = []
        chain = _chain(self.walker, node, self.calls, extra_nodes)
        if chain is not None:
            self.refs.append(".".join(chain))
            for extra in extra_nodes:
                self._visit(extra)
        else:
            self._visit(node.child_by_field_name("object"))
