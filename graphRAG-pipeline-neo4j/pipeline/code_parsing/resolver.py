"""Turns the raw references stored on code entities into graph edges.

Input is a list of entity dicts as stored in Neo4j (id, type, file, qualname, parent, name,
bases, calls, refs, locals, imports); output is a set of (source_id, edge_type, target_id).
Pure functions, no database, so the same code resolves a single file or a whole project:
whatever is in the list is what can be resolved.

Edges: CALLS (Function -> Function/Class), USES (Function -> anything it reads),
INHERITS (Class -> Class), IMPORTS (Module -> Module).
Anything that cannot be resolved (builtins, third-party code, values of unknown type)
creates no edge; the raw text stays on the entity for the LLM to read."""
from typing import Dict, List, Optional, Set, Tuple, Union

Entity = dict
ModuleRef = Tuple[str, str]  # ("module", dotted name)
Edge = Tuple[str, str, str]


def resolve_links(entities: List[Entity]) -> Set[Edge]:
    return _Resolver(entities).edges()


class _Resolver:
    def __init__(self, entities: List[Entity]):
        self.by_key: Dict[Tuple[str, str], Entity] = {}      # (file, qualname) -> entity
        self.members: Dict[Tuple[str, Optional[str]], Dict[str, Entity]] = {}  # (file, parent) -> {name: entity}
        self.modules: Dict[str, Entity] = {}                 # file -> Module entity
        self.entities = entities
        for entity in entities:
            if entity["type"] == "Module":
                self.modules[entity["file"]] = entity
            else:
                self.by_key[(entity["file"], entity["qualname"])] = entity
                self.members.setdefault((entity["file"], entity.get("parent")), {})[entity["name"]] = entity
        self.import_maps = {file: _parse_imports(module) for file, module in self.modules.items()}

    def edges(self) -> Set[Edge]:
        edges: Set[Edge] = set()
        for entity in self.entities:
            if entity["type"] == "Function":
                for chain in entity.get("calls") or []:
                    self._link(edges, entity, chain, "CALLS")
                for chain in entity.get("refs") or []:
                    self._link(edges, entity, chain, "USES")
            elif entity["type"] == "Class":
                for chain in entity.get("bases") or []:
                    target, _ = self._resolve_chain(chain, entity)
                    if _is_entity(target) and target["type"] == "Class":
                        edges.add((entity["id"], "INHERITS", target["id"]))
            elif entity["type"] == "Module":
                for _, module, imported in self.import_maps[entity["file"]].values():
                    module = self._absolute_module(entity["file"], module)
                    submodule = f"{module}.{imported}" if module else imported
                    for dotted in (module, submodule if imported else ""):  # `from pkg import utils` imports pkg.utils
                        target_file = self._file_of(dotted)
                        if target_file and target_file != entity["file"]:
                            edges.add((entity["id"], "IMPORTS", self.modules[target_file]["id"]))
        return edges

    def _link(self, edges: Set[Edge], source: Entity, chain: str, kind: str):
        target, last = self._resolve_chain(chain, source)
        if _is_entity(target):
            if kind == "CALLS" and target["type"] != "Variable":
                edges.add((source["id"], "CALLS", target["id"]))
            elif target["id"] != source["id"]:
                edges.add((source["id"], "USES", target["id"]))
        elif _is_entity(last) and last["type"] == "Variable":
            edges.add((source["id"], "USES", last["id"]))  # e.g. self.client.get(): uses `client`

    def _resolve_chain(self, chain: str, scope: Entity):
        """Follow `a.b.c` one segment at a time. Returns (final, last_entity_reached);
        final is None when a segment cannot be resolved."""
        segments = chain.split(".")
        current = self._head(segments[0], scope)
        last = current if _is_entity(current) else None
        for segment in segments[1:]:
            if current is None:
                break
            current = self._step(current, segment)
            if _is_entity(current):
                last = current
        return current, last

    def _head(self, name: str, scope: Entity) -> Union[Entity, ModuleRef, None]:
        file = scope["file"]
        if name in ("self", "cls"):
            return self._enclosing_class(scope)
        current = scope
        while current is not None:  # innermost function scope outwards (class scopes are not visible by bare name)
            if current["type"] == "Function":
                nested = self.by_key.get((file, f"{current['qualname']}.{name}"))
                if nested:
                    return nested
                if name in (current.get("locals") or []):
                    return None
            current = self._parent(current)
        top_level = self.by_key.get((file, name))
        if top_level:
            return top_level
        return self._import_binding(file, name)

    def _step(self, current: Union[Entity, ModuleRef], segment: str) -> Union[Entity, ModuleRef, None]:
        if isinstance(current, tuple):  # module: its top-level symbol, or a submodule
            file = self._file_of(current[1])
            if file and (file, segment) in self.by_key:
                return self.by_key[(file, segment)]
            submodule = f"{current[1]}.{segment}"
            return ("module", submodule) if self._file_of(submodule) else None
        if current["type"] == "Class":
            return self._member(current, segment)
        return None  # the type of a function result or variable is unknown

    def _member(self, cls: Entity, name: str) -> Optional[Entity]:
        for klass in self._mro(cls):
            found = self.members.get((klass["file"], klass["qualname"]), {}).get(name)
            if found:
                return found
        return None

    def _mro(self, cls: Entity) -> List[Entity]:
        """The class followed by its bases, nearest first. Bases may live in other files."""
        ordered, queue, seen = [], [cls], set()
        while queue:
            klass = queue.pop(0)
            if klass["id"] in seen:
                continue
            seen.add(klass["id"])
            ordered.append(klass)
            for base in klass.get("bases") or []:
                target, _ = self._resolve_chain(base, klass)
                if _is_entity(target) and target["type"] == "Class":
                    queue.append(target)
        return ordered

    def _enclosing_class(self, scope: Entity) -> Optional[Entity]:
        current = scope
        while current is not None:
            if current["type"] == "Class":
                return current
            current = self._parent(current)
        return None

    def _parent(self, entity: Entity) -> Optional[Entity]:
        if not entity.get("parent"):
            return None
        return self.by_key.get((entity["file"], entity["parent"]))

    def _import_binding(self, file: str, name: str) -> Union[Entity, ModuleRef, None]:
        binding = self.import_maps.get(file, {}).get(name)
        if binding is None:
            return None
        _, module, imported = binding
        module = self._absolute_module(file, module)
        if not imported:  # `import a.b as c`
            return ("module", module)
        module_file = self._file_of(module)
        if module_file and (module_file, imported) in self.by_key:
            return self.by_key[(module_file, imported)]
        submodule = f"{module}.{imported}" if module and not module.endswith(".") else f"{module}{imported}"
        return ("module", submodule) if self._file_of(submodule) else None

    def _absolute_module(self, file: str, module: str) -> str:
        """Resolve a relative import (`.utils`, `..pkg.x`) against the importing file's package."""
        if not module.startswith("."):
            return module
        dots = len(module) - len(module.lstrip("."))
        package = file.replace("\\", "/").split("/")[:-1]
        package = package[:len(package) - (dots - 1)] if dots > 1 else package
        rest = module.lstrip(".")
        return ".".join(package + ([rest] if rest else []))

    def _file_of(self, dotted: str) -> Optional[str]:
        """File for a dotted module name: exact match, else a unique suffix match
        (imports are written from the project root, file ids may carry extra folders)."""
        if not dotted:
            return None
        exact = [f for f in self.modules if _dotted(f) == dotted]
        if len(exact) == 1:
            return exact[0]
        suffix = [f for f in self.modules if _dotted(f).endswith("." + dotted)]
        return suffix[0] if len(suffix) == 1 and not exact else None


def _dotted(file: str) -> str:
    """"pkg/utils.py" -> "pkg.utils", "pkg/__init__.py" -> "pkg"."""
    parts = file.replace("\\", "/").rsplit(".", 1)[0].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _parse_imports(module: Entity) -> Dict[str, Tuple[str, str, str]]:
    bindings = {}
    for entry in module.get("imports") or []:
        alias, module_name, imported = entry.split("|", 2)
        bindings[alias] = (alias, module_name, imported)
    return bindings


def _is_entity(value) -> bool:
    return isinstance(value, dict)
