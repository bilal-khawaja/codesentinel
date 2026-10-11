from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class LanguageSpec:
    """Tree-sitter node type names for one language. Adding a language = adding a spec."""
    name: str
    extensions: tuple
    function_types: tuple
    class_types: tuple
    import_types: tuple
    call_types: tuple
    wrapper_types: tuple = ()  # nodes that wrap a definition, e.g. python decorators


PYTHON = LanguageSpec(
    name="python",
    extensions=(".py",),
    function_types=("function_definition",),
    class_types=("class_definition",),
    import_types=("import_statement", "import_from_statement"),
    call_types=("call",),
    wrapper_types=("decorated_definition",),
)

LANGUAGE_SPECS = [PYTHON]


def get_spec_for_path(path: str) -> Optional[LanguageSpec]:
    """Return the spec for a source file, or None if the extension is not code we parse."""
    extension = Path(path).suffix.lower()
    for spec in LANGUAGE_SPECS:
        if extension in spec.extensions:
            return spec
    return None
