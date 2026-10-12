import sys
import time
from contextlib import contextmanager


def safe_print(*args, **kwargs):
    """print() that cannot raise on text the console encoding cannot show. Windows consoles
    default to cp1252, which has no emoji, no Arabic, no CJK: a plain print() of such text
    (a tool argument, a retrieved row) raises UnicodeEncodeError. Tracing is only a log, so
    it must never be able to fail the tool it is logging. Unencodable characters become '?'."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        encoding = getattr(kwargs.get("file") or sys.stdout, "encoding", None) or "ascii"
        text = " ".join(str(a) for a in args).encode(encoding, errors="replace").decode(encoding)
        print(text, **{k: v for k, v in kwargs.items() if k in ("file", "end", "flush")})


def log_stage(stage: str, message: str = ""):
    safe_print(f"\n{'=' * 70}")
    safe_print(f"[{stage}]")
    if message:
        safe_print(message)
    safe_print(f"{'=' * 70}")


@contextmanager
def trace_tool(tool_name: str, **kwargs):
    start = time.perf_counter()
    safe_print(f"\n TOOL START: {tool_name}")

    if kwargs:
        for key, value in kwargs.items():
            safe_print(f"   {key}: {value}")

    try:
        yield

    finally:
        elapsed = time.perf_counter() - start
        safe_print(f"TOOL END: {tool_name} ({elapsed:.2f}s)")