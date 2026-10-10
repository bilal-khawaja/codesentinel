import time
from contextlib import contextmanager


def log_stage(stage: str, message: str = ""):
    print(f"\n{'=' * 70}")
    print(f"[{stage}]")
    if message:
        print(message)
    print(f"{'=' * 70}")


@contextmanager
def trace_tool(tool_name: str, **kwargs):
    start = time.perf_counter()
    print(f"\n TOOL START: {tool_name}")

    if kwargs:
        for key, value in kwargs.items():
            print(f"   {key}: {value}")

    try:
        yield

    finally:
        elapsed = time.perf_counter() - start
        print(f"✅ TOOL END: {tool_name} ({elapsed:.2f}s)")