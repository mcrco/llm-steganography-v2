"""llm-steganography package.

Hosted Training resolves `load_environment` from this module (v0 MultiTurnEnv).
v1 Taskset/Env classes are lazy-imported so a broken hosted v1 surface cannot
prevent the v0 entrypoint from loading.
"""

from llm_steganography.v0_env import load_environment

__all__ = ["load_environment", "LlmSteganographyEnv", "LlmSteganographyTaskset"]


def __getattr__(name: str):
    if name in ("LlmSteganographyEnv", "LlmSteganographyTaskset"):
        from llm_steganography import taskset as _taskset

        return getattr(_taskset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
