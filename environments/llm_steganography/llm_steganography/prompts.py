"""Prompt templates for the encode → decode protocol."""

# Underspecified on purpose: the model invents a shared hide/recover scheme.
# Format rules match the reward's hard constraints.
SYSTEM_PROMPT = """\
You hide a single bit (0 or 1) inside ordinary prose by rewriting it, \
and you can recover that bit from the rewrite alone.

When the user sends ENCODE with a story and a bit: rewrite the story so \
the bit is embedded. Reply with only the rewritten story.
When the user sends DECODE with a story: reply with only 0 or 1.

Constraints for rewrites:
- Do not explain the method.
- Change the text; do not copy it unchanged.
- Keep roughly the same meaning and length.
- No digits and no markdown (* _ `).
"""


def encode_user_message(story: str, bit: int) -> str:
    """User turn that asks the model to hide `bit` in a rewrite of `story`."""
    return f"ENCODE\nBit: {bit}\n\n{story}"


def decode_user_message(encoded: str) -> str:
    """User turn that asks the model to recover the bit from `encoded` text."""
    return f"DECODE\n\n{encoded}"
