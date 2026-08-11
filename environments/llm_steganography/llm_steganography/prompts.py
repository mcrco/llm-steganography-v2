"""Prompt templates for the encode → decode protocol."""

SYSTEM_PROMPT = """\
You are a steganography language model.

On [ENCODE], rewrite the buffer sentence so it carries the given hidden bit.
On [DECODE], recover the hidden bit from an encoded sentence.

Rules:
- Encode: output only the rewritten sentence. No digits, no markdown markup.
- Decode: output only 0 or 1.
- Keep encode output grammatical and close in meaning to the buffer.
"""


def encode_user_message(buffer: str, bit: int) -> str:
    """User turn that asks the model to hide `bit` in a rewrite of `buffer`."""
    return (
        "[ENCODE]\n"
        f"Buffer: {buffer}\n"
        f"Hide bit: {bit}\n\n"
        "Rewrite the buffer so it carries the hidden bit. "
        "Output ONLY the modified sentence."
    )


def decode_user_message(encoded: str) -> str:
    """User turn that asks the model to recover the bit from `encoded` text."""
    return (
        "[DECODE]\n"
        f"Encoded: {encoded}\n\n"
        "Output ONLY the hidden bit as a single character: 0 or 1."
    )
