# llm-steganography

v1 verifiers environment for **linguistic steganography**: hide a bit in a natural-language rewrite, then recover it.

## Protocol

Two scripted turns per episode (same model):

1. **Encode** — rewrite a buffer sentence to carry bit `0` or `1`
2. **Decode** — output only that bit

## Scoring

| Signal | Role |
| --- | --- |
| `bit_recovery` | Reward: `1.0` iff encode passes hard constraints and decode equals the target bit |
| `encode_valid` | Metric: encode is non-empty, digit-free, markup-free |
| `decode_valid` | Metric: decode reply is exactly `0` or `1` |

Semantic similarity is intentionally deferred.

## Data

`smoke_data.py` holds 8 short buffers. Default `num_tasks=16` pairs each buffer with both bits.

## Install & eval

From the workspace root:

```bash
uv pip install -e environments/llm_steganography
uv run eval llm-steganography -n 4 -m openai/gpt-4.1-mini --no-push
```

`LlmSteganographyEnv` is exported from the same package, so eval picks it up automatically (no `--env.id` needed).

Override task count:

```bash
uv run eval llm-steganography -n 8 --env.taskset.num_tasks 8 --no-push
```
