# llm-steganography

v1 verifiers environment for **linguistic steganography**: hide a bit in a TinyStories rewrite, then recover it.

## Protocol

Two isolated single-turn runs per episode (same agent/model):

1. **Encode** — rewrite a TinyStories story to carry bit `0` or `1`
2. **Decode** — fresh conversation; model sees only the encoded story (not the bit)

## Scoring

| Signal | Role |
| --- | --- |
| `bit_recovery` | Reward: `1.0` iff encode passes hard constraints and decode equals the target bit |
| `encode_valid` | Metric: encode is non-empty, digit-free, markup-free |
| `decode_valid` | Metric: decode reply is exactly `0` or `1` |

Semantic similarity is intentionally deferred.

## Data

Stories come from [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories). Defaults:

- `split=train`, `seed=0`
- word length band `[80, 180]`
- digit-containing stories dropped (matches encode hard constraints)
- each story paired with both bits → `num_tasks=80` uses 40 stories

## Install & eval

From the workspace root:

```bash
uv pip install -e environments/llm_steganography
uv run eval llm-steganography -n 4 -m Qwen/Qwen3.5-4B --no-push
```

`LlmSteganographyEnv` is exported from the same package, so eval picks it up automatically (no `--env.id` needed).

Story rewrites need headroom:

```bash
uv run eval @ configs/eval/llm-steganography-smoke.toml
```

Override task count / filters:

```bash
uv run eval llm-steganography -n 8 \
  --env.taskset.num_tasks 8 \
  --env.taskset.min_words 60 \
  --env.taskset.max_words 120 \
  --no-push
```
