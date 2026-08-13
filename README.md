# llm-steganography-v2

Prime Intellect **v1** environment for linguistic steganography: hide a bit in a TinyStories rewrite, then decode it.

Built for eval / RL via [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers).

The original CS159 LoRA/Gumbel experiment lives untouched at [`mcrco/llm-steganography`](https://github.com/mcrco/llm-steganography).

## Quick start

```bash
# one-time: Prime CLI + auth
uv tool install -U prime
prime login

# workspace already includes verifiers; install the env package
uv pip install -e environments/llm_steganography

# smoke eval (needs a configured OpenAI-compatible endpoint)
uv run eval llm-steganography -n 4 -m openai/gpt-4.1-mini --no-push
```

The package exports both `LlmSteganographyTaskset` and `LlmSteganographyEnv`. The env defaults to the `null` (plain chat) harness.

## Environment

| Piece | Role |
| --- | --- |
| Encode turn | Rewrite a TinyStories story to carry bit `0`/`1` |
| Decode turn | Emit only that bit (isolated conversation) |
| Reward | Bit recovery, gated by hard format constraints |
| Data | TinyStories (`num_tasks=80` → 40 stories × both bits) |

Package: [`environments/llm_steganography`](environments/llm_steganography).

## Project layout

```
configs/                 # endpoints, eval, RL templates from prime lab setup
environments/
  llm_steganography/     # the v1 taskset + scripted encode→decode env
```
