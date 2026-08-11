# llm-steganography

Prime Intellect **v1** environment for linguistic steganography: hide a bit in a sentence rewrite, then decode it.

Built for eval / RL via [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers). See [docs/PROCESS.md](docs/PROCESS.md) for setup rationale.

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

The package exports both `LlmSteganographyTaskset` and `LlmSteganographyEnv`; eval auto-selects the env from the taskset package.

## Environment

| Piece | Role |
| --- | --- |
| Encode turn | Rewrite buffer to carry bit `0`/`1` |
| Decode turn | Emit only that bit |
| Reward | Bit recovery, gated by hard format constraints |
| Data | 8 smoke buffers × 2 bits (`num_tasks=16`) |

Package: [`environments/llm_steganography`](environments/llm_steganography).

## Project layout

```
configs/                 # endpoints, eval, RL templates from prime lab setup
docs/PROCESS.md          # how/why this was built
environments/
  llm_steganography/     # the v1 taskset + scripted encode→decode env
```
