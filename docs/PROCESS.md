# Process notes

How this workspace was stood up and why the environment looks the way it does.

## Context

The original CS159 LoRA/Gumbel experiment lives at [`mcrco/llm-steganography`](https://github.com/mcrco/llm-steganography) (kept as-is). This repo (`llm-steganography-v2`) is a clean rebuild as a **Prime Intellect v1 eval/RL environment** — no local training loop.

Claude’s August 2026 marking commitments make the *idea* timely again, but provider watermarking ≠ this task: here the model itself must learn a shared encode/decode convention in language.

## Setup steps taken

1. Install Prime CLI: `uv tool install prime`
2. Clear legacy training scripts from the repo root
3. `prime lab setup --agents cursor --no-interactive` — creates uv project, installs `verifiers`, configs, agent guidance
4. `uv run init llm-steganography` — scaffolds a native v1 taskset package under `environments/`
5. Implement encode→decode as a `SingleAgentEnv` with scripted turns + bit-recovery reward
6. `uv pip install -e environments/llm_steganography` and import/smoke checks

## Design choices

- **v1, not v0** — current verifiers direction; taskset owns data/scoring, env owns the two-turn control flow.
- **Scripted `Env.run()`** — decode input depends on the encode reply, so turns cannot be fully pre-baked like `alphabet_sort`.
- **Hard constraints gate reward** — digits/markup would trivialize hiding; invalid format → `0` reward.
- **Smoke buffers only** — fixed 8 sentences × 2 bits for fast iteration; larger corpora can replace `smoke_data.py` later.
- **No similarity reward yet** — bit recovery first, as agreed.
- **Package-local deps only** — environment `pyproject.toml` stays lean (`verifiers`); root lock stays lab-managed.

## Layout

```
.
├── configs/                         # lab eval / RL / endpoint templates
├── docs/PROCESS.md                  # this file
├── environments/
│   └── llm_steganography/           # installable env package
│       └── llm_steganography/
│           ├── prompts.py           # system + turn templates
│           ├── scoring.py           # parsers / constraints / reward helper
│           ├── smoke_data.py        # tiny buffer list
│           └── taskset.py           # Taskset + Env + rewards/metrics
├── pyproject.toml                   # workspace: verifiers
└── README.md
```

## Next natural steps

1. Run a real eval once an inference endpoint/API key is configured (`prime login` / `configs/endpoints.toml`)
2. Inspect format failure modes via `encode_valid` / `decode_valid`
3. Add a similarity secondary reward when bit recovery is non-zero
4. Grow beyond smoke buffers (Gutenberg-style corpora or a hosted dataset)
