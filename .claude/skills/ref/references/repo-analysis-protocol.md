# Repo Analysis Protocol

Structured protocol for analyzing a paper's code repository (cloned to `refs/related-work/<Name>/repo/`). The output goes to `refs/related-work/<Name>/repo_analysis.md` — one analysis per ref folder (NOT a global accumulator). Extract architectural patterns, component implementations, and reusable code so the user can understand how the repo is built and what to reuse or adapt.

## Project Context

For what to prioritize when analyzing, see **`docs/framing.md`** (the research identity). Single source — not restated here. The protocol below is general-purpose.

## Step 1: Orientation

```bash
# Top-level structure
find <repo> -maxdepth 2 -type f \( -name "*.py" -o -name "*.sh" -o -name "*.yaml" -o -name "*.yml" -o -name "*.md" -o -name "*.json" \) | head -80
# README
cat <repo>/README.md 2>/dev/null || cat <repo>/readme.md 2>/dev/null
# Configs
find <repo> -name "*.yaml" -o -name "*.yml" | head -20
# Dependencies
cat <repo>/requirements.txt 2>/dev/null || cat <repo>/setup.py 2>/dev/null || cat <repo>/pyproject.toml 2>/dev/null
```

Record the cloned commit hash (`git -C <repo> rev-parse --short HEAD`) for reproducibility.

## Step 2: Identify the 10 Key Components

Locate and analyze these. Not all repos have all components — skip what's absent and note it.

1. **Entry Points** — main scripts for training, evaluation, inference, data generation; their args/configs.
   `grep -rn "if __name__" <repo> --include="*.py" | head -20` ; `find <repo> -name "*.sh" | head -15`
2. **Special Token / Action Handling** — how action tokens (`<search>`, `<think>`, `<route>`, `<answer>`) are defined, parsed, and how generation is intercepted on emit.
   `grep -rn "<think>\|<route>\|<answer>\|<spawn>\|special_token\|stop_token" <repo> --include="*.py" | head -30`
3. **Environment / Tool Interaction** — how the model calls tools (retriever, LLM API, web search): HTTP? subprocess? in-process? How responses are injected back into context.
   `grep -rn "requests\.\|httpx\|urllib\|subprocess\|server\|endpoint\|api_call" <repo> --include="*.py" | head -20`
4. **Token Masking** — where retrieved/observation/env tokens are excluded from loss/gradient (attention masks, loss masks, token-type IDs).
   `grep -rn "mask\|loss_mask\|ignore_index\|masked" <repo> --include="*.py" | head -30`
5. **Reward Function** — where reward is computed; components (accuracy/F1/EM, format, cost, process reward); how cost is incorporated.
   `grep -rn "reward\|compute_reward\|f1_score\|exact_match\|penalty" <repo> --include="*.py" | head -30`
6. **RL Training Loop** — algorithm (GRPO/PPO/Reinforce++), rollout generation, count per prompt, advantage computation, cost-aware advantage.
   `grep -rn "grpo\|ppo\|reinforce\|advantage\|rollout\|group_size" <repo> --include="*.py" | head -30`
7. **SFT / Cold-Start Pipeline** — supervised phase before RL? how trajectories are generated and filtered.
   `grep -rn "sft\|cold.start\|supervised\|trajectory\|filter" <repo> --include="*.py" | head -20`
8. **Data Pipeline** — how training questions are loaded; format (jsonl/parquet/HF datasets); how episodes are stored.
   `grep -rn "dataset\|load_data\|jsonl\|parquet\|hotpotqa\|musique" <repo> --include="*.py" | head -20`
9. **Multi-GPU / Infrastructure** — retriever/trainer separation, GPU allocation, FSDP/DeepSpeed/DDP.
   `grep -rn "fsdp\|deepspeed\|ddp\|tensor_parallel\|num_gpu" <repo> --include="*.py" --include="*.yaml" --include="*.sh" | head -20`
10. **Evaluation** — how metrics (F1/EM, Pareto/cost) are computed; supported benchmarks.
    `grep -rn "evaluate\|f1\|exact_match\|pareto\|benchmark\|eval_dataset" <repo> --include="*.py" | head -20`

## Step 3: Produce `repo_analysis.md`

Keep code/formulas copied from the repo inside fenced code blocks (verbatim). For any *prose* math (e.g. describing a reward or advantage formula in your own words), use LaTeX — `$...$` inline, `$$...$$` display — not Unicode super/subscripts.

Write to `refs/related-work/<Name>/repo_analysis.md` using this format:

```markdown
# Repo Analysis: [REPO_NAME]

**Path:** `refs/related-work/<Name>/repo/` · **Source:** [git URL] · **Analyzed:** [date] (commit `<hash>`)

## Overview
- **Paper:** [title] (see `paper_analysis.md`)
- **Framework:** [veRL / OpenRLHF / TRL / custom / prompting-only / ...]
- **RL Algorithm:** [GRPO / PPO / Reinforce++ / DPO / none ...]
- **Base Model:** [Qwen-2.5-7B / Llama-3 / ...]
- **Key Innovation:** [one sentence]

## File Structure Map
[directory tree of key files with one-line descriptions]

## Component Inventory
[For each of the 10 components present: location (file:line), method, key details, actual formulas/hyperparameters from code. Use tables where it aids scanning. Note absent components explicitly.]

## Relevance to this project
### Components We Can Directly Reuse
[list with file paths]
### Components We Need to Modify
[list with what changes are needed]
### Components That Don't Apply
[list with why]
### Key Code Snippets Worth Studying
[file:line ranges for the most instructive sections]
```

## Adaptive Focus

If the caller specifies a focus (e.g. "focus on reward design"), deep-dive that component instead of a shallow pass over all 10. With no focus, infer the repo's primary contribution from its README/structure and weight the analysis toward those components. If the repo doesn't map to the template (a library, a prompting-only system with no RL), skip irrelevant sections and note what's absent rather than forcing the template.
