# TurboLoRA

Boosting TinyLoRA with Bayesian optimization: when an adapter has only a handful of trainable scalars, gradient-free search over them can replace RL.

Everything targets math reasoning (GSM8K / MATH-style tasks, graded with `math-verify`) on Unity HPC.

## Adapters (`src/turbolora/adapters.py`)

| name | trainable per module | ΔW |
|---|---|---|
| `lora` | A, B (rank r) | BA |
| `loraxs` | R ∈ ℝ^{r×r} | UΣ R Vᵀ (frozen truncated SVD) |
| `tinylora` | v ∈ ℝᵘ, one global v (`--tie`) or one per module (`--no-tie`) | UΣ (Σᵢ vᵢPᵢ) Vᵀ, fixed random Pᵢ |
| `turbolora` | same as TinyLoRA | U√Σ (Σᵢ vᵢPᵢ) √ΣVᵀ, trained by BO instead of GRPO |

All of them export as a standard PEFT LoRA dir so vLLM can eval them unchanged.

### Parameter budget

With `n` layers x 7 adapted projections (Qwen2.5-7B: 28 x 7 = 196 modules), every adapter is swept with `--rank`:

| adapter | trainable params | notes |
|---|---|---|
| `lora` | Σ r (d_in + d_out) — ~1.3M at r=1 on Qwen2.5-7B, can't go lower | alpha = 2r |
| `loraxs` | 196 r² — 784 at r=2 | R ∈ ℝ^{r×r} per module |
| `tinylora` | r² tied (default), 196 r² with `--no-tie` | `--proj-dim` defaults to r² (full R basis) |
| `turbolora` | u · ⌈196 / n_tie⌉ | `--proj-dim`, `--tie` (BO PoC defaults: u=1, tie=98) |

TinyLoRA with u = r² is LoRA-XS with a random basis for R, so `--no-tie` matches LoRA-XS's count; the default `--tie`
shares one v across the whole network and the count is just r²: r=1 → 1, r=2 → 4, r=8 → 64, r=32 → 1024.
The paper's 13-parameter config is `--rank 2 --proj-dim 13` (u > r² is only meaningful with tying).

## Training setup (`grpo.py`)

TinyLoRA-paper recipe: 64 problems x 4 rollouts per optimizer step, 3 epochs, no KL, clip 0.2, constant LR with
10 warmup steps, AdamW-8bit, completions capped at 1024 tokens (`--max-completion`). Rollouts come from a
colocated vLLM (~50% of VRAM); the base and adapter are kept in bf16. Prompts longer than 512 tokens are dropped
(75 of 8,521 on `hard`) because the colocated vLLM path never truncates them. Runs are named
`outputs/runs/<model>/<task>/<adapter>-<loss>-lr<lr>[-<cfg>]/seed<N>`; `run.json` holds the resolved config and
`checkpoint-*/` (every 25 steps) the curves the dashboard plots.

### Learning rates

`lora` defaults to 5e-6: the optimal LoRA LR is ~10x the full-FT LR of the same task in both SFT and RL
([LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)), and SimpleRL-Zoo trains full-FT at 5e-7.

The frozen-SVD adapters need far hotter LRs — the LoRA-XS paper fine-tunes R at 4e-3 (math instruction tuning,
r ≤ 64; 7e-4 at r = 128), 1e-3 (commonsense reasoning) and 6e-4–2e-3 (GLUE), and TinyLoRA re-sweeps LR per update
size up to 2e-4 because "changes in update size are known to alter effective learning rate". Lacking compute for a per-config sweep, `loraxs`/`tinylora` default to an equal-update-norm rule:
Adam moves every parameter ~lr per step, so the first-step norm is ‖ΔR‖ ≈ lr·r (LoRA-XS, unit basis) or
lr·r·√u (TinyLoRA, ‖Pᵢ‖ ≈ r), and the default sets it to 1e-3 for every config — `lr = 1e-3 / (r·√u)`
(`R_STEP_NORM` in `grpo.py`, `--lr` overrides). The constant is the geometric midpoint of what the fixed-5e-6 pilot
runs showed: configs at ‖ΔR‖ ≤ 3e-4 per step stayed flat for 200+ steps, and the one at 5e-3 learned fast then
diverged. Since all these adapters share the same frozen U, Σ, Vᵀ, equal ‖ΔR‖ is equal weight-space speed.

Current sweep: Qwen2.5-7B (base) on `hard`, seed 0, rank ∈ {1, 2, 8, 32} for `lora`, `loraxs`,
`tinylora` and `tinylora --no-tie` (`CFG=r<rank>[-notie]`).

## Layout

```
src/turbolora/
  models.py          MODELS: HF id + raw-text prompt style (Qwen2.5, Llama 3, Ministral, DeepSeek)
  tasks.py           TASKS: SimpleRL-Zoo tiers (easy/medium/hard) for training, gsm8k/math500/aime24/amc23/minerva/olympiad for eval; extract/grade/reward
  adapters.py        adapter attach/export
  grpo.py            shared GRPO/GSPO trainer (Unsloth + TRL), preempt-safe checkpoints
  train_{lora,loraxs,tinylora}.py   GRPO entry points
  bo.py              generic GP + Thompson-sampling search with heteroskedastic noise
  train_turbolora.py BO entry point: objective = vLLM pass rate on a random train subset
  eval.py            greedy vLLM eval of a base model or a trained adapter
slurm/               baseline.sh, train.sh, bo.sh, eval.sh
dashboard/           uv run dashboard/serve.py -> live dashboard at localhost:8000 (baselines, runs, curves)
                     uv run dashboard/build.py -> standalone dashboard.html snapshot to share
tests/
collaborators-poc/   original proof-of-concept BO pipeline (kept for reference)
```

Outputs are gitignored: `outputs/baselines/<model>/<task>.json`, `outputs/runs/<model>/<task>/<cfg>/seed<N>/` (with `run.json`, `final_adapter/`, `eval/`).

## Usage

```bash
# baseline eval of an untrained model
MODEL=qwen2.5-7b TASKS="gsm8k math500" sbatch slurm/baseline.sh

# GRPO, 3 seeds (array 0-2); CFG suffixes the run dir so configs of one adapter don't collide
MODEL=qwen2.5-7b TASK=hard ADAPTER=tinylora CFG=r2 sbatch slurm/train.sh --rank 2

# single seed, rank sweep
for r in 1 2 8 32; do MODEL=qwen2.5-7b TASK=hard ADAPTER=loraxs CFG=r$r sbatch -a 0 slurm/train.sh --rank $r; done
MODEL=qwen2.5-7b TASK=hard ADAPTER=tinylora CFG=r2-notie sbatch -a 0 slurm/train.sh --rank 2 --no-tie

# BO, 3 seeds
MODEL=qwen2.5-7b TASK=easy sbatch slurm/bo.sh --tie 98 --proj-dim 1

# eval a trained adapter
ADAPTER=outputs/runs/qwen2.5-7b/hard/tinylora-grpo-lr5e-6-r2/seed0/final_adapter TASKS="gsm8k math500" sbatch slurm/eval.sh

# locally / interactively
uv run -m turbolora.train_tinylora --model qwen2.5-7b --task hard --out outputs/runs/x --max-steps 3
uv run -m turbolora.eval --model qwen2.5-7b --tasks gsm8k
uv run pytest
```

Jobs run on `gpu-preempt` with requeue; training checkpoints on SIGTERM/SIGUSR1 and resumes from the last checkpoint (BO resumes from `trials.json`). `--qos=short` evals allow one queued job per user, so submit them one at a time. `HF_HOME` points at `.hf-cache/`; nothing is written outside the repo.

## References

- TinyLoRA: arXiv 2602.04118
- LoRA-XS: arXiv 2405.17604
- SimpleRL-Zoo data tiers and prompts: arXiv 2503.18892
- LoRA Without Regret (LoRA LR = 10x full-FT): https://thinkingmachines.ai/blog/lora/
