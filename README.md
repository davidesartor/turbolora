# TurboLoRA

Boosting TinyLoRA with Bayesian optimization: when an adapter has only a handful of trainable scalars, gradient-free search over them can replace RL.

Everything targets math reasoning (GSM8K / MATH-style tasks, graded with `math-verify`) on Unity HPC.

## Adapters (`src/turbolora/adapters.py`)

| name | trainable per module | ΔW |
|---|---|---|
| `lora` | A, B (rank r) | BA |
| `loraxs` | R ∈ ℝ^{r×r} | UΣ R Vᵀ (frozen truncated SVD) |
| `tinylora` | v ∈ ℝᵘ, one global v (default) or one per module (`--untie`) | UΣ (Σᵢ vᵢPᵢ) Vᵀ, fixed random Pᵢ; trained by GRPO (`train_tinylora`) or TuRBO (`train_turbolora`) |

All of them export as a standard PEFT LoRA dir so vLLM can eval them unchanged.

### Parameter budget

With `n` layers x 7 adapted projections (Qwen2.5-7B: 28 x 7 = 196 modules), every adapter is swept with `--rank`:

| adapter | trainable params | notes |
|---|---|---|
| `lora` | Σ r (d_in + d_out) — ~1.3M at r=1 on Qwen2.5-7B, can't go lower | alpha = 2r |
| `loraxs` | 196 r² — 784 at r=2 | R ∈ ℝ^{r×r} per module |
| `tinylora` | r² tied (default), 196 r² with `--untie` | `--proj-dim` defaults to r² (full R basis) |

TinyLoRA with u = r² is LoRA-XS with a random basis for R, so `--untie` matches LoRA-XS's count; the default
shares one v across the whole network and the count is just r²: r=1 → 1, r=2 → 4, r=8 → 64, r=32 → 1024.
The paper's 13-parameter config is `--rank 2 --proj-dim 13` (u > r² is only meaningful with tying).

## Training setup (`grpo.py`)

TinyLoRA-paper recipe: 64 problems x 4 rollouts per optimizer step, 3 epochs, no KL, clip 0.2, constant LR with
10 warmup steps, AdamW-8bit, completions capped at 1024 tokens (`--max-completion`). Rollouts come from a
colocated vLLM (~50% of VRAM); the base and adapter are kept in bf16. Prompts longer than 512 tokens are dropped
(75 of 8,521 on `hard`) because the colocated vLLM path never truncates them. Runs are named
`outputs/runs/<family>/<model>/<adapter>-<loss>/<cfg>[-lr<lr>]/seed<N>`; the train set is per model (`run.json` records it); `run.json` holds the resolved config and
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

### Sweeps (3 seeds each)

| models | task | `lora` r | `loraxs` r | `tinylora-grpo` u | `tinylora-turbo` u |
|---|---|---|---|---|---|
| Qwen2.5-1.5B, -Instruct, -Math | easy | 1, 2, 8, 32 | 1, 2, 8, 32 | 1, 2, 4, 8, 16, 64, 256 | 1, 2, 4, 8, 16, 64, 256 x b1, b4 |
| Qwen2.5-7B, -Instruct, -Math | hard | 1, 2, 8, 32 | 1, 2, 8, 32 | 1, 2, 4, 8, 16, 64, 256 | 1, 2, 4, 8, 16, 64, 256 x b1, b4 |
| Llama-3.1-8B, Mistral-7B, DeepSeek-7B-Math | hard | 1, 2, 8, 32 | 1, 2, 8, 32 | 1, 2, 4, 8, 16 | 1, 2, 4, 8, 16 x b1, b4 |

`tinylora` is always r=2 tied; b in the turbo column is `--batch`. Every turbo cell also has a `-t1` twin
(`--no-greedy`): the same (batch, prompts, 1 completion) split, but scored on completions sampled at T=1 exactly as
GRPO's rollouts are, instead of greedy ones (`tmp/tools/launch_noisy.sh` submits the twins of every existing cell). Every one of the 24 models in `models.py`
also has an untrained baseline (all six eval tasks at eval@1 and eval@4), including the ones no adapter was
trained on.

Early `tinylora` u1024 extras (1.5B done, 7B partial) were archived out of `outputs/runs` on 2026-09-06; not part of the grid.

Qwen2.5-3B has baselines only: no training run of any adapter was ever launched for it (held back on 2026-09-03 to save GPU hours).

## Layout

```
src/turbolora/
  models.py          MODELS: HF id + raw-text prompt style (Qwen2.5, Llama 3, Ministral, DeepSeek)
  tasks.py           TASKS: SimpleRL-Zoo tiers (easy/medium/hard) for training, gsm8k/math500/aime24/amc23/minerva/olympiad for eval; extract/grade/reward
  adapters.py        adapter attach/export
  grpo.py            shared GRPO/GSPO trainer (Unsloth + TRL), preempt-safe checkpoints
  train_{lora,loraxs,tinylora}.py   GRPO entry points
  bo.py              fixed-noise GP on logit pass rates, trial log, posterior-mean pick
  turbo.py           TuRBO-1 trust-region search on bo.py's GP
  train_bo.py        search objective (TinyLoRA θ = every v concatenated; `--untie` for one v per module): vLLM pass rate on a random train subset
  train_turbolora.py TuRBO entry point (slurm/bo.sh)
  eval.py            greedy vLLM eval of a base model or a trained adapter
slurm/               baseline.sh, train.sh, bo.sh, eval.sh (job setup shared via common.sh, family.sh);
                     resume_sweep.sh resubmits unfinished runs, resume_watch.sh loops it until the sweep is idle
dashboard/           uv run dashboard/serve.py -> live dashboard at localhost:8000 (baselines, runs, curves)
                     uv run dashboard/build.py -> standalone dashboard.html snapshot to share
tests/
collaborators-poc/   original proof-of-concept BO pipeline (kept for reference)
```

Outputs are gitignored. Evals always land in an `eval@K/` dir (K=1 greedy, K=4 sampled) holding the 6 `<task>.json.gz` completion sets and a `summary.json`. The untrained model is the `base` adapter: `outputs/runs/<family>/<model>/base/eval@K/`. Training runs are `outputs/runs/<family>/<model>/<adapter>-<loss>/<cfg>/seed<N>/` with `run.json`, `curves.jsonl` (per-step training metrics, rewritten every step), `final_adapter/` (PEFT export written at startup and refreshed at every snapshot, so a resumed run rebuilds the same SVD bases) and `snapshots/step-N/` at steps 1, 2, 4, … and the last: `trainable.safetensors` plus `eval@1/`; the last snapshot also gets `eval@4/`.

## Usage

```bash
# baseline eval of an untrained model
MODEL=qwen2.5-7b TASKS="gsm8k math500" sbatch slurm/baseline.sh

# GRPO, 3 seeds (array 0-2); CFG suffixes the run dir so configs of one adapter don't collide
MODEL=qwen2.5-7b TASK=hard ADAPTER=tinylora CFG=r2 sbatch slurm/train.sh --rank 2

# single seed, rank sweep
for r in 1 2 8 32; do MODEL=qwen2.5-7b TASK=hard ADAPTER=loraxs CFG=r$r sbatch -a 0 slurm/train.sh --rank $r; done
MODEL=qwen2.5-7b TASK=hard ADAPTER=tinylora CFG=r2-notie sbatch -a 0 slurm/train.sh --rank 2 --untie

# BO, 3 seeds
MODEL=qwen2.5-7b TASK=easy CFG=u1 sbatch slurm/bo.sh --proj-dim 1
MODEL=qwen2.5-7b TASK=easy CFG=u1-notie sbatch slurm/bo.sh --proj-dim 1 --untie
MODEL=qwen2.5-7b TASK=easy CFG=r2-u1-b1-t1 sbatch slurm/bo.sh --proj-dim 1 --batch 1 --no-greedy   # sampled (GRPO T=1) objective

# eval a snapshot (training already evals every snapshot; this is for backfills / extra tasks)
ADAPTERS=outputs/runs/qwen2.5/qwen2.5-7b/tinylora-grpo/r2-u64/seed0/snapshots/step-000393 TASKS="gsm8k math500" sbatch slurm/eval.sh
SAMPLES=4 ADAPTERS=... sbatch slurm/eval.sh   # sampled eval@4 (T=1) instead of greedy eval@1

# resubmit every run whose run.json lacks `steps`, skipping seeds already queued
GO=1 slurm/resume_sweep.sh

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

## Pending: SVD sign consistency of pre-fix requeued runs

The frozen SVD bases (`loraxs`, `tinylora`) are recomputed at every (re)start, and `torch.linalg.svd` picks
singular-vector signs per GPU model. Before the sign pin in `adapters.py` (2026-09-07), a run requeued onto a
different card kept training `R`/`v` against flipped `U`, so its export and its snapshots are in inconsistent
spaces. The 2026-09-07 audit classified 109 runs clean, 49 unclean and 22 `loraxs`
undecidable. Unclean `tinylora` runs are kept: the turbo twin, the export and the eval all pin to the same
`lora_B`, so they stay consistent.

All seven cards have now been SVD'd (one job per `--constraint`), and there are no
sign families beyond A100-PCIE-40G = A100-SXM4-80G: every other pair of cards flips 6-13 of the 84 measured
singular vectors, though the factors themselves agree to ~2e-3. So an undecidable run is clean only if every
one of its instances ran on the same card model. To finish:

1. Re-measure A16 — its `.pt` predates `svd_signs.py` (196 matrices of one model, not layer 0 of the six) and
   `compare.py` chokes on the mismatched keys, so it is excluded from the comparison above.
2. For each undecidable run, map its requeue history (node -> card, from `sacct -j <id> -o JobID,NodeList,Start`)
   and mark it clean if every instance ran on one card model, otherwise unclean. Update the csv and the memory note.
