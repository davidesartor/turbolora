# LlMBO

Code for the experiments: TinyLoRA adapters on math reasoning, trained either by GRPO or by LlMBO, gradient-free
Bayesian optimization (TuRBO-1) over their few trainable scalars, with LoRA and LoRA-XS as GRPO baselines.

In the code LlMBO is the `turbo` path (`turbo.py`, `train_turbolora.py`, `slurm/train_turbo.sh`, run dirs
`tinylora-turbo`), and the Python package is `turbolora`.

## Setup

Requirements: Linux x86_64, Python 3.12, [uv](https://docs.astral.sh/uv/), one CUDA GPU per run
(≥24 GB for 1.5B models, ≥48 GB for 7–8B), ~60 GB host RAM. `uv.lock` pins every dependency:

```bash
uv sync
uv run pytest        # CPU-only unit tests (unsloth is stubbed)
```

Models and data come from the Hugging Face Hub (Llama and Mistral are gated: accept their licenses and
`uv run hf auth login`). The SLURM scripts run offline against `HF_HOME=.hf-cache`, so fill the cache once on a
machine with network access:

```bash
export HF_HOME=$PWD/.hf-cache
uv run python -c "from turbolora.tasks import TASKS; [TASKS[t]('train') for t in ('easy', 'hard')]; [TASKS[t]() for t in ('gsm8k', 'math500', 'aime24', 'amc23', 'minerva', 'olympiad')]"
uv run hf download Qwen/Qwen2.5-7B   # once per model; HF ids in src/turbolora/models.py
```

The SLURM headers (`-p`, `--constraint`, `module load cuda/...`) need adapting to your cluster.

## Code structure

```
src/turbolora/
  models.py           MODELS: short name -> HF id + raw-text prompt style (SimpleRL-Zoo prompts)
  tasks.py            TASKS: SimpleRL-Zoo training tiers (easy/medium/hard), single-source gsm8k/math train sets,
                      eval benchmarks gsm8k/math500/aime24/amc23/minerva/olympiad; answer extraction, grading (math-verify), reward
  adapters.py         lora / loraxs / tinylora attach and export as a standard PEFT LoRA dir (vLLM loads it unchanged)
  grpo.py             shared GRPO trainer (Unsloth + TRL, colocated vLLM rollouts), checkpoint/resume, snapshot evals
  train_lora.py       GRPO entry points, one per adapter
  train_loraxs.py
  train_tinylora.py
  bo.py               fixed-noise GP on logit pass rates, trial log, posterior-mean pick
  turbo.py            TuRBO-1 trust-region search on bo.py's GP
  train_bo.py         BO objective: pass rate of a TinyLoRA θ on a random train subset, scored with vLLM
  train_turbolora.py  LlMBO entry point
  eval.py             vLLM eval of a base model or trained adapter (greedy, or --samples K at T=1)
slurm/
  train_{lora,loraxs,tinylora,turbo}.sh   one 3-seed array job per adapter config
  eval_{baseline,tasks,train}.sh          standalone evals (training already evals every snapshot)
  sweep.py                                status / submission of the full grid below
  watch.sh                                optional watcher that resubmits unfinished seeds
dashboard/            local results browser (uv run dashboard/serve.py)
tests/
```

Adapters (`--rank r`, TinyLoRA `--proj-dim u`), with ΔW per adapted projection and trainable parameter count:

| adapter | ΔW | trainable params |
|---|---|---|
| `lora` | BA, rank r | Σ r (d_in + d_out) |
| `loraxs` | UΣ R Vᵀ, R ∈ ℝ^{r×r}, frozen truncated SVD of W | r² per module |
| `tinylora` | UΣ (Σᵢ vᵢPᵢ) Vᵀ, fixed random Pᵢ, v ∈ ℝᵘ | u, one v shared by all modules (`--untie`: one per module) |

Every hyperparameter default in the entry points' argparse is the setting used in the experiments; each run
records its resolved config in `run.json`.

## Reproducing the experiments

```bash
# untrained baselines, at the same 1024-token budget as every trained eval (greedy, then 4 samples at T=1)
uv run -m turbolora.eval --model qwen2.5-7b --tasks gsm8k math500 aime24 amc23 minerva olympiad --max-tokens 1024
uv run -m turbolora.eval --model qwen2.5-7b --tasks gsm8k math500 aime24 amc23 minerva olympiad --max-tokens 1024 --samples 4

# GRPO, seeds 0-2 as an array; CFG names the run dir and sets the adapter size
MODEL=qwen2.5-7b TASK=hard CFG=r32 sbatch slurm/train_lora.sh         # --rank 32
MODEL=qwen2.5-7b TASK=hard CFG=r8 sbatch slurm/train_loraxs.sh        # --rank 8
MODEL=qwen2.5-7b TASK=hard CFG=r2-u8 sbatch slurm/train_tinylora.sh   # --rank 2 --proj-dim 8

# LlMBO on the same TinyLoRA space: CFG=r<rank>-u<u>-b<batch>[-t1]; needs the tinylora-grpo run with the same
# model/u/seed (its export fixes the SVD bases); default objective = greedy pass rate, -t1 = sampled at T=1
MODEL=qwen2.5-7b TASK=hard CFG=r2-u8-b1-t1 sbatch slurm/train_turbo.sh

# the full grid: coverage table, then the missing sbatch lines (dry run; --go submits)
uv run slurm/sweep.py status
uv run slurm/sweep.py submit [--go]
```

Grid run by `sweep.py` (`tinylora` is always r=2 with one tied v; b = LlMBO `--batch`):

| models | train tier | `lora` r | `loraxs` r | `tinylora` GRPO u | `tinylora` LlMBO u |
|---|---|---|---|---|---|
| Qwen2.5-1.5B, -Instruct, -Math | easy | 1, 2, 8, 32 | 1, 2, 8, 32 | 1, 2, 4, 8, 16, 64, 256 | 1, 2, 4, 8, 16, 64, 256 x b1, b4 |
| Qwen2.5-7B, -Instruct, -Math | hard | 1, 2, 8, 32 | 1, 2, 8, 32 | 1, 2, 4, 8, 16, 64, 256 | 1, 2, 4, 8, 16, 64, 256 x b1, b4 |
| Llama-3.1-8B, Mistral-7B, DeepSeek-7B-Math | hard | 1, 2, 8, 32 | 1, 2, 8, 32 | 1, 2, 4, 8, 16 | 1, 2, 4, 8, 16 x b1, b4 |

Every LlMBO cell runs with both objectives (greedy and `-t1`). Seeds 0-2 everywhere, 0-7 for `tinylora` GRPO
u ∈ {1, 2, 4} and their b4 `-t1` LlMBO twins.

Jobs are preemptible: training checkpoints on SIGTERM/SIGUSR1 and resumes from the last checkpoint (LlMBO from
`trials.json`) when requeued.

## Outputs

Runs land in `outputs/runs/<family>/<model>/<adapter>-<method>/<cfg>/seed<N>/`; untrained baselines in
`outputs/runs/<family>/<model>/base/`.

- `run.json`: resolved config, stamped with `steps` when the run finishes
- `curves.jsonl`: per-step training metrics
- `trials.json` (LlMBO): every evaluated θ and its pass rate
- `final_adapter/`: PEFT LoRA export of the final adapter
- `snapshots/step-N/` at steps 1, 2, 4, … and the last: `trainable.safetensors` and `eval@1/` (greedy); the last
  snapshot also has `eval@4/` (4 samples at T=1)
- `eval@K/<task>.json.gz` holds every completion and its grade, `eval@K/summary.json` per-task `accuracy` (mean over all K completions) and, for K>1, `pass_any` (pass@K)

## References

- TinyLoRA: arXiv 2602.04118
- LoRA-XS: arXiv 2405.17604
- SimpleRL-Zoo data tiers and prompts: arXiv 2503.18892
- TuRBO: arXiv 1910.01739
