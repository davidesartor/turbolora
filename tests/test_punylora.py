"""CPU tests: the Binomial-likelihood GP and `puny_lora.search` on a toy objective; `train_punylora.run` with unsloth (conftest), vLLM and CUDA faked."""

import importlib.machinery
import json
import os
import signal
import sys
import types
from pathlib import Path

import pytest
import torch
from datasets import Dataset

vllm = types.ModuleType("vllm")
vllm.__spec__ = importlib.machinery.ModuleSpec("vllm", None)


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.n = 1
        self.__dict__.update(kwargs)


vllm.SamplingParams = FakeSamplingParams  # type: ignore[attr-defined]
sys.modules["vllm"] = vllm

import unsloth  # noqa: E402  conftest stub
from turbolora import puny_lora, train_punylora  # noqa: E402
from turbolora.models import MODELS  # noqa: E402

SMALL = ["--theta-range", "0.1", "--batch", "4", "--n-baseline", "4", "--n-sobol", "4", "--n-evals", "4", "--thompson-candidates", "64", "--inducing", "16", "--fit-steps", "50"]


def bernoulli_counts(p, n, seed):
    g = torch.Generator().manual_seed(seed)
    return int(torch.binomial(torch.tensor(float(n)), torch.tensor(float(p)), generator=g).item())


def test_binomial_gp_recovers_a_bump_from_single_completions():
    """n=1 per θ: 200 Bernoulli draws around a bump at θ=0.03 are enough for the posterior mean to peak near it."""
    torch.manual_seed(0)
    X = torch.rand(200, 1, dtype=torch.float64) * 0.2 - 0.1
    p = torch.sigmoid(1.0 - 400 * (X - 0.03) ** 2)
    hits = torch.bernoulli(p)
    trials = [dict(theta=x.tolist(), hits=int(h), n=1) for x, h in zip(X, hits)]
    gp = puny_lora.fit_gp(trials, theta_range=0.1, n_inducing=32, steps=200)

    grid = torch.linspace(-1, 1, 201, dtype=torch.float64)[:, None]
    with torch.no_grad():
        post = gp(grid)
    assert abs(grid[post.mean.argmax()].item() * 0.1 - 0.03) < 0.02
    assert post.variance[0] > post.variance[100]  # the edges are less certain than the middle


def test_counts_pools_repeated_thetas():
    trials = [dict(theta=[0.0], hits=1, n=2), dict(theta=[0.0], hits=0, n=2), dict(theta=[0.1], hits=2, n=2)]
    thetas, hits, n = puny_lora.counts(trials)
    assert thetas == [[0.0], [0.1]] and hits.tolist() == [1.0, 2.0] and n.tolist() == [4.0, 2.0]


def test_search_defaults():
    args = puny_lora.argument_parser().parse_args([])
    assert (args.seed, args.theta_range, args.batch) == (0, None, 1)
    assert (args.n_baseline, args.n_sobol, args.n_evals, args.thompson_candidates) == (16, 16, None, 2048)


def bump(thetas, batch):
    """Pass rate peaks at (0.05, -0.05); 64 draws per θ so the pick lands near the peak."""
    p = torch.sigmoid(2.0 - 400 * ((thetas - torch.tensor([0.05, -0.05])) ** 2).sum(-1))
    return [(bernoulli_counts(pi, 64, batch * 100 + i), 64) for i, pi in enumerate(p.tolist())]


def test_search_baseline_sobol_thompson_then_best_posterior(tmp_path):
    seen = []
    args = puny_lora.argument_parser().parse_args(SMALL)
    chosen = puny_lora.search(args, lambda t, b: seen.append((t.tolist(), b)) or bump(t, b), dim=2, out=tmp_path)

    trials = json.loads((tmp_path / "trials.json").read_text())
    assert [t["batch"] for t in trials] == [b for b in range(6) for _ in range(4)]
    assert [t["trial"] for t in trials] == list(range(24))
    assert [t["baseline"] for t in trials] == [True] * 4 + [False] * 20
    assert all(t["theta"] == [0.0, 0.0] for t in trials[:4])
    assert all(-0.1 <= x <= 0.1 for t in trials[4:] for x in t["theta"])
    assert [b for _, b in seen] == list(range(6))
    assert all(t["n"] == 64 and 0 <= t["hits"] <= 64 for t in trials)

    assert chosen["steps"] == 4 and chosen["theta"] in [t["theta"] for t in trials]
    assert 0 < chosen["posterior"] < 1
    baseline = [t for t in trials if t["baseline"]]
    assert chosen["baseline"] == pytest.approx(sum(t["hits"] for t in baseline) / 256)
    assert torch.tensor(chosen["theta"]).norm() < 0.2  # in range, and the pick sits on the higher side of the bump
    assert chosen["posterior"] >= chosen["baseline"] - 0.1


def test_search_resumes_from_trials_json(tmp_path):
    args = puny_lora.argument_parser().parse_args(SMALL)
    calls = []

    def objective(t, b):
        calls.append(b)
        if b == 3:
            os.kill(os.getpid(), signal.SIGUSR1)
        return bump(t, b)

    assert puny_lora.search(args, objective, 2, tmp_path) is None
    assert calls == [0, 1, 2, 3]
    assert len(json.loads((tmp_path / "trials.json").read_text())) == 16

    calls.clear()
    chosen = puny_lora.search(args, objective, 2, tmp_path)
    assert calls == [4, 5] and chosen is not None
    assert len(json.loads((tmp_path / "trials.json").read_text())) == 24


def test_search_snapshots_pick_at_powers_of_two(tmp_path):
    args = puny_lora.argument_parser().parse_args([*SMALL, "--n-evals", "5"])
    steps = []
    puny_lora.search(args, bump, 2, tmp_path, on_snapshot=lambda step, theta: steps.append((step, len(theta))))
    assert steps == [(1, 2), (2, 2), (4, 2), (5, 2)]


DATASET = Dataset.from_dict({"question": ["1+1?", "4-2?", "2+2?", "3+3?"], "answer": ["2", "2", "4", "6"]})


class FakeModel(torch.nn.Module):
    """Unsloth's fast_inference surface: `load_lora` snapshots the live θ under a fresh id, `fast_generate` answers per prompt under its LoRA."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5, bias=False)
        self.v = torch.nn.Parameter(torch.zeros(2))
        self.calls: list[dict] = []
        self.loras: dict[int, list[float]] = {}

    def load_lora(self, save_directory, load_tensors=False):
        assert load_tensors
        lora_id = len(self.loras) + 1
        self.loras[lora_id] = self.v.tolist()
        return types.SimpleNamespace(id=lora_id, path=save_directory)

    def fast_generate(self, prompts, sampling_params, lora_request, use_tqdm):
        requests = lora_request if isinstance(lora_request, list) else [lora_request] * len(prompts)
        assert len(requests) == len(prompts)
        self.calls.append(dict(prompts=prompts, sampling=sampling_params, lora_ids=[r.id for r in requests]))
        # a rollout is right iff the question is one of the "2" ones -> per-question pass rate 0 or 1
        return [
            types.SimpleNamespace(outputs=[types.SimpleNamespace(text=f"\\boxed{{{2 if '2?' in p else 0}}}")] * sampling_params.n)
            for p in prompts
        ]


class FakeAdapter:
    calls: dict = {}
    exports: list[dict] = []

    @staticmethod
    def attach(model, rank, seed, **kwargs):
        FakeAdapter.calls = dict(rank=rank, seed=seed, **kwargs)
        model.linear.weight.requires_grad_(False)
        return model

    @staticmethod
    def export(model, out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "adapter_config.json").write_text("{}")
        FakeAdapter.exports.append(dict(dir=out_dir, theta=model.v.tolist()))


class FakeTokenizer:
    def save_pretrained(self, out_dir):
        (Path(out_dir) / "tokenizer.json").write_text("{}")


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    model = FakeModel()
    loads: dict = {}

    def from_pretrained(**kwargs):
        loads.update(kwargs)
        return model, FakeTokenizer()

    monkeypatch.setattr(unsloth.FastLanguageModel, "from_pretrained", from_pretrained)
    monkeypatch.setitem(train_punylora.TASKS, "gsm8k", lambda split: DATASET)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: types.SimpleNamespace(total_memory=24 * 2**30))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 3 * 2**30)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "FakeGPU")
    monkeypatch.chdir(tmp_path)
    FakeAdapter.exports.clear()
    return model, loads


def parse(*extra: str, out: str):
    base = ["--model", "qwen2.5-7b", "--task", "gsm8k", "--out", out, "--n-questions", "8", "--k-rollouts", "1", "--eval-tasks", "gsm8k"]
    return train_punylora.argument_parser().parse_args([*base, *SMALL, *extra])


def test_argument_parser_defaults():
    args = train_punylora.argument_parser().parse_args(["--model", "qwen2.5-7b", "--task", "gsm8k", "--out", "o"])
    assert (args.n_questions, args.k_rollouts, args.batch, args.vllm_share) == (16, 1, 1, 0.85)
    assert (args.n_evals, args.theta_range, args.max_completion, args.rank, args.proj_dim) == (None, None, 1024, 2, 1)


def test_run_resolves_theta_range_and_n_evals_from_dataset(stubbed, tmp_path):
    args = parse(out=str(tmp_path / "run"))
    args.theta_range = None
    args.n_evals = None
    train_punylora.run(args, FakeAdapter)
    # 4 prompts -> 1 step/epoch -> 3 · 5e-4; 3·4·4 = 48 completions / (8 questions per call · 1 rollout) = 6 batches
    assert args.theta_range == pytest.approx(1.5e-3)
    assert args.n_evals == 6
    assert json.loads((tmp_path / "run" / "run.json").read_text())["steps"] == 6


def test_run_wires_model_adapter_objective_and_outputs(stubbed, tmp_path):
    model, loads = stubbed
    out = tmp_path / "run"
    train_punylora.run(parse(out=str(out)), FakeAdapter)
    spec = MODELS["qwen2.5-7b"]

    # same load as grpo.run; vLLM owns the card and serves one LoRA per θ in the batch
    assert loads["model_name"] == spec.hf_id
    assert loads["max_lora_rank"] == 8 and loads["fast_inference"]
    assert (loads["gpu_memory_utilization"], loads["max_loras"]) == (0.85, 4)
    assert FakeAdapter.calls == dict(rank=2, seed=0, proj_dim=1, tie=0)

    # one vLLM call per batch, batch·n_questions prompts, each prompt under the LoRA holding its θ
    trials = json.loads((out / "trials.json").read_text())
    assert len(trials) == 24 and all(t["n"] == 2 for t in trials)
    rollouts = [c for c in model.calls if c["sampling"].temperature == 1.0]
    assert len(rollouts) == 6 and all(len(c["prompts"]) == 8 for c in rollouts)
    for call, batch in zip(rollouts, range(6)):
        rows = [t for t in trials if t["batch"] == batch]
        ids = call["lora_ids"]
        assert ids == [i for i in ids[::2] for _ in range(2)] and len(set(ids)) == 4
        assert torch.allclose(torch.tensor([model.loras[i] for i in ids[::2]]), torch.tensor([t["theta"] for t in rows]))
        assert call["sampling"].seed == batch and call["sampling"].stop == list(spec.prompt.stop)
    assert all(p.startswith("<|im_start|>system") for p in rollouts[0]["prompts"])
    assert model.v.dtype == torch.float32 and not any(p.requires_grad for p in model.parameters())

    # snapshots of the GP's pick after GP-guided batches 1, 2 and 4 (the last); export only at the last
    snapshots = sorted(d.name for d in (out / "snapshots").iterdir())
    assert snapshots == ["step-000001", "step-000002", "step-000004"]
    evals = [c for c in model.calls if c["sampling"].temperature == 0.0]
    assert all(len(c["prompts"]) == len(DATASET) for c in evals) and 1 <= len(evals) <= 3
    for name in snapshots:
        assert (out / "snapshots" / name / "trainable.safetensors").exists()
        assert json.loads((out / "snapshots" / name / "eval.json").read_text())["gsm8k"]["n"] == len(DATASET)
    assert [e["dir"] for e in FakeAdapter.exports] == [str(out / "snapshots" / "step-000004"), str(out / "final_adapter")]

    summary = json.loads((out / "run.json").read_text())
    assert summary["theta"] in [t["theta"] for t in trials]
    assert torch.allclose(torch.tensor(FakeAdapter.exports[-1]["theta"]), torch.tensor(summary["theta"]))
    assert (out / "final_adapter" / "tokenizer.json").exists()
    assert (summary["adapter"], summary["loss"], summary["steps"], summary["params"], summary["gpu"]) == ("fakeadapter", "puny_lora", 4, 2, "FakeGPU")
    assert {"baseline", "baseline_sem", "posterior", "theta_range"} <= summary.keys() and 0 < summary["posterior"] < 1
    assert (summary["rank"], summary["proj_dim"], summary["tie"], summary["peak_vram_gb"]) == (2, 1, 0, 3.0)


def test_run_skips_export_when_search_is_interrupted(stubbed, tmp_path, monkeypatch):
    out = tmp_path / "run"
    monkeypatch.setattr(puny_lora, "search", lambda *a, **k: None)
    train_punylora.run(parse(out=str(out)), FakeAdapter)
    # the config half of run.json is out for the dashboard, the summary half is not
    config = json.loads((out / "run.json").read_text())
    assert "steps" not in config and config["max_steps"] == 4 and config["design"] == 2
    assert not (out / "final_adapter").exists()


def test_main_parses_adapter_args(monkeypatch):
    seen = {}
    monkeypatch.setattr(train_punylora, "run", lambda args: seen.update(vars(args)))
    monkeypatch.setattr(sys, "argv", ["train_punylora", "--model", "qwen2.5-7b", "--task", "gsm8k", "--out", "o", "--proj-dim", "3", "--untie"])
    train_punylora.main()
    assert (seen["proj_dim"], seen["untie"]) == (3, True)
