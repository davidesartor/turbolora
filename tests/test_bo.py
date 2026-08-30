"""CPU tests: the GP and `bo.search` on a toy objective; `train_turbolora.run` with unsloth (conftest), vLLM and CUDA faked."""

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
        self.__dict__.update(kwargs)


vllm.SamplingParams = FakeSamplingParams  # type: ignore[attr-defined]
sys.modules["vllm"] = vllm

import unsloth  # noqa: E402  conftest stub
from turbolora import bo, train_turbolora  # noqa: E402
from turbolora.models import MODELS  # noqa: E402

SMALL = ["--n-baseline", "2", "--n-sobol", "2", "--n-evals", "4", "--thompson-candidates", "50"]


def test_heteroskedastic_gp_denoises_toward_the_optimum():
    torch.manual_seed(0)
    X = torch.linspace(-0.1, 0.1, 25, dtype=torch.float64)[:, None]
    truth = 0.8 - 20 * (X - 0.03) ** 2
    Y = truth + 0.02 * torch.randn_like(truth)
    gp = bo.HeteroskedasticGP(X, Y, torch.full_like(Y, 0.02**2), n_samples=32)

    grid = torch.linspace(-0.1, 0.1, 201, dtype=torch.float64)[:, None]
    with torch.no_grad():
        posterior = gp.posterior(grid)
    assert abs(grid[posterior.mean.argmax()].item() - 0.03) < 0.01
    assert posterior.sample().shape[-2:] == (201, 1)
    with torch.no_grad():
        assert (gp.posterior(X).mean - truth).abs().max() < 0.03


def test_fit_gp_reads_trials():
    trials = [dict(theta=[x, -x], value=0.5 + x, sem=0.02) for x in torch.linspace(-0.1, 0.1, 8).tolist()]
    gp = bo.fit_gp(trials, n_samples=32)
    with torch.no_grad():
        mean = gp.posterior(torch.tensor([[0.1, -0.1], [-0.1, 0.1]], dtype=torch.float64)).mean
    assert mean[0] > mean[1]


def test_search_defaults():
    args = bo.argument_parser().parse_args([])
    assert (args.seed, args.theta_min, args.theta_max) == (0, -0.1, 0.1)
    assert (args.n_baseline, args.n_sobol, args.n_evals, args.thompson_candidates) == (10, 10, 600, 2000)


def quadratic(theta, trial):
    """Peak at (0.05, -0.05); signal ≫ noise so the GP's pick is the best observed."""
    return 1.0 - 100 * ((theta - torch.tensor([0.05, -0.05])) ** 2).sum().item(), 0.01


def test_search_baseline_sobol_thompson_then_best_posterior(tmp_path):
    seen = []
    args = bo.argument_parser().parse_args([*SMALL, "--n-evals", "8"])
    chosen = bo.search(args, lambda t, i: seen.append((t.tolist(), i)) or quadratic(t, i), dim=2, n_samples=32, out=tmp_path)

    trials = json.loads((tmp_path / "trials.json").read_text())
    assert [(t["theta"], t["trial"]) for t in trials] == seen
    assert [t["baseline"] for t in trials] == [True] * 2 + [False] * 8
    assert all(t["theta"] == [0.0, 0.0] for t in trials[:2])
    assert all(-0.1 <= x <= 0.1 for t in trials[2:] for x in t["theta"])
    assert trials[2]["theta"] != trials[3]["theta"]
    assert chosen is not None and chosen["steps"] == 10
    assert chosen["theta"] == max(trials[2:], key=lambda t: t["value"])["theta"]


def test_search_resumes_from_trials_json(tmp_path):
    done = [
        dict(trial=0, baseline=True, theta=[0.0, 0.0], value=0.5, sem=0.02),
        dict(trial=1, baseline=True, theta=[0.0, 0.0], value=0.6, sem=0.02),
        dict(trial=2, baseline=False, theta=[0.07, -0.01], value=0.7, sem=0.02),
    ]
    (tmp_path / "trials.json").write_text(json.dumps(done))
    seen = []
    bo.search(bo.argument_parser().parse_args(SMALL), lambda t, i: seen.append(i) or quadratic(t, i), 2, 32, tmp_path)
    trials = json.loads((tmp_path / "trials.json").read_text())
    assert trials[:3] == done and len(trials) == 6
    assert seen == [3, 4, 5]


def test_search_stops_after_current_trial_on_signal(tmp_path):
    def objective(theta, trial):
        if trial == 1:
            os.kill(os.getpid(), signal.SIGUSR1)
        return quadratic(theta, trial)

    assert bo.search(bo.argument_parser().parse_args(SMALL), objective, 2, 32, tmp_path) is None
    assert len(json.loads((tmp_path / "trials.json").read_text())) == 2


DATASET = Dataset.from_dict({"question": ["1+1?", "4-2?", "2+2?", "3+3?"], "answer": ["2", "2", "4", "6"]})


class FakeModel(torch.nn.Module):
    """Unsloth's fast_inference model surface: `load_lora` hands out fresh ids, `fast_generate` answers per prompt."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5, bias=False)
        self.v = torch.nn.Parameter(torch.zeros(2))
        self.calls: list[dict] = []
        self.next_lora_id = 1

    def load_lora(self, save_directory):
        assert (Path(save_directory) / "adapter_config.json").exists()
        self.next_lora_id += 1
        return types.SimpleNamespace(id=self.next_lora_id - 1, path=save_directory)

    def fast_generate(self, prompts, sampling_params, lora_request, use_tqdm):
        self.calls.append(dict(prompts=prompts, sampling=sampling_params, lora_id=lora_request.id, theta=self.v.tolist()))
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
        model.linear.weight.requires_grad_(False)  # only `v` stays trainable -> θ ∈ ℝ²
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
    monkeypatch.setitem(train_turbolora.TASKS, "gsm8k", lambda split: DATASET)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: types.SimpleNamespace(total_memory=80 * 2**30))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 3 * 2**30)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "FakeGPU")
    monkeypatch.chdir(tmp_path)
    FakeAdapter.exports.clear()
    return model, loads


def parse(*extra: str, out: str):
    base = ["--model", "qwen2.5-7b", "--task", "gsm8k", "--out", out, "--n-questions", "3", "--k-rollouts", "2"]
    return train_turbolora.argument_parser().parse_args([*base, *SMALL, *extra])


def test_argument_parser_defaults():
    args = train_turbolora.argument_parser().parse_args(["--model", "qwen2.5-7b", "--task", "gsm8k", "--out", "o"])
    assert (args.n_questions, args.k_rollouts, args.temperature, args.top_p, args.sem_floor) == (32, 4, 0.7, 0.95, 0.01)
    assert (args.n_evals, args.max_completion) == (600, 1024)


def test_run_wires_model_adapter_objective_and_outputs(stubbed, tmp_path):
    model, loads = stubbed
    out = tmp_path / "run"
    train_turbolora.run(parse(out=str(out)), FakeAdapter, rank=2, proj_dim=1, tie=7)
    spec = MODELS["qwen2.5-7b"]

    # same load as grpo.run, adapter attached with its kwargs
    assert loads["model_name"] == spec.hf_id
    assert loads["max_seq_length"] == 512 + 1024
    assert loads["max_lora_rank"] == 8 and loads["fast_inference"]
    assert loads["gpu_memory_utilization"] == 0.45
    assert FakeAdapter.calls == dict(rank=2, seed=0, proj_dim=1, tie=7)

    # each trial's θ lands in the model (as float32) before export; one fresh vLLM adapter id per trial from the candidate dir
    trials = json.loads((out / "trials.json").read_text())
    assert len(trials) == 6
    assert torch.allclose(torch.tensor([c["theta"] for c in model.calls]), torch.tensor([t["theta"] for t in trials]))
    assert model.v.dtype == torch.float32
    assert [c["lora_id"] for c in model.calls] == [1, 2, 3, 4, 5, 6]
    assert [e["dir"] for e in FakeAdapter.exports[:6]] == [str(out / "candidate")] * 6
    assert all(t["sem"] >= 0.01 for t in trials)

    # objective: n questions x K rollouts, raw-text prompts with the model's stop strings
    assert all(len(c["prompts"]) == 3 and c["sampling"].n == 2 for c in model.calls)
    assert model.calls[0]["sampling"].stop == list(spec.prompt.stop)
    assert all(p.startswith("<|im_start|>system") for p in model.calls[0]["prompts"])

    # export: final_adapter written from the chosen (searched) θ, run.json for the dashboard
    summary = json.loads((out / "run.json").read_text())
    assert summary["theta"] in [t["theta"] for t in trials[2:]]
    assert FakeAdapter.exports[-1]["dir"] == str(out / "final_adapter")
    assert torch.allclose(torch.tensor(FakeAdapter.exports[-1]["theta"]), torch.tensor(summary["theta"]))
    assert (out / "final_adapter" / "tokenizer.json").exists()
    assert (summary["adapter"], summary["loss"], summary["steps"], summary["params"], summary["gpu"]) == ("fakeadapter", "bo", 6, 2, "FakeGPU")
    assert (summary["rank"], summary["proj_dim"], summary["tie"], summary["peak_vram_gb"]) == (2, 1, 7, 3.0)


def test_run_skips_export_when_search_is_interrupted(stubbed, tmp_path, monkeypatch):
    out = tmp_path / "run"
    monkeypatch.setattr(bo, "search", lambda *a, **k: None)
    train_turbolora.run(parse(out=str(out)), FakeAdapter, rank=2)
    assert not (out / "run.json").exists() and not (out / "final_adapter").exists()


def test_main_forwards_adapter_args(monkeypatch):
    seen = {}
    monkeypatch.setattr(train_turbolora, "run", lambda args, adapter, **kw: seen.update(adapter=adapter.__name__, **kw))
    monkeypatch.setattr(sys, "argv", ["train_turbolora", "--model", "qwen2.5-7b", "--task", "gsm8k", "--out", "o", "--proj-dim", "3", "--tie", "7"])
    train_turbolora.main()
    assert seen == dict(adapter="TurboLoRA", rank=2, proj_dim=3, tie=7)
