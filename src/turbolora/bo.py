"""Gradient-free search: θ=0 replicates → Sobol → fixed-noise GP → batched Thompson sampling over a boxed vector, with a resumable trial log (after collaborators-poc)."""

import argparse
import json
import signal
from pathlib import Path
from typing import Callable
from jaxtyping import Float
from torch import Tensor

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

Objective = Callable[
    [Float[Tensor, "B T"], int], list[tuple[float, float]]
]  # (θ batch, batch index) -> (mean, SEM) per row


def fit_gp(trials: list[dict]) -> SingleTaskGP:
    """GP on the objective with each trial's sem² as observation noise plus a learned noise floor (Matérn 5/2 ARD, normalized θ, standardized value)."""
    X = torch.tensor([t["theta"] for t in trials], dtype=torch.float64)
    Y = torch.tensor([[t["value"]] for t in trials], dtype=torch.float64)
    Yvar = torch.tensor([[t["sem"] ** 2] for t in trials], dtype=torch.float64)
    gp = SingleTaskGP(
        X,
        Y,
        Yvar,
        covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=X.shape[-1])),
        input_transform=Normalize(d=X.shape[-1]),
        outcome_transform=Standardize(m=1),
    )
    # the floor absorbs misfit beyond the binomial sem; the model holds the sem² already in standardized units
    gp.likelihood = FixedNoiseGaussianLikelihood(noise=gp.likelihood.noise.detach(), learn_additional_noise=True).to(X)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return gp.eval()


def argument_parser() -> argparse.ArgumentParser:
    """Search arguments (defaults = collaborators' pipeline_config.sh); the objective's own get added on top."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--theta-range",
        type=float,
        default=None,
        help="search box [-range, range]^dim; None = objective-specific default",
    )
    parser.add_argument(
        "--n-baseline",
        type=int,
        default=8,
        help="θ=0 replicates: a well-measured base point that also seeds the noise model",
    )
    parser.add_argument(
        "--n-sobol",
        type=int,
        default=8,
        help="quasi-random trials before the GP takes over",
    )
    parser.add_argument(
        "--n-evals",
        type=int,
        default=None,
        help="GP-guided trials, on top of the θ=0 replicates and the Sobol design; None = objective-specific default",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="θ's evaluated per objective call (one vLLM batch, one LoRA each); 1 = sequential",
    )
    parser.add_argument("--thompson-candidates", type=int, default=2048)
    return parser


def pick(trials: list[dict]) -> tuple[dict, float]:
    """Denoised best-observed: the evaluated trial with the highest GP posterior mean (θ=0 competes like any other point)."""
    if len({tuple(t["theta"]) for t in trials}) < 2:
        return trials[0], trials[0]["value"]
    gp = fit_gp(trials)
    with torch.no_grad():
        means = gp.posterior(
            torch.tensor([t["theta"] for t in trials], dtype=torch.float64)
        ).mean.squeeze(-1)
    return trials[int(means.argmax())], means.max().item()


def search(
    args: argparse.Namespace,
    objective: Objective,
    dim: int,
    out: Path,
    on_snapshot: Callable[[int, dict], None] | None = None,
) -> dict | None:
    """Maximize `objective` over [-θ_range, θ_range]^dim, logging to `<out>/trials.json`; returns the evaluated trial with the best GP posterior mean.

    Batches: the initial design (θ=0 replicates, then Sobol) in chunks of `batch`, then `n_evals` Thompson batches (one
    posterior draw per row). Each trial's sem² is taken as its observation noise. Resumes from the log; a SIGUSR1/SIGTERM
    finishes the running batch and returns None. `on_snapshot(step, pick)` fires after GP-guided batch 1, 2, 4, ... and
    the last with the current pick.
    """
    trials_path = out / "trials.json"
    trials: list[dict] = (
        json.loads(trials_path.read_text()) if trials_path.exists() else []
    )
    if args.theta_range is None or args.n_evals is None:
        raise ValueError(
            "theta_range and n_evals must be resolved by the caller before search"
        )
    bounds = torch.tensor(
        [[-args.theta_range] * dim, [args.theta_range] * dim], dtype=torch.float64
    )
    # a fixed design, so a resumed run continues it
    sobol = SobolEngine(dim, scramble=True, seed=args.seed).draw(args.n_sobol).double()
    sobol = bounds[0] + (bounds[1] - bounds[0]) * sobol
    design = torch.cat([torch.zeros(args.n_baseline, dim, dtype=torch.float64), sobol])
    design_batches = [design[i : i + args.batch] for i in range(0, len(design), args.batch)]
    n_batches = len(design_batches) + args.n_evals

    # slurm preemption/wall-limit signal: finish the running batch, then leave the loop
    stop = argparse.Namespace(requested=False)
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, lambda *_: setattr(stop, "requested", True))
    # logs predating the batch axis were sequential: their trial index is their batch index
    while (b := trials[-1].get("batch", trials[-1]["trial"]) + 1 if trials else 0) < n_batches and not stop.requested:
        if b < len(design_batches):
            thetas = design_batches[b]
        else:
            # Thompson sampling: one posterior draw per row over uniform random candidates, each row's argmax is its θ
            gp = fit_gp(trials)
            candidates = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(
                args.thompson_candidates, dim, dtype=torch.float64
            )
            with torch.no_grad():
                draws = gp.posterior(candidates).rsample(torch.Size([args.batch])).squeeze(-1)
            thetas = candidates[draws.argmax(-1)]
        results = objective(thetas, b)
        for theta, (value, sem) in zip(thetas, results):
            trials.append(
                dict(
                    trial=len(trials),
                    batch=b,
                    baseline=b < len(design_batches) and bool((theta == 0).all()),
                    theta=theta.tolist(),
                    value=value,
                    sem=sem,
                )
            )
        trials_path.write_text(json.dumps(trials, indent=1))
        best = max(t["value"] for t in trials)
        batch_best = max(v for v, _ in results)
        print(f"batch {b}: best {batch_best:.4f} over {len(results)} θ's (best so far {best:.4f})")
        # snapshot steps count GP-guided batches only: the θ=0 replicates and the Sobol design are the initial design
        step = b + 1 - len(design_batches)
        if on_snapshot and step >= 1 and (not step & (step - 1) or step == args.n_evals):
            on_snapshot(step, pick(trials)[0])
    if stop.requested:
        print("stopped on signal; rerun to resume")
        return None

    chosen, posterior_mean = pick(trials)
    baseline = torch.tensor([t["value"] for t in trials if t["baseline"]])
    baseline_sem = baseline.std(unbiased=len(baseline) > 1) / len(baseline) ** 0.5
    print(
        f"selected trial {chosen['trial']}: observed {chosen['value']:.4f}, posterior mean {posterior_mean:.4f}, θ={chosen['theta']}; "
        f"θ=0 baseline {baseline.mean():.4f} ± {baseline_sem:.4f}"
    )
    return dict(
        chosen,
        steps=len(trials),
        baseline=baseline.mean().item(),
        baseline_sem=baseline_sem.item(),
    )
