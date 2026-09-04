"""Gradient-free search: θ=0 replicates → Sobol → fixed-noise GP → Thompson sampling over a boxed vector, with a resumable trial log (after collaborators-poc)."""

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
    [Float[Tensor, "T"], int], tuple[float, float]
]  # (θ, trial) -> (mean, SEM)


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

    Each trial's sem² is taken as its observation noise. Resumes from the log; a SIGUSR1/SIGTERM finishes the running
    trial and returns None. `on_snapshot(step, pick)` fires after GP-guided trial 1, 2, 4, ... and the last (θ=0 replicates and Sobol design excluded) with the current pick.
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
    # a fixed sequence, so a resumed run continues it
    sobol = SobolEngine(dim, scramble=True, seed=args.seed).draw(args.n_sobol).double()
    sobol = bounds[0] + (bounds[1] - bounds[0]) * sobol

    # slurm preemption/wall-limit signal: finish the running trial, then leave the loop
    stop = argparse.Namespace(requested=False)
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, lambda *_: setattr(stop, "requested", True))
    while len(trials) < args.n_baseline + args.n_sobol + args.n_evals and not stop.requested:
        i = len(trials)
        if i < args.n_baseline:
            theta = torch.zeros(dim, dtype=torch.float64)
        elif i - args.n_baseline < args.n_sobol:
            theta = sobol[i - args.n_baseline]
        else:
            # Thompson sampling: one posterior draw over uniform random candidates, take its argmax
            gp = fit_gp(trials)
            candidates = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(
                args.thompson_candidates, dim, dtype=torch.float64
            )
            with torch.no_grad():
                theta = candidates[gp.posterior(candidates).sample().argmax()]
        value, sem = objective(theta, i)
        trials.append(
            dict(
                trial=i,
                baseline=i < args.n_baseline,
                theta=theta.tolist(),
                value=value,
                sem=sem,
            )
        )
        trials_path.write_text(json.dumps(trials, indent=1))
        best = max(t["value"] for t in trials)
        print(
            f"trial {i}: {value:.4f} ± {sem:.4f} (best {best:.4f}) θ={[round(x, 4) for x in theta.tolist()]}"
        )
        # snapshot steps count GP-guided trials only: the θ=0 replicates and the Sobol design are the initial design
        step = len(trials) - args.n_baseline - args.n_sobol
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
