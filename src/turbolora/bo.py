"""Gradient-free search: θ=0 replicates → Sobol → heteroskedastic GP → Thompson sampling over a boxed vector, with a resumable trial log (port of collaborators-poc)."""

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
from botorch.posteriors import GPyTorchPosterior
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

Objective = Callable[
    [Float[Tensor, "T"], int], tuple[float, float]
]  # (θ, trial) -> (mean, SEM)


class HeteroskedasticGP(SingleTaskGP):
    """Collaborators' model: a GP on log observation variance supplies fixed per-point noise to the GP on the objective."""

    def __init__(
        self,
        train_X: Float[Tensor, "N T"],
        train_Y: Float[Tensor, "N 1"],
        train_Yvar: Float[Tensor, "N 1"],
        n_samples: int,
    ):
        d = train_X.shape[-1]
        X_scaler, Y_scaler, log_Yvar_scaler = (
            Normalize(d=d),
            Standardize(m=1),
            Standardize(m=1),
        )
        X, (Y, _), (log_Yvar, _) = (
            X_scaler(train_X),
            Y_scaler(train_Y),
            log_Yvar_scaler(train_Yvar.log()),
        )
        for scaler in (X_scaler, Y_scaler, log_Yvar_scaler):
            scaler.eval()

        # noise GP: variance of a log sample variance over n samples ≈ 2/(n-1)
        noise_likelihood = GaussianLikelihood()
        noise_likelihood.noise = torch.tensor(
            2.0 / (n_samples - 1), dtype=torch.float64
        )
        noise_model = SingleTaskGP(X, log_Yvar, likelihood=noise_likelihood)
        fit_gpytorch_mll(
            ExactMarginalLogLikelihood(noise_model.likelihood, noise_model)
        )
        noise_model.eval().requires_grad_(False)
        with torch.no_grad():
            predicted_var = log_Yvar_scaler.untransform(noise_model.posterior(X).mean)[
                0
            ].exp()

        # objective GP with that noise fixed (in standardized-Y units)
        super().__init__(
            X,
            Y,
            likelihood=FixedNoiseGaussianLikelihood(
                noise=predicted_var.squeeze(-1) / Y_scaler._stdvs_sq[0]
            ),
            covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d)),
        )
        self.X_scaler, self.Y_scaler, self.noise_model = X_scaler, Y_scaler, noise_model
        fit_gpytorch_mll(ExactMarginalLogLikelihood(self.likelihood, self))
        self.eval()

    def posterior(self, X: Float[Tensor, "M T"], **_) -> GPyTorchPosterior:
        self.eval()
        return self.Y_scaler.untransform_posterior(
            GPyTorchPosterior(self(self.X_scaler(X)))
        )


def fit_gp(trials: list[dict], n_samples: int) -> HeteroskedasticGP:
    X = torch.tensor([t["theta"] for t in trials], dtype=torch.float64)
    Y = torch.tensor([[t["value"]] for t in trials], dtype=torch.float64)
    Yvar = torch.tensor([[t["sem"] ** 2] for t in trials], dtype=torch.float64)
    return HeteroskedasticGP(X, Y, Yvar, n_samples)


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
        help="searched trials, on top of the θ=0 replicates; None = objective-specific default",
    )
    parser.add_argument("--thompson-candidates", type=int, default=2048)
    return parser


def search(
    args: argparse.Namespace, objective: Objective, dim: int, n_samples: int, out: Path
) -> dict | None:
    """Maximize `objective` over [-θ_range, θ_range]^dim, logging to `<out>/trials.json`; returns the evaluated trial with the best GP posterior mean.

    `n_samples` is what each objective value averages over (sets the noise model). Resumes from the log; a SIGUSR1/SIGTERM
    finishes the running trial and returns None.
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
    while len(trials) < args.n_baseline + args.n_evals and not stop.requested:
        i = len(trials)
        if i < args.n_baseline:
            theta = torch.zeros(dim, dtype=torch.float64)
        elif i - args.n_baseline < args.n_sobol:
            theta = sobol[i - args.n_baseline]
        else:
            # Thompson sampling: one posterior draw over uniform random candidates, take its argmax
            gp = fit_gp(trials, n_samples)
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
    if stop.requested:
        print("stopped on signal; rerun to resume")
        return None

    # final pick: denoised best-observed; θ=0 competes like any other point (its replicates just make it well-measured)
    gp = fit_gp(trials, n_samples)
    with torch.no_grad():
        means = gp.posterior(
            torch.tensor([t["theta"] for t in trials], dtype=torch.float64)
        ).mean.squeeze(-1)
    chosen = trials[int(means.argmax())]
    baseline = torch.tensor([t["value"] for t in trials if t["baseline"]])
    baseline_sem = baseline.std(unbiased=len(baseline) > 1) / len(baseline) ** 0.5
    print(
        f"selected trial {chosen['trial']}: observed {chosen['value']:.4f}, posterior mean {means.max():.4f}, θ={chosen['theta']}; "
        f"θ=0 baseline {baseline.mean():.4f} ± {baseline_sem:.4f}"
    )
    return dict(
        chosen,
        steps=len(trials),
        baseline=baseline.mean().item(),
        baseline_sem=baseline_sem.item(),
    )
