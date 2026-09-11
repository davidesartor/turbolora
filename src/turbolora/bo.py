"""Fixed-noise GP, denoised pick and search arguments shared by turbo.search (after collaborators-poc)."""

import argparse
import math
from pathlib import Path
from typing import Callable
from jaxtyping import Float
from torch import Tensor

import torch
from botorch.exceptions import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.priors import LogNormalPrior
from gpytorch.mlls import ExactMarginalLogLikelihood

Objective = Callable[
    [Float[Tensor, "B T"], int], list[tuple[float, float]]
]  # (θ batch, batch index) -> (mean, SEM) per row
Search = Callable[[argparse.Namespace, Objective, int, Path, Callable[[int, dict], None] | None], dict | None]


def fit_gp(trials: list[dict], bounds: Float[Tensor, "2 D"] | None = None, region: float = 1.0) -> SingleTaskGP:
    """GP on the objective with each trial's sem² as observation noise plus a learned noise floor (Matérn 5/2 ARD, normalized θ, standardized value).

    Lengthscales get the dimension-scaled prior LogNormal(√2 + log(region·√D), √3) with unit signal variance (Hvarfner 2024;
    AdaScale-TuRBO scales it by the region side), so the MLL fit is MAP. `bounds` fixes the normalization box; None = data range.
    """
    X = torch.tensor([t["theta"] for t in trials], dtype=torch.float64)
    Y = torch.tensor([[t["value"]] for t in trials], dtype=torch.float64)
    Yvar = torch.tensor([[t["sem"] ** 2] for t in trials], dtype=torch.float64)
    dim = X.shape[-1]
    loc = math.sqrt(2) + math.log(region * math.sqrt(dim))
    kernel = MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_prior=LogNormalPrior(loc, math.sqrt(3)))
    kernel.lengthscale = math.exp(loc)
    gp = SingleTaskGP(
        X,
        Y,
        Yvar,
        covar_module=kernel,
        input_transform=Normalize(d=dim, bounds=bounds),
        outcome_transform=Standardize(m=1),
    )
    # the floor absorbs misfit beyond the binomial sem; the model holds the sem² already in standardized units
    gp.likelihood = FixedNoiseGaussianLikelihood(noise=gp.likelihood.noise.detach(), learn_additional_noise=True).to(X)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    try:
        fit_gpytorch_mll(mll)
    except ModelFittingError:  # L-BFGS line search can die on near-flat MLLs; Adam is slower but does not
        fit_gpytorch_mll(mll, optimizer=fit_gpytorch_mll_torch)
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
