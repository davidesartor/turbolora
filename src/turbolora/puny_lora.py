"""Gradient-free search on hit counts: θ=0 batch → Sobol batch → sparse variational GP with a Binomial likelihood → batched Thompson sampling, resumable trial log."""

import argparse
import json
import signal
from pathlib import Path
from typing import Callable

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import _OneDimensionalLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.priors import LogNormalPrior
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributions import Binomial
from torch.quasirandom import SobolEngine

Objective = Callable[[Float[Tensor, "B T"], int], list[tuple[int, int]]]  # (θ batch, batch index) -> (hits, n) per row


class BinomialLikelihood(_OneDimensionalLikelihood):
    """hits ~ Binomial(n, sigmoid(f)); the ELBO gets `n` as a keyword and integrates f by Gauss-Hermite quadrature."""

    def forward(self, function_samples: Tensor, n: Tensor, **kwargs) -> Binomial:
        return Binomial(total_count=n, logits=function_samples)


class LogitGP(ApproximateGP):
    """Sparse variational GP on the logit pass rate over θ/θ_range ∈ [-1, 1]^d, Matérn 5/2 ARD on fixed inducing points."""

    def __init__(self, inducing: Float[Tensor, "M D"], mean: float):
        strategy = VariationalStrategy(
            self, inducing, CholeskyVariationalDistribution(len(inducing)), learn_inducing_locations=False
        )
        super().__init__(strategy)
        self.mean_module = ConstantMean()
        self.mean_module.constant.data.fill_(mean)
        # the latent scale is not pinned by data variance as in a Gaussian model: priors keep it O(1) logit
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=inducing.shape[-1], lengthscale_prior=LogNormalPrior(0.0, 1.0)),
            outputscale_prior=LogNormalPrior(0.0, 1.0),
        )
        self.double()

    def forward(self, x: Float[Tensor, "N D"]) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


def counts(trials: list[dict]) -> tuple[list[list[float]], Int[Tensor, "N"], Int[Tensor, "N"]]:
    """Pool the rows of each distinct θ (a Binomial is closed under summing draws with the same p)."""
    pooled: dict[tuple, list[int]] = {}
    for t in trials:
        hits_n = pooled.setdefault(tuple(t["theta"]), [0, 0])
        hits_n[0] += t["hits"]
        hits_n[1] += t["n"]
    thetas = [list(k) for k in pooled]
    hits, n = (torch.tensor(col, dtype=torch.float64) for col in zip(*pooled.values()))
    return thetas, hits, n


def fit_gp(
    trials: list[dict], theta_range: float, n_inducing: int = 256, steps: int = 300, init: LogitGP | None = None
) -> LogitGP:
    """Fit the ELBO by Adam on the pooled counts; `init` warm-starts from the previous fit (same inducing set)."""
    thetas, hits, n = counts(trials)
    X = torch.tensor(thetas, dtype=torch.float64) / theta_range
    dim = X.shape[-1]
    inducing = SobolEngine(dim, scramble=True, seed=0).draw(n_inducing).double() * 2 - 1
    inducing = torch.cat([torch.zeros(1, dim, dtype=torch.float64), inducing])
    pass_rate = (hits.sum() + 0.5) / (n.sum() + 1)
    gp = LogitGP(inducing, torch.logit(pass_rate).item())
    if init is not None:
        gp.load_state_dict(init.state_dict())
    likelihood = BinomialLikelihood()
    mll = VariationalELBO(likelihood, gp, num_data=len(X))
    optimizer = torch.optim.Adam(gp.parameters(), lr=0.05)
    gp.train()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = -mll(gp(X), hits, n=n)
        loss.backward()
        optimizer.step()
    return gp.eval()


def thompson(gp: LogitGP, theta_range: float, dim: int, batch: int, n_candidates: int) -> Float[Tensor, "B D"]:
    """One posterior draw per batch row over uniform random candidates, each row's argmax is its θ."""
    candidates = torch.rand(n_candidates, dim, dtype=torch.float64) * 2 - 1
    with torch.no_grad():
        f = gp(candidates).rsample(torch.Size([batch]))
    return candidates[f.argmax(-1)] * theta_range


def pick(gp: LogitGP, trials: list[dict], theta_range: float) -> tuple[list[float], float]:
    """Denoised best-observed: the evaluated θ with the highest posterior mean logit (θ=0 competes like any other point)."""
    thetas, _, _ = counts(trials)
    with torch.no_grad():
        means = gp(torch.tensor(thetas, dtype=torch.float64) / theta_range).mean
    return thetas[int(means.argmax())], means.max().item()


def argument_parser() -> argparse.ArgumentParser:
    """Search arguments; the objective's own get added on top."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--theta-range", type=float, default=None, help="search box [-range, range]^dim; None = objective-specific default")
    parser.add_argument("--batch", type=int, default=1, help="θ's evaluated per objective call (one vLLM batch, one LoRA each)")
    parser.add_argument("--n-baseline", type=int, default=16, help="θ=0 rows in the initial design")
    parser.add_argument("--n-sobol", type=int, default=16, help="quasi-random rows in the initial design")
    parser.add_argument("--n-evals", type=int, default=None, help="GP-guided batches after the initial design; None = objective-specific default")
    parser.add_argument("--thompson-candidates", type=int, default=2048)
    parser.add_argument("--inducing", type=int, default=256, help="fixed Sobol inducing points of the sparse GP (+ the origin)")
    parser.add_argument("--fit-steps", type=int, default=300, help="Adam steps per ELBO fit")
    return parser


def search(
    args: argparse.Namespace,
    objective: Objective,
    dim: int,
    out: Path,
    on_snapshot: Callable[[int, list[float]], None] | None = None,
) -> dict | None:
    """Maximize the pass rate over [-θ_range, θ_range]^dim, logging rows to `<out>/trials.json`; returns the evaluated θ with the best posterior mean.

    Batches: the initial design (θ=0 replicates, then Sobol) in chunks of `batch`, then `n_evals` Thompson batches.
    Resumes from the log; SIGUSR1/SIGTERM finishes the running batch and returns None. `on_snapshot(step, θ)` fires after
    GP-guided batch 1, 2, 4, ... and the last with the current pick.
    """
    trials_path = out / "trials.json"
    trials: list[dict] = json.loads(trials_path.read_text()) if trials_path.exists() else []
    if args.theta_range is None or args.n_evals is None:
        raise ValueError("theta_range and n_evals must be resolved by the caller before search")
    # a fixed design, so a resumed run continues it
    sobol = SobolEngine(dim, scramble=True, seed=args.seed).draw(args.n_sobol).double() * 2 - 1
    design = torch.cat([torch.zeros(args.n_baseline, dim, dtype=torch.float64), sobol * args.theta_range])
    design_batches = [design[i : i + args.batch] for i in range(0, len(design), args.batch)]
    n_batches = len(design_batches) + args.n_evals

    # slurm preemption/wall-limit signal: finish the running batch, then leave the loop
    stop = argparse.Namespace(requested=False)
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, lambda *_: setattr(stop, "requested", True))
    gp: LogitGP | None = None
    while (b := trials[-1]["batch"] + 1 if trials else 0) < n_batches and not stop.requested:
        if b < len(design_batches):
            thetas = design_batches[b]
        else:
            gp = fit_gp(trials, args.theta_range, args.inducing, args.fit_steps, init=gp)
            thetas = thompson(gp, args.theta_range, dim, args.batch, args.thompson_candidates)
        results = objective(thetas, b)
        for theta, (hits, n) in zip(thetas, results):
            trials.append(dict(trial=len(trials), batch=b, baseline=b == 0 and bool((theta == 0).all()), theta=theta.tolist(), hits=hits, n=n))
        trials_path.write_text(json.dumps(trials, indent=1))
        hits, n = (sum(r[i] for r in results) for i in range(2))
        print(f"batch {b}: {hits}/{n} = {hits / n:.4f} over {len(results)} θ's")
        # snapshot steps count GP-guided batches only
        step = b + 1 - len(design_batches)
        if on_snapshot and step >= 1 and (not step & (step - 1) or step == args.n_evals):
            gp = fit_gp(trials, args.theta_range, args.inducing, args.fit_steps, init=gp)
            on_snapshot(step, pick(gp, trials, args.theta_range)[0])
    if stop.requested:
        print("stopped on signal; rerun to resume")
        return None

    gp = fit_gp(trials, args.theta_range, args.inducing, args.fit_steps, init=gp)
    theta, mean_logit = pick(gp, trials, args.theta_range)
    base_hits = sum(t["hits"] for t in trials if t["baseline"])
    base_n = sum(t["n"] for t in trials if t["baseline"])
    baseline = base_hits / base_n
    baseline_sem = (baseline * (1 - baseline) / base_n) ** 0.5
    posterior = torch.sigmoid(torch.tensor(mean_logit)).item()
    print(f"selected θ={theta}: posterior pass rate {posterior:.4f}; θ=0 baseline {baseline:.4f} ± {baseline_sem:.4f}")
    return dict(theta=theta, posterior=posterior, steps=args.n_evals, baseline=baseline, baseline_sem=baseline_sem)
