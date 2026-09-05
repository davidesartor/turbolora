"""TuRBO-1 after the BoTorch tutorial (TurboState / update_state / generate_batch), on bo.py's fixed-noise GP and trial log."""

import argparse
import json
import math
import signal
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from jaxtyping import Float
from torch import Tensor
from torch.quasirandom import SobolEngine

from turbolora.bo import Objective, fit_gp, pick


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--tr-init",
        type=float,
        default=0.8,
        help="initial region side in unit-cube coordinates",
    )
    parser.add_argument(
        "--tr-min",
        type=float,
        default=0.5**7,
        help="side below which the region restarts from a fresh Sobol design",
    )
    parser.add_argument("--tr-max", type=float, default=1.6)
    parser.add_argument(
        "--succ-tol",
        type=int,
        default=3,
        help="consecutive improving batches that double the side (paper value; the BoTorch tutorial uses 10)",
    )
    parser.add_argument(
        "--fail-tol",
        type=int,
        default=None,
        help="consecutive non-improving batches that halve it (default max(4, dim)/batch)",
    )
    return parser


@dataclass
class TurboState:
    """Trust-region state, persisted to `<out>/turbo.json`; `restart_trial` is the first trial of the current region."""

    length: float
    length_min: float
    length_max: float
    success_tolerance: int
    failure_tolerance: int
    success_counter: int = 0
    failure_counter: int = 0
    best_value: float = -float("inf")
    restart_triggered: bool = False
    restarts: int = 0
    restart_trial: int = 0


def update_state(state: TurboState, best_mean: float) -> TurboState:
    """Noisy-case rule (paper §2): the region's best posterior mean beating the previous best by 1e-3 (relative) is a success; tolerances double or halve the side."""
    if state.best_value == -float("inf"):
        state.best_value = best_mean
        return state
    if best_mean > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter, state.failure_counter = state.success_counter + 1, 0
    else:
        state.success_counter, state.failure_counter = 0, state.failure_counter + 1
    if state.success_counter == state.success_tolerance:
        state.length, state.success_counter = (
            min(2.0 * state.length, state.length_max),
            0,
        )
    elif state.failure_counter == state.failure_tolerance:
        state.length, state.failure_counter = state.length / 2.0, 0
    state.best_value = max(state.best_value, best_mean)
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def generate_batch(
    state: TurboState,
    gp: SingleTaskGP,
    X: Float[Tensor, "N D"],
    Y: Float[Tensor, "N"],
    batch_size: int,
    seed: int,
) -> Float[Tensor, "B D"]:
    """BoTorch's Thompson step in the unit cube: a lengthscale-scaled box around X[argmax Y], Sobol perturbations of ~20 of its coordinates, `MaxPosteriorSampling` over them.

    Pass posterior means as Y so the center is the denoised best (the paper's noisy-case choice).
    """
    dim = X.shape[-1]
    n_candidates = min(5000, max(2000, 200 * dim))
    x_center = X[Y.argmax(), :].clone()
    weights = gp.covar_module.lengthscale.squeeze(0).detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    pert = SobolEngine(dim, scramble=True, seed=seed).draw(n_candidates).to(X)
    pert = tr_lb + (tr_ub - tr_lb) * pert
    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, dtype=X.dtype) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim, size=(len(ind),))] = True
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    thompson_sampling = MaxPosteriorSampling(model=gp, replacement=False)
    with torch.no_grad():
        return thompson_sampling(X_cand, num_samples=batch_size)


def search(
    args: argparse.Namespace,
    objective: Objective,
    dim: int,
    out: Path,
    on_snapshot: Callable[[int, dict], None] | None = None,
) -> dict | None:
    """Maximize `objective` over [-θ_range, θ_range]^dim with TuRBO-1; same log, resume, signal and snapshot contract as bo.search.

    The region's GP (unit-cube inputs, lengthscale prior scaled by the side) sees the θ=0 replicates plus every trial since the
    region started; success/failure and the center use its posterior mean. When the side drops below `tr_min` the region restarts
    from a fresh Sobol design of `n_sobol` points. Snapshots and the final pick use bo.pick over all trials.
    """
    trials_path = out / "trials.json"
    state_path = out / "turbo.json"
    trials: list[dict] = (
        json.loads(trials_path.read_text()) if trials_path.exists() else []
    )
    if args.theta_range is None or args.n_evals is None:
        raise ValueError(
            "theta_range and n_evals must be resolved by the caller before search"
        )
    if args.fail_tol is None:
        args.fail_tol = math.ceil(max(4.0, float(dim)) / args.batch)
    bounds = torch.tensor(
        [[-args.theta_range] * dim, [args.theta_range] * dim], dtype=torch.float64
    )
    unit_bounds = torch.tensor([[0.0] * dim, [1.0] * dim], dtype=torch.float64)
    to_theta = lambda x: bounds[0] + (bounds[1] - bounds[0]) * x  # noqa: E731
    to_unit = lambda theta: (theta - bounds[0]) / (bounds[1] - bounds[0])  # noqa: E731

    def region_design(state: TurboState) -> list[Float[Tensor, "B D"]]:
        """The first region starts like bo.search (θ=0 replicates, then Sobol); a restart gets its own Sobol design."""
        sobol = to_theta(
            SobolEngine(dim, scramble=True, seed=args.seed + 1000 * state.restarts)
            .draw(args.n_sobol)
            .double()
        )
        points = (
            torch.cat([torch.zeros(args.n_baseline, dim, dtype=torch.float64), sobol])
            if state.restarts == 0
            else sobol
        )
        return list(points.split(args.batch))

    def new_state(restarts: int = 0, restart_trial: int = 0) -> TurboState:
        return TurboState(
            args.tr_init,
            args.tr_min,
            args.tr_max,
            args.succ_tol,
            args.fail_tol,
            restarts=restarts,
            restart_trial=restart_trial,
        )

    state = (
        TurboState(**json.loads(state_path.read_text()))
        if state_path.exists()
        else new_state()
    )

    stop = argparse.Namespace(requested=False)
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, lambda *_: setattr(stop, "requested", True))
    while (
        step := len({t["batch"] for t in trials if not t["design"]})
    ) < args.n_evals and not stop.requested:
        b = trials[-1]["batch"] + 1 if trials else 0
        design = region_design(state)
        done = len(trials) - state.restart_trial
        if done < sum(len(d) for d in design):
            thetas, is_design = design[done // args.batch], True
        else:
            # the region's GP in unit-cube coordinates; its posterior mean drives success, center and restart
            local = [
                t for t in trials if t["baseline"] or t["trial"] >= state.restart_trial
            ]
            X = to_unit(torch.tensor([t["theta"] for t in local], dtype=torch.float64))
            gp = fit_gp(
                [dict(t, theta=x.tolist()) for t, x in zip(local, X)],
                unit_bounds,
                state.length,
            )
            with torch.no_grad():
                means = gp.posterior(X).mean.squeeze(-1)
            state = update_state(state, means.max().item())
            if state.restart_triggered:
                state = new_state(state.restarts + 1, len(trials))
                print(
                    f"batch {b}: region below {args.tr_min:.2e}, restart {state.restarts} from a fresh Sobol design"
                )
                thetas, is_design = region_design(state)[0], True
            else:
                thetas, is_design = (
                    to_theta(
                        generate_batch(state, gp, X, means, args.batch, args.seed + b)
                    ),
                    False,
                )
        results = objective(thetas, b)
        for theta, (value, sem) in zip(thetas, results):
            trials.append(
                dict(
                    trial=len(trials),
                    batch=b,
                    design=is_design,
                    baseline=is_design and bool((theta == 0).all()),
                    theta=theta.tolist(),
                    value=value,
                    sem=sem,
                    length=state.length,
                )
            )
        trials_path.write_text(json.dumps(trials, indent=1))
        state_path.write_text(json.dumps(asdict(state), indent=1))
        best = max(t["value"] for t in trials)
        batch_best = max(v for v, _ in results)
        print(
            f"batch {b}: best {batch_best:.4f} over {len(results)} θ's (best so far {best:.4f}, side {state.length:.3g}, "
            f"best mean {state.best_value:.4f})"
        )
        step = len({t["batch"] for t in trials if not t["design"]})
        if (
            on_snapshot
            and not is_design
            and (not step & (step - 1) or step == args.n_evals)
        ):
            on_snapshot(step, pick(trials)[0])
    if stop.requested:
        print("stopped on signal; rerun to resume")
        return None

    chosen, posterior_mean = pick(trials)
    baseline = torch.tensor([t["value"] for t in trials if t["baseline"]])
    baseline_sem = baseline.std(unbiased=len(baseline) > 1) / len(baseline) ** 0.5
    print(
        f"selected trial {chosen['trial']}: observed {chosen['value']:.4f}, posterior mean {posterior_mean:.4f}, θ={chosen['theta']}; "
        f"θ=0 baseline {baseline.mean():.4f} ± {baseline_sem:.4f}; {state.restarts} restarts"
    )
    return dict(
        chosen,
        steps=len(trials),
        baseline=baseline.mean().item(),
        baseline_sem=baseline_sem.item(),
    )
