"""Figure fig_obsmodel (Gaussian on the pass rate vs on its logit, each against the Beta posterior in its own space): `uv run --no-project --with matplotlib --with numpy --with scipy overleaf/scripts/obsmodel.py` from the repo root."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.special import betaln, digamma, polygamma
from scipy.stats import beta, norm

from common import BASE_STYLE, COLORS, OUT, PERCENT, TEXT_WIDTH, finish

N_PROMPTS = 256  # one LlMBO_T=1 evaluation: one completion on each of n prompts
SOLVED = [0, 3, 26, 128, 243]
POSTERIOR_FILL = "0.88"
OURS_STYLE = dict(color=COLORS["bo"], ls="--", lw=1.2)


def logit_beta_pdf(y: np.ndarray, a: float, b: float) -> np.ndarray:
    """Density of logit(X) for X ~ Beta(a, b)."""
    return np.exp(a * y - (a + b) * np.logaddexp(0, y) - betaln(a, b))


def draw_posterior(ax, x: np.ndarray, density: np.ndarray) -> None:
    ax.fill_between(x, density, color=POSTERIOR_FILL, lw=0)
    ax.plot(x, density, color="black", lw=1)
    ax.set_yticks([])
    ax.set_ylim(0, 1.15 * density.max())


def main() -> None:
    fig, axes = plt.subplots(2, len(SOLVED), figsize=(TEXT_WIDTH, 2.7))
    for col, s in enumerate(SOLVED):
        a, b = 1 + s, 1 + N_PROMPTS - s
        rate_post = beta(a, b)

        # pass-rate space: plug-in binomial Gaussian, a point mass at s = 0 or n
        lo, hi = rate_post.ppf([1e-4, 1 - 1e-4])
        rate = np.linspace(max(lo - 0.25 * (hi - lo), 0), min(hi + 0.25 * (hi - lo), 1), 1000)
        draw_posterior(axes[0, col], rate * 100, rate_post.pdf(rate))
        p_hat = s / N_PROMPTS
        sd_rate = np.sqrt(p_hat * (1 - p_hat) / N_PROMPTS)
        if sd_rate > 0:
            axes[0, col].plot(rate * 100, norm(p_hat, sd_rate).pdf(rate), **BASE_STYLE)
        else:
            axes[0, col].axvline(p_hat * 100, **BASE_STYLE)
            axes[0, col].text(0.95, 0.85, r"$\sigma = 0$", transform=axes[0, col].transAxes, ha="right", fontsize=7)

        # logit space: Gaussian with the exact moments of logit(Beta), as handed to the GP
        mu, sd_logit = digamma(a) - digamma(b), np.sqrt(polygamma(1, a) + polygamma(1, b))
        y = np.linspace(mu - 4.5 * sd_logit, mu + 4.5 * sd_logit, 1000)
        draw_posterior(axes[1, col], y, logit_beta_pdf(y, a, b))
        axes[1, col].plot(y, norm(mu, sd_logit).pdf(y), **OURS_STYLE)

        axes[0, col].set_title(f"$s = {s}$ of $n = {N_PROMPTS}$")
        axes[0, col].set_xlabel("pass rate", labelpad=1)
        axes[0, col].xaxis.set_major_formatter(PERCENT)
        axes[1, col].set_xlabel("logit pass rate", labelpad=1)

    axes[0, 0].set_ylabel("Gaussian on\nthe rate")
    axes[1, 0].set_ylabel("Gaussian on\nthe logit (ours)")
    handles = [
        Patch(facecolor=POSTERIOR_FILL, edgecolor="black", lw=1, label=r"$\mathrm{Beta}(1 + s, 1 + n - s)$ posterior"),
        Line2D([], [], **BASE_STYLE, label=r"$\mathcal{N}(\hat p, \hat p(1 - \hat p)/n)$, $\hat p = s/n$"),
        Line2D([], [], **OURS_STYLE, label=r"$\mathcal{N}(y, \sigma^2)$, moments of the logit"),
    ]
    finish(fig, handles, OUT / "fig_obsmodel.pdf")


if __name__ == "__main__":
    main()
