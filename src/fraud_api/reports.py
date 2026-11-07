"""Report artifacts for the README: the Optuna trials table and the importance chart."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

REPORTS_DIR = Path("reports")
TRIALS_FILE = "optuna_trials.csv"
IMPORTANCE_PLOT = "param_importance.png"

_BAR = "#2a78d6"
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_MUTED = "#898781"


def plot_param_importances(importances: dict[str, float], path: Path) -> None:
    """Horizontal bars, most important on top, each labelled with its share."""
    names = list(importances)[::-1]
    values = [importances[name] for name in names]

    fig, ax = plt.subplots(figsize=(7, 0.42 * len(names) + 1.2), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    bars = ax.barh(names, values, height=0.5, color=_BAR)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_width() + max(values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0%}",
            va="center",
            fontsize=9,
            color=_INK_SECONDARY,
        )

    ax.set_title(
        "Which hyperparameters mattered (fANOVA importance)",
        loc="left",
        fontsize=11,
        color=_INK,
    )
    ax.set_xlim(0, max(values) * 1.15)
    ax.xaxis.set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=9, labelcolor=_INK_SECONDARY)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(_MUTED)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)
