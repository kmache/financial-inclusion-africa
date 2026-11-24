"""Style et graphiques réutilisables (palettes validées daltonisme, nombres au format français).

Conventions de couleur, identiques dans tout le projet :

- ``YES`` (bleu) = a un compte bancaire, ``NO`` (orange) = n'a pas de compte : réservés à la cible ;
- ``COMPOSITION`` : modalités d'une variable explicative (camemberts) ;
- ``BLUES`` : échelle séquentielle à une seule teinte (cartes de chaleur).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import blended_transform_factory

from financial_inclusion_africa.config import FIGURES_DIR

# Palette catégorielle de référence (ordre fixe, validée CVD) et encres du texte
SERIES = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
YES, NO = SERIES[0], SERIES[1]
COMPOSITION = [
    "#1baf7a",
    "#6250d6",
    "#eda100",
    "#e87ba4",
]  # validée : ΔE CVD adjacent ≥ 16
SURFACE = "#fcfcfb"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e6e5e0"
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue",
    [
        "#f3f8fe",
        "#cde2fb",
        "#9ec5f4",
        "#6da7ec",
        "#3987e5",
        "#256abf",
        "#184f95",
        "#0d366b",
    ],
)
LABELS_TARGET = {0: "N'a pas de compte", 1: "A un compte bancaire"}
NNBSP = " "  # espace fine insécable


def fr(text: str) -> str:
    """Convertit un nombre formaté à l'anglaise (``8,735.5`` ; ``14.1%``) au format français."""
    return text.replace(",", NNBSP).replace(".", ",").replace("%", f"{NNBSP}%")


def fmt_int(n: float) -> str:
    return fr(f"{n:,.0f}")


def fmt_pct(x: float, decimals: int = 1) -> str:
    return fr(f"{x:.{decimals}%}")


def set_style() -> None:
    """Style sobre : fond clair, grille discrète, axes haut/droite masqués."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 14,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": INK_3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "legend.frameon": False,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "axes.formatter.use_locale": False,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    """Enregistre la figure dans ``reports/figures/<name>.png``."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight")


def pie(
    ax: plt.Axes, counts: pd.Series, title: str, colors: list[str] | None = None
) -> None:
    """Camembert de composition (≤ 4 parts) : % et effectif dans chaque part, noms en légende."""
    colors = colors or COMPOSITION[: len(counts)]
    total = counts.sum()
    wedges, _, autotexts = ax.pie(
        counts.to_numpy(),
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=lambda p: f"{fmt_pct(p / 100)}\n({fmt_int(p * total / 100)})",
        pctdistance=0.68,
        wedgeprops={"edgecolor": SURFACE, "linewidth": 2},
        textprops={"color": INK, "fontsize": 8.5},
    )
    for t, share in zip(autotexts, counts / total, strict=True):
        t.set_visible(share >= 0.05)  # pas de texte illisible dans une part minuscule
    ax.legend(
        wedges,
        [f"{k} · {fmt_pct(v / total)}" for k, v in counts.items()],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=1,
        fontsize=9,
        handlelength=1,
    )
    ax.set_title(title, loc="center")
    ax.axis("equal")


def count_bars(
    ax: plt.Axes, counts: pd.Series, title: str, threshold: int | None = None
) -> None:
    """Barres horizontales d'effectifs, triées ; ligne optionnelle de seuil de rareté."""
    counts = counts.sort_values()
    ax.barh(counts.index.astype(str), counts.to_numpy(), color=SERIES[0], height=0.65)
    for y, v in enumerate(counts.to_numpy()):
        ax.annotate(
            fmt_int(v),
            (v, y),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK_2,
        )
    if threshold is not None:
        ax.axvline(threshold, color=SERIES[7], lw=1.2, ls="--")
        ax.text(
            threshold,
            1.0,
            f" seuil {threshold}",
            transform=blended_transform_factory(ax.transData, ax.transAxes),
            va="bottom",
            fontsize=8,
            color=INK_2,
        )
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.15)


def rate_bars(
    ax: plt.Axes,
    table: pd.DataFrame,
    title: str,
    global_rate: float | None = None,
    sort: bool = True,
) -> None:
    """Part des personnes qui ont un compte par modalité (table issue de ``eda.target_rate``)."""
    t = table.sort_values("taux") if sort else table.iloc[::-1]
    y = np.arange(len(t))
    ax.barh(y, t["taux"].to_numpy(), color=YES, height=0.65)
    ax.set_yticks(y, [str(i) for i in t.index])
    for yi, (rate, n) in enumerate(zip(t["taux"], t["effectif"], strict=True)):
        ax.annotate(
            f"{fmt_pct(rate)}  (n = {fmt_int(n)})",
            (rate, yi),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK_2,
        )
    if global_rate is not None:
        ax.axvline(global_rate, color=INK_3, lw=1, ls="--")
        ax.text(
            global_rate,
            1.0,
            f" global {fmt_pct(global_rate)}",
            transform=blended_transform_factory(ax.transData, ax.transAxes),
            va="bottom",
            fontsize=8,
            color=INK_2,
        )
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: fmt_pct(v, 0)))
    ax.set_xlim(0, min(1.0, t["taux"].max() * 1.5))
    ax.set_title(title, pad=16)
    ax.grid(axis="y", visible=False)


def heatmap(
    ax: plt.Axes,
    table: pd.DataFrame,
    title: str,
    fmt: str = "{:.0%}",
    vmin=None,
    vmax=None,
) -> None:
    """Carte de chaleur séquentielle (un seul bleu), valeur écrite dans chaque case, « — » si vide."""
    data = table.to_numpy(dtype=float)
    im = ax.imshow(data, cmap=BLUES, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(
        range(table.shape[1]), [str(c) for c in table.columns], rotation=30, ha="right"
    )
    ax.set_yticks(range(table.shape[0]), [str(i) for i in table.index])
    lo, hi = im.get_clim()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                ax.annotate(
                    "—", (j, i), ha="center", va="center", color=INK_3, fontsize=8
                )
                continue
            dark = (v - lo) / (hi - lo + 1e-12) > 0.55
            ax.annotate(
                fr(fmt.format(v)),
                (j, i),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if dark else INK,
            )
    ax.grid(False)
    ax.set_title(title)
    for s in ax.spines.values():
        s.set_visible(False)


def fr_axis(ax: plt.Axes, axis: str = "both", fmt: str = "{:.1f}") -> None:
    """Graduations au format français (virgule décimale) sur l'axe ``x``, ``y`` ou les deux."""
    formatter = mpl.ticker.FuncFormatter(lambda v, _: fr(fmt.format(v)))
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(formatter)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(formatter)


def reliability(
    ax: plt.Axes,
    y_true,
    p,
    n_bins: int = 10,
    color: str = INK,
    label: str | None = None,
    ci: bool = True,
    hist: bool = False,
    lim: float = 1.0,
) -> None:
    """Diagramme de fiabilité (slide 41) : dans chaque intervalle de probabilité, fréquence
    observée des positifs en fonction de la probabilité moyenne annoncée.

    ``ci`` : intervalle de confiance à 95 % de la fréquence observée (approximation normale) :
    un intervalle peu peuplé est bruité. ``hist`` : répartition des probabilités en bas du
    graphique (barres grises), pour voir où se trouvent les personnes. ``lim`` : zoom sur
    ``[0 ; lim]``.
    """
    from financial_inclusion_africa.evaluate import reliability_table

    t = reliability_table(y_true, p, n_bins)
    conf, acc, n = t["annoncé (conf)"], t["observé (acc)"], t["effectif"]
    if hist:
        axh = ax.twinx()
        width = 1 / n_bins
        centers = (np.arange(n_bins) + 0.5) * width
        counts = np.bincount(
            np.clip((np.asarray(p) * n_bins).astype(int), 0, n_bins - 1),
            minlength=n_bins,
        )
        axh.bar(centers, counts, width=width * 0.9, color=GRID, zorder=0)
        axh.set_ylim(0, counts.max() / 0.22)  # barres dans le bas du graphique
        axh.set_yticks([])
        axh.set_box_aspect(1)
        axh.grid(False)
        for s in axh.spines.values():
            s.set_visible(False)
        ax.set_zorder(axh.get_zorder() + 1)
        ax.patch.set_visible(False)
    ax.plot([0, lim], [0, lim], ls="--", lw=1, color=INK_3, zorder=1)
    if ci:
        half = 1.96 * np.sqrt(acc * (1 - acc) / n)
        ax.errorbar(
            conf, acc, yerr=half, fmt="none", ecolor=color, elinewidth=1, alpha=0.6
        )
    ax.plot(conf, acc, "o-", color=color, ms=4.5, lw=1.6, label=label, zorder=3)
    ax.set(xlim=(0, lim), ylim=(0, lim))
    ax.set_box_aspect(1)
    ax.set_xlabel("Probabilité annoncée (moyenne de l'intervalle)")
    ax.set_ylabel("Fréquence observée")
    fr_axis(ax)
