import re
from dataclasses import dataclass
from itertools import chain

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch, Rectangle
from tqdm import tqdm

from src.evaluation.metrics import transitions

sns.set_theme(style="whitegrid", palette="colorblind", font="DejaVu Sans",
              rc={"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                  "legend.fontsize": 8, "figure.titlesize": 9, "pdf.fonttype": 42})
WIDTH = 6.3  

METRICS = {"change_rate": ("change rate", ".3f")}
LOWER, HIGHER = "#4393c3", "#d6604d"
PANELS = 9

STEPS = {
    "binary": {-1: ("yes → no", LOWER), 1: ("no → yes", HIGHER)},
    "ordinal": {-2: ("2+ levels lower", "#2166ac"), -1: ("1 level lower", "#92c5de"),
                1: ("1 level higher", "#f4a582"), 2: ("2+ levels higher", "#b2182b")},
    "continuous": {-1: ("lower", LOWER), 1: ("higher", HIGHER)},
}


@dataclass
class FigureData:
    tests: pd.DataFrame
    pairs: pd.DataFrame
    schema: dict
    config: object


def safe(*parts) -> str:
    return "_".join(re.sub(r"->|[^\w.-]+", "_", str(p)).strip("_") for p in parts if p)


def _row_label(sensitive_attribute: str, comparison: str) -> str:
    ref, cf = comparison.split("->")
    return f"{sensitive_attribute}: {cf if ref == 'original' else f'{ref} → {cf}'}"


def labelled(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(row=[_row_label(a, c) for a, c in zip(df["sensitive_attribute"], df["comparison"])])


def pretty(variable: str) -> str:
    return variable.replace("_", " ")


def pretty_category(value) -> str:
    return str(value).replace("_", " ").lower()


def _tilt(ax) -> None:
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


def star_levels(alpha: float) -> list:
    return sorted((t for t in {alpha, 0.01, 0.001} if t <= alpha), reverse=True)


def _stars(p: float, alpha: float) -> str:
    return "*" * sum(p < t for t in star_levels(alpha))


def interval(v, lo, hi, fmt: str = ".2f") -> str | None:
    """'value [lower, upper]', e.g. '0.140 [0.090, 0.220]'; None when the value is missing, no brackets without a CI."""
    ci = "" if pd.isna(lo) else f" [{lo:{fmt.lstrip('+')}}, {hi:{fmt.lstrip('+')}}]"
    return None if pd.isna(v) else f"{v:{fmt}}{ci}"


def _p_text(p: float) -> str:
    return "" if pd.isna(p) else "<0.001" if p < 0.001 else f"{p:.3f}"


def result(r, alpha: float) -> tuple[str, str]:
    """Metric with CI and adjusted p with stars of one test row."""
    if pd.isna(r.metric_value):
        return "no valid pairs", ""
    metric = interval(r.metric_value, r.metric_ci_lower, r.metric_ci_upper, METRICS[r.metric][1])
    p = r.adjusted_p_value
    return metric, f"{_p_text(p)} {_stars(p, alpha)}".strip()


def unchanged(t: pd.DataFrame) -> bool:
    """True if no case changed in any test of an output"""
    return not t["metric_value"].sum()


def _mark_rows(ax, rows: list, heatmap=False) -> None:
    """Separate sensitive attribute groups"""
    for k in range(1, len(rows)):
        if rows[k].split(":")[0] != rows[k - 1].split(":")[0]:
            ax.axhline(k if heatmap else k - 0.5, color="0.2" if heatmap else "0.6", lw=1.2 if heatmap else 0.7)
    if rows[-1].startswith("control:") and not heatmap:
        ax.axhspan(len(rows) - 1.5, len(rows) - 0.5, color="0.92", zorder=0)


def _mark_significant(ax, rows: list, t: pd.DataFrame, alpha: float, heatmap=False) -> None:
    p = t.set_index("row")["adjusted_p_value"]
    for y, row in enumerate(rows):
        if stars := _stars(p.get(row, np.nan), alpha):
            ax.text(1.01, y + 0.5 * heatmap, stars, transform=ax.get_yaxis_transform(), va="center", fontsize=8)


def _panels(n: int, rows: int, extra: float = 0):
    """One panel per model, two per row"""
    ncols = min(n, 2)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(WIDTH, 0.8 + nrows * (0.45 + extra + 0.19 * rows)), squeeze=False, layout="constrained")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, [(ax, i % ncols == 0, i + ncols >= n) for i, ax in enumerate(axes.flat[:n])]


def overview(d: FigureData):
    t = labelled(d.tests[d.tests["metric_value"].notna()])
    for model, g in t.groupby("model", sort=False):
        rows, groups = list(dict.fromkeys(g["row"])), list(g.groupby("metric", sort=False))
        fig, axes = plt.subplots(1, len(groups), figsize=(WIDTH, 2.2 + 0.3 * len(rows)), squeeze=False, layout="constrained",
                                 width_ratios=[p["variable"].nunique() for _, p in groups])
        for ax, (metric, p) in zip(axes[0], groups):
            name, fmt = METRICS[metric]
            columns = list(dict.fromkeys(p["variable"]))
            cell = lambda values: p.pivot(index="row", columns="variable", values=values).reindex(index=rows, columns=columns)
            grid, adjusted, sig = cell("metric_value"), cell("adjusted_p_value"), cell("statistically_significant")
            labels = grid.map(lambda v: "" if np.isnan(v) else f"{v:{fmt}}") + "\n" + adjusted.map(_p_text)
            sns.heatmap(grid, ax=ax, annot=labels, fmt="", cbar=False, linewidths=0.5, linecolor="white",
                        annot_kws={"fontsize": 5.5}, cmap="Blues", vmin=0, vmax=np.nanmax(grid.to_numpy(), initial=0) or 1,
                        yticklabels=rows if ax is axes[0][0] else False, xticklabels=[pretty(c) for c in columns])
            for i, j in zip(*np.nonzero(sig.to_numpy() == True)):
                ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor="#f16913", lw=2))
            ax.set(xlabel="", ylabel="")
            ax.set_title(name.replace(" (", "\n("), fontsize=8)
            _tilt(ax)
            ax.tick_params(axis="y", rotation=0)
            _mark_rows(ax, rows, heatmap=True)
        fig.suptitle(model)
        yield "overview", safe(model, "overview"), fig


def change_plot(d: FigureData, field):
    t = labelled(d.tests[d.tests["variable"] == field])
    if unchanged(t):
        return
    kind = t["variable_type"].iloc[0]
    steps = STEPS[kind]
    df = labelled(d.pairs.dropna(subset=[f"{field}__ref", f"{field}__cf"]))
    diff = df[f"{field}__cf"].astype(float) - df[f"{field}__ref"].astype(float)
    step = np.sign(diff) * np.minimum(np.ceil(diff.abs()), max(steps))
    counts = pd.crosstab([df["model"], df["row"]], step)
    count = counts.reindex(columns=list(steps), fill_value=0)
    share = count.div(counts.sum(axis=1), axis=0) * 100
    label = lambda n, v: f"{n:.0f} ({v:.1f}%)"
    sides = [[k for k in steps if np.sign(k) == side] for side in (-1, 1)]
    # room beside the longest bar for its label: about 0.045 of the half axis per character
    lim = max([4] + [v / (0.85 - 0.045 * len(label(n, v))) for ks in sides
                     for n, v in zip(count[ks].sum(axis=1), share[ks].sum(axis=1))])
    models, rows = list(dict.fromkeys(t["model"])), list(dict.fromkeys(t["row"]))
    fig, grid = _panels(len(models), len(rows))
    for (ax, first, bottom), model in zip(grid, models):
        s, c = (x.reindex(pd.MultiIndex.from_product([[model], rows])).fillna(0).droplevel(0) for x in (share, count))
        for side, ks in zip((-1, 1), sides):
            end, n = np.zeros(len(rows)), np.zeros(len(rows))
            for k in sorted(ks, key=abs):
                ax.barh(range(len(rows)), side * s[k].to_numpy(), left=side * end, color=steps[k][1], height=0.7)
                end += s[k].to_numpy()
                n += c[k].to_numpy()
            for y, (v, m) in enumerate(zip(end, n)):
                if m:
                    ax.text(side * (v + lim * 0.02), y, label(m, v), ha="left" if side > 0 else "right", va="center",
                            fontsize=7)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(len(rows)), rows)
        ax.set(xlim=(-lim, lim), ylim=(len(rows) - 0.5, -0.5), xlabel="% of cases" if bottom else "")
        ax.set_xticks([x for x in ax.get_xticks() if abs(x) <= min(100, 0.9 * lim)])
        ax.xaxis.set_major_formatter(lambda x, _: f"{abs(x):.0f}")
        ax.tick_params(labelleft=first, labelbottom=bottom)
        ax.grid(visible=False)
        ax.set_title(model.split("/")[-1])
        _mark_rows(ax, rows)
        _mark_significant(ax, rows, t[t["model"] == model], d.config.statistics.alpha)
    fig.legend(handles=[Patch(color=steps[k][1], label=steps[k][0]) for k in sorted(steps)], loc="outside lower center",
               ncols=len(steps), frameon=False)
    yield kind, safe(field), fig


def category_plot(d: FigureData, field):
    t = labelled(d.tests[d.tests["variable"] == field])
    if unchanged(t):
        return
    ref, cf = f"{field}__ref", f"{field}__cf"
    df = labelled(d.pairs.dropna(subset=[ref, cf]))
    cats = [v for v in d.schema["fields"][field]["values"] if v in set(df[ref]) | set(df[cf])]
    share = lambda col: pd.crosstab([df["model"], df["row"]], df[col], normalize="index").reindex(columns=cats, fill_value=0)
    delta = (share(cf) - share(ref)) * 100
    lim = np.nanmax(np.abs(delta.to_numpy()), initial=0) or 1
    models, rows = list(dict.fromkeys(t["model"])), list(dict.fromkeys(t["row"]))
    fig, grid = _panels(len(models), len(rows), extra=0.8)
    for (ax, first, _), model in zip(grid, models):
        g = delta.reindex(pd.MultiIndex.from_product([[model], rows])).droplevel(0)
        sns.heatmap(g, ax=ax, annot=g.map(lambda v: "" if np.isnan(v) else f"{v:+.1f}" if round(v, 1) else "0"), fmt="",
                    cmap="RdBu_r", vmin=-lim, vmax=lim, cbar=False, linewidths=0.5, linecolor="white",
                    annot_kws={"fontsize": 6.5}, yticklabels=rows, xticklabels=[pretty_category(c) for c in cats])
        ax.set(xlabel="", ylabel="")
        ax.set_title(model.split("/")[-1])
        _tilt(ax)
        ax.tick_params(axis="y", rotation=0)
        ax.tick_params(labelleft=first)
        _mark_rows(ax, rows, heatmap=True)
        _mark_significant(ax, rows, t[t["model"] == model], d.config.statistics.alpha, heatmap=True)
    yield "nominal", safe(field), fig


def transition_plot(d: FigureData, field):
    significant = labelled(d.tests[(d.tests["variable"] == field) & (d.tests["statistically_significant"] == True)])
    if significant.empty:
        return
    pairs = [d.pairs[(d.pairs["model"] == r.model) & (d.pairs["sensitive_attribute"] == r.sensitive_attribute) & (d.pairs["comparison"] == r.comparison)]
             [[f"{field}__ref", f"{field}__cf"]].dropna().to_numpy().T for r in significant.itertuples()]
    seen = set(chain.from_iterable(chain.from_iterable(pairs)))
    cats = [v for v in d.schema["fields"][field]["values"] if v in seen]
    labels, rows = [pretty_category(c) for c in cats], list(significant.itertuples())
    parts = np.array_split(np.arange(len(pairs)), -(-len(pairs) // PANELS))
    for part, idx in enumerate(parts, 1):
        n, ncols = len(idx), min(len(idx), 3)
        nrows = -(-n // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(WIDTH, 0.2 + nrows * (1.2 + (WIDTH - 1.3) / ncols)), squeeze=False,
                                 layout="constrained")
        for i, (ax, k) in enumerate(zip(axes.flat, idx)):
            tr = transitions(*pairs[k], cats)
            pct = tr.pivot(index="original", columns="counterfactual", values="row_pct").loc[cats, cats].fillna(0)
            counts = tr.pivot(index="original", columns="counterfactual", values="count").loc[cats, cats]
            sns.heatmap(pct, ax=ax, annot=counts, fmt="d", cmap="Blues", vmin=0, vmax=100, cbar=False, square=True,
                        linewidths=0.5, linecolor="white", annot_kws={"fontsize": 6.5},
                        xticklabels=labels, yticklabels=labels if i % ncols == 0 else False)
            ax.set(xlabel="", ylabel="")
            ax.set_title(f"{rows[k].model.split('/')[-1]}\n{rows[k].row}", fontsize=8)
            _tilt(ax)
        for ax in axes.flat[n:]:
            ax.set_visible(False)
        yield "nominal", safe(field, "transitions", part if len(parts) > 1 else None), fig


BY_TYPE = {
    "binary": [change_plot],
    "ordinal": [change_plot],
    "continuous": [change_plot],
    "nominal": [category_plot, transition_plot],
    "free_text": [],
}
CROSS = [overview]


def make_figures(d: FigureData, figures_dir) -> int:
    vis, count = d.config.visualization, 0
    items = chain((fig for fn in CROSS for fig in fn(d)),
                  (fig for name, f in d.schema["fields"].items() for fn in BY_TYPE[f["type"]] for fig in fn(d, name)))
    for category, name, fig in tqdm(items, desc="figures    saved", unit=" figures", leave=False):
        count += 1
        (figures_dir / category).mkdir(parents=True, exist_ok=True)
        for fmt in vis.formats:
            fig.savefig(figures_dir / category / f"{count:03d}_{name}.{fmt}", dpi=vis.dpi, bbox_inches="tight")
        plt.close(fig)
    return count
