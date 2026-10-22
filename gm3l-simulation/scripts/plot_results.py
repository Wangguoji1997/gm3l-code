#!/usr/bin/env python3
"""Reproduce the simulation figure structures from a complete result table.

Figure contract: compare parameter-error distributions and their sample-size
trends (Figures 1 and 2), then posterior concentration (Figure 3). This is a
Python/matplotlib quantitative grid, retaining the established gray-scale
encodings, all replications and outliers, and natural logarithms. Boxplots use
1.5-IQR whiskers; trend points and bars are medians of log MSE, not logs of
median MSE. Trend lines are least-squares fits against log(n). Figure 3 bars
start at zero and have no error bars. No scientific values are clipped or
floored. PDF/SVG text remains editable, with PNG previews and numeric exports.

This reproduces numerical summaries and panel structures, not pixel-identical
PDF rendering across operating systems, fonts, and matplotlib versions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from matplotlib.patches import Patch
import numpy as np


PARAMETER_METHODS = ("SMLE", "ORACLE", "FINAL")
CLUSTERING_METHODS = ("GMM", "SMLE", "FINAL")
ALL_METHODS = set(PARAMETER_METHODS) | set(CLUSTERING_METHODS)
PARAMETERS = ("alpha", "mu", "sigma", "pi")
SYMBOLS = {"alpha": r"$\alpha$", "mu": r"$\mu$",
           "sigma": r"$\Sigma$", "pi": r"$\pi$"}
HATCHES = {"SMLE": "", "ORACLE": "///", "FINAL": "...", "GMM": ""}
COLORS = {"SMLE": "#111111", "ORACLE": "#666666", "FINAL": "#999999"}
LINESTYLES = {"SMLE": (0, (5, 2)), "ORACLE": (0, (6, 2, 1.5, 2)), "FINAL": (0, (1, 2))}
MARKERS = {"SMLE": "x", "ORACLE": "o", "FINAL": "^"}


def load_results(directory):
    """Reject missing/duplicate/failed cells instead of silently selecting data."""
    directory = Path(directory)
    config = json.loads((directory / "configuration.json").read_text(encoding="utf-8"))
    reps = config["replications"]
    ns = config["n_grid"]
    if isinstance(reps, bool) or not isinstance(reps, int) or reps < 1:
        raise ValueError("configuration.replications must be a positive integer")
    if (not isinstance(ns, list) or not ns
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in ns)
            or ns != sorted(set(ns))):
        raise ValueError("configuration.n_grid must be nonempty, sorted, unique positive integers")
    with (directory / "all_results.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"replication", "n", "estimator", "status", "log_mse_zeta"}
        required.update(f"mse_{p}" for p in PARAMETERS)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing result columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    expected = {(rep, n, method) for rep in range(1, reps + 1) for n in ns for method in ALL_METHODS}
    seen, groups, source_values = set(), {}, []
    for row in rows:
        key = (int(row["replication"]), int(row["n"]), row["estimator"])
        if key not in expected:
            raise ValueError(f"Unexpected result cell: {key}")
        if key in seen:
            raise ValueError(f"Duplicate result cell: {key}")
        seen.add(key)
        if row["status"] != "ok":
            raise ValueError(f"Non-ok result cell: {key}, status={row['status']}")
        rep, n, method = key
        if method in PARAMETER_METHODS:
            for parameter in PARAMETERS:
                value = float(row[f"mse_{parameter}"])
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"Nonpositive/nonfinite parameter MSE: {key}, {parameter}")
                log_value = float(np.log(value))
                groups.setdefault((parameter, n, method), []).append(log_value)
                source_values.append(dict(replication=rep, n=n, estimator=method,
                                          parameter=parameter, log_mse=log_value))
        if method in CLUSTERING_METHODS:
            # Already stable log-domain values: never take log(float mse_zeta).
            value = float(row["log_mse_zeta"])
            if not np.isfinite(value) or value > 0:
                raise ValueError(f"Invalid log posterior MSE: {key}")
            groups.setdefault(("zeta", n, method), []).append(value)
            source_values.append(dict(replication=rep, n=n, estimator=method,
                                      parameter="zeta", log_mse=value))
    missing = expected - seen
    if missing:
        raise ValueError(f"Incomplete results: {len(missing)} missing cells; first={sorted(missing)[0]}")
    source_values.sort(key=lambda r: (r['parameter'], r['n'], r['estimator'], r['replication']))
    return config, ns, groups, source_values


def _axes_style(ax):
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3, width=.7)
    ax.grid(True, color="#d8d8d8", linewidth=.45, linestyle=(0, (3, 2)))


def draw_boxes(ax, parameter, ns, groups, summary):
    for ni, n in enumerate(ns):
        for mi, method in enumerate(PARAMETER_METHODS):
            values = groups[parameter, n, method]
            stats = boxplot_stats(values, whis=1.5)[0]
            drawn = ax.bxp([stats], positions=[ni + (-.24, 0, .24)[mi]], widths=.19,
                           patch_artist=True, showfliers=True, manage_ticks=False,
                           boxprops=dict(facecolor="white", edgecolor="black", linewidth=.8),
                           medianprops=dict(color="black", linewidth=1.15),
                           whiskerprops=dict(color="black", linewidth=.7),
                           capprops=dict(color="black", linewidth=.7),
                           flierprops=dict(marker="o", markerfacecolor="white", markeredgecolor="black",
                                           markersize=2.2, markeredgewidth=.6))
            drawn["boxes"][0].set_hatch(HATCHES[method])
            np.testing.assert_array_equal(drawn["fliers"][0].get_ydata(), stats["fliers"])
            summary.append(dict(parameter=parameter, n=n, estimator=method,
                                replications=len(values), q1=stats['q1'], median=stats['med'],
                                q3=stats['q3'], lower_whisker=stats['whislo'],
                                upper_whisker=stats['whishi'], outliers=len(stats['fliers'])))
    ax.set_xlim(-.5, len(ns) - .5)
    ax.set_xticks(range(len(ns)), [str(n) for n in ns], rotation=30, ha="right")
    ax.set_xlabel(r"Sample size $n$")
    ax.set_ylabel("log(MSE)")
    ax.legend(handles=[Patch(facecolor="white", edgecolor="black", hatch=HATCHES[m], label=m)
                       for m in PARAMETER_METHODS], loc="best", fontsize=6.5,
              frameon=True, facecolor="white", edgecolor="#dddddd", handlelength=1.1,
              borderpad=.3, labelspacing=.25)
    _axes_style(ax)


def draw_trend(ax, parameter, ns, groups, trend_values, slopes):
    x = np.log(np.asarray(ns, dtype=float))
    for method in PARAMETER_METHODS:
        y = np.array([np.median(groups[parameter, n, method]) for n in ns])
        label = method
        if len(ns) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            grid = np.linspace(x[0], x[-1], 256)
            ax.plot(grid, intercept + slope * grid, color=COLORS[method],
                    linestyle=LINESTYLES[method], linewidth=1.2)
            label = f"{method}: {slope:.2f}"
            slopes.append(dict(parameter=parameter, estimator=method,
                               slope=float(slope), intercept=float(intercept)))
        ax.plot(x, y, linestyle="none", color=COLORS[method], marker=MARKERS[method],
                markerfacecolor="white", markersize=4.5, markeredgewidth=1., label=label)
        for n, xx, yy in zip(ns, x, y):
            trend_values.append(dict(parameter=parameter, n=n, estimator=method,
                                     log_n=float(xx), median_log_mse=float(yy)))
    ax.set_xlabel("log(n)")
    ax.set_ylabel("Median of log(MSE)")
    ax.legend(loc="best", title="Fitted slope" if len(ns) > 1 else None,
              fontsize=6.5, title_fontsize=6.5, frameon=True,
              facecolor="white", edgecolor="#dddddd", borderpad=.3, labelspacing=.25)
    _axes_style(ax)


def draw_clustering(ns, groups):
    fig, ax = plt.subplots(figsize=(183 / 25.4, 110 / 25.4), layout="constrained")
    x = np.arange(len(ns), dtype=float)
    styles = {"GMM": dict(facecolor="#bdbdbd", hatch=""),
              "SMLE": dict(facecolor="white", hatch=""),
              "FINAL": dict(facecolor="white", hatch="...")}
    medians = {method: np.array([np.median(groups['zeta', n, method]) for n in ns])
               for method in CLUSTERING_METHODS}
    records = []
    for offset, method in zip((-.28, 0, .28), CLUSTERING_METHODS):
        bars = ax.bar(x + offset, medians[method], width=.23, bottom=0,
                      edgecolor="black", linewidth=.9, zorder=3, label=method, **styles[method])
        for n, bar, value in zip(ns, bars, medians[method]):
            if bar.get_height() != value or bar.get_y() != 0:
                raise AssertionError("Bar height or baseline differs from source summary")
            records.append(dict(parameter="zeta", n=n, estimator=method,
                                replications=len(groups['zeta', n, method]), median_log_mse=float(value)))
            ax.annotate(f"{value:.2f}" if method == "GMM" else f"{value:,.0f}",
                        (bar.get_x() + bar.get_width() / 2, value), xytext=(0, -5),
                        textcoords="offset points", ha="center", va="top", fontsize=7)
    minimum = min(float(values.min()) for values in medians.values())
    span = max(abs(minimum), 1.)
    ax.set_ylim(minimum - .14 * span, .07 * span)
    ax.axhline(0, color="black", linewidth=.7)
    ax.set_xticks(x, [str(n) for n in ns])
    ax.set_xlabel(r"Sample size $n$")
    ax.set_ylabel(r"Median of $\log(\mathrm{MSE}_{\zeta})$")
    ax.legend(loc="lower left", fontsize=7, frameon=True, facecolor="white", edgecolor="#dddddd")
    _axes_style(ax)
    # Keep the same values visible when GMM is compressed by large log errors.
    zoom = span > 50 * max(float(np.abs(medians['GMM']).max()), 1.)
    if zoom:
        inset = ax.inset_axes([.16, .34, .37, .24], zorder=10)
        inset.bar(x, medians['GMM'], width=.44, bottom=0, edgecolor="black",
                  linewidth=.8, **styles['GMM'])
        gmm_min = float(medians['GMM'].min())
        inset.set_ylim(gmm_min - .18 * max(abs(gmm_min), 1.), .08 * max(abs(gmm_min), 1.))
        inset.set_title("GMM (separate y-axis)", fontsize=7.5)
        inset.set_xticks(x, [str(n) for n in ns])
        inset.tick_params(labelsize=6)
        _axes_style(inset)
    return fig, records, bool(zoom)


def _write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _export(fig, output, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    all_axes = [axis for ax in fig.axes for axis in (ax, *ax.child_axes)]
    for ax in all_axes:
        legend = ax.get_legend()
        legend_text = legend.get_texts() if legend else []
        for label in (ax.xaxis.label, ax.yaxis.label, ax.title, ax._left_title,
                      *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.texts, *legend_text):
            if label.get_visible() and label.get_text():
                box = label.get_window_extent(renderer)
                # Matplotlib also creates labels for ticks outside axis limits.
                if label in ax.get_yticklabels():
                    if not ax.get_ylim()[0] <= label.get_position()[1] <= ax.get_ylim()[1]:
                        continue
                if label in ax.get_xticklabels():
                    if not ax.get_xlim()[0] <= label.get_position()[0] <= ax.get_xlim()[1]:
                        continue
                if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width + .5 or box.y1 > fig.bbox.height + .5:
                    raise RuntimeError(f"Figure text falls outside canvas: {stem}: {label.get_text()}")
    fig.savefig(output / f"{stem}.pdf", metadata={"CreationDate": None})
    fig.savefig(output / f"{stem}.svg", metadata={"Date": None})
    fig.savefig(output / f"{stem}.png", dpi=300)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Directory with configuration.json and all_results.csv")
    parser.add_argument("--output", type=Path, required=True, help="Directory for newly generated figures and summary tables")
    args = parser.parse_args(argv)
    config, ns, groups, source_values = load_results(args.results)
    input_hash = hashlib.sha256((args.results / "all_results.csv").read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "axes.linewidth": .7, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "hatch.linewidth": .5, "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    summary, trends, slopes = [], [], []
    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 142 / 25.4), layout="constrained")
    for col, parameter in enumerate(PARAMETERS[:3]):
        draw_boxes(axes[0, col], parameter, ns, groups, summary)
        draw_trend(axes[1, col], parameter, ns, groups, trends, slopes)
        axes[0, col].set_title(f"{chr(97 + col)}  {SYMBOLS[parameter]}", loc="left")
        axes[1, col].set_title(f"{chr(100 + col)}  {SYMBOLS[parameter]}", loc="left")
    _export(fig, args.output, "figure1_parameter_errors")
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 83 / 25.4), layout="constrained")
    draw_boxes(axes[0], "pi", ns, groups, summary)
    draw_trend(axes[1], "pi", ns, groups, trends, slopes)
    axes[0].set_title(r"a  $\pi$", loc="left")
    axes[1].set_title(r"b  $\pi$", loc="left")
    _export(fig, args.output, "figure2_edge_probability")
    fig, medians, zoom = draw_clustering(ns, groups)
    _export(fig, args.output, "figure3_clustering")
    _write_csv(args.output / "plotted_values.csv", source_values)
    _write_csv(args.output / "boxplot_statistics.csv", summary)
    _write_csv(args.output / "trend_points.csv", trends)
    _write_csv(args.output / "fitted_slopes.csv", slopes)
    _write_csv(args.output / "clustering_medians.csv", medians)
    if input_hash != hashlib.sha256((args.results / "all_results.csv").read_bytes()).hexdigest():
        raise RuntimeError("Input result table changed during plotting")
    audit = dict(status="numerically_checked", source_sha256=input_hash,
                 n_grid=ns, replications_per_cell=config["replications"],
                 all_cells_complete_and_ok=True, all_outliers_retained=True,
                 log_base="natural", trend_statistic="median(log(MSE))",
                 clustering_statistic="median(stable log posterior MSE)",
                 clustering_bar_baseline=0, gmm_separate_axis_inset=zoom,
                 pixel_identical_pdf_claim=False, visual_review="Inspect generated PNG previews")
    (args.output / "plot_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
