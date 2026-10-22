#!/usr/bin/env python3
"""Matched-split grayscale boxplots from metrics.csv; no model fitting or labels.

Figure contract: compare distributions, without assuming a winning method.
Quantitative comparison: per-split natural log posterior MSE_zeta. Complete
matched splits and a fixed method order are used. PDF/SVG have editable text;
PNG is a high-resolution preview. Synthetic checks must use --synthetic.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

METRICS = ("log_MSE_zeta",)
SEMANTIC_FIELDS = (
    "score_type", "alignment", "alignment_source", "mapping_rule",
    "unseen_cluster_policy", "score_aggregation",
)
DISPLAY_NAMES = {
    "GMM": "GMM", "GM3L_SMLE": "GM³L (SMLE)", "GM3L_FINAL": "GM³L (FINAL)",
    "attSBM2019": "attSBM", "CSBM_IRMAP2022": "CSBM-IRMAP", "BCDC2025": "BCDC",
    "CLSBM2025": "CLSBM*", "SSC_IPA2022": "SSC-IPA", "NSBM2022": "NSBM",
}


def read_matched_metrics(path: str | Path, methods: list[str] | None = None) -> dict[str, Any]:
    """Validate all rows, then use the full common-complete split intersection."""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"split_id", "method", "status", "MSE_zeta_status", *METRICS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"metrics.csv missing columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    if not rows:
        raise ValueError("metrics.csv has no records")
    available = list(dict.fromkeys(row["method"] for row in rows))
    methods = available if methods is None else list(methods)
    if not methods or len(set(methods)) != len(methods) or not set(methods).issubset(available):
        raise ValueError("Methods must be unique and all present in metrics.csv")
    selected = [row for row in rows if row["method"] in methods]
    index = {(row["split_id"], row["method"]): row for row in selected}
    if len(index) != len(selected):
        raise ValueError("Duplicate split/method rows are not allowed")
    split_ids = sorted(set(row["split_id"] for row in selected))
    protocols = {row.get("protocol", "") for row in selected if row.get("protocol", "")}
    if len(protocols) > 1:
        raise ValueError("Do not combine different protocols in one figure")
    for row in selected:
        if row["status"] == "complete":
            location = f"split={row['split_id']}, method={row['method']}"
            if row["MSE_zeta_status"] == "zero_in_input":
                raise ValueError(
                    f"Zero posterior MSE_zeta in the supplied input (zero_in_input; {location}) has "
                    "log_MSE_zeta = -inf and cannot be shown on a finite boxplot. "
                    "If probability exports were rounded, re-evaluate native log-probability/logit exports. "
                    "If truly exact zero, use an explicit-zero visualization; "
                    "preserve this result without flooring the value or dropping the split."
                )
            if row["MSE_zeta_status"] not in {"positive", "underflow"}:
                raise ValueError(f"Complete row has invalid MSE_zeta_status ({location})")
            for metric in METRICS:
                try:
                    value = float(row[metric])
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Complete row has missing/non-numeric {metric} ({location})") from exc
                if not math.isfinite(value) or value > 0:
                    raise ValueError(f"Complete row requires finite {metric} <= 0 ({location})")
                row[metric] = value
    common = [split for split in split_ids if all(index.get((split, method), {}).get("status") == "complete" for method in methods)]
    if not common:
        raise ValueError("No jointly complete splits; no comparison figure can be drawn")
    # Explicitly list missing cells, even if a hand-created CSV omitted them.
    exclusions = []
    for split in split_ids:
        if split in common:
            continue
        for method in methods:
            row = index.get((split, method))
            exclusions.append({"split_id": split, "method": method,
                               "status": row["status"] if row else "missing",
                               "reason": row.get("reason", "") if row else "No metrics row"})
    for method in methods:
        signatures = {tuple(index[(split, method)].get(key, "") for key in SEMANTIC_FIELDS) for split in common}
        if len(signatures) > 1:
            raise ValueError(f"Score semantics change across paired splits for {method}")
    return {"methods": methods, "requested_split_ids": split_ids, "common_valid_split_ids": common,
            "paired_n": len(common), "excluded_split_records": exclusions,
            "rows": [index[(split, method)] for split in common for method in methods],
            "protocol": next(iter(protocols), None)}


def _csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_results(
    metrics_path: str | Path, output_dir: str | Path, methods: list[str] | None = None,
    *, synthetic: bool = False, dpi: int = 600,
) -> dict[str, Any]:
    """Plot recorded per-split log MSE values, without recomputing or flooring them."""
    if not 150 <= dpi <= 600:
        raise ValueError("dpi must lie between 150 and 600")
    cohort = read_matched_metrics(metrics_path, methods)
    # Importing the evaluator is unnecessary: plotting never opens y or models.
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    methods = cohort["methods"]
    metrics = METRICS
    index = {(row["split_id"], row["method"]): row for row in cohort["rows"]}
    ids = cohort["common_valid_split_ids"]
    tables = []
    for method in methods:
        item = {"method": method, "display_name": DISPLAY_NAMES.get(method, method), "paired_n": len(ids)}
        for metric in METRICS:
            values = np.asarray([index[(split, method)][metric] for split in ids])
            for statistic, value in (("mean", values.mean()), ("median", np.median(values)),
                                     ("q1", np.quantile(values, .25)), ("q3", np.quantile(values, .75)),
                                     ("minimum", values.min()), ("maximum", values.max())):
                item[f"{metric}_{statistic}"] = float(value)
            item[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
        tables.append(item)
    styles = {
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
        "pdf.fonttype": 42, "svg.fonttype": "none", "text.usetex": False,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    }
    with mpl.rc_context(styles):
        width = max(3.5, min(7.2, .45 * len(methods) + 1.2))
        fig, axes_array = plt.subplots(1, len(metrics), figsize=(width, 3.45), squeeze=False)
        axes = axes_array.ravel()
        fig.subplots_adjust(left=.13, right=.985, bottom=.34, top=.80,
                            wspace=.34)
        all_within = True
        medians_correct = True
        labels = [DISPLAY_NAMES.get(method, method) for method in methods]
        for letter, metric, ax in zip("abc", metrics, axes):
            values = [np.asarray([index[(split, method)][metric] for split in ids]) for method in methods]
            box = ax.boxplot(values, positions=np.arange(1, len(methods) + 1), widths=.62,
                             patch_artist=True, showfliers=True, whis=1.5,
                             boxprops={"edgecolor": "#222222", "linewidth": .75},
                             medianprops={"color": "black", "linewidth": 1.2},
                             whiskerprops={"color": "#222222", "linewidth": .7},
                             capprops={"color": "#222222", "linewidth": .7},
                             flierprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "#222222",
                                         "markersize": 2.2, "markeredgewidth": .6})
            for patch in box["boxes"]:
                patch.set_facecolor("#D6D6D6")
            ax.set_xticks(np.arange(1, len(methods) + 1), labels, rotation=52, ha="right", rotation_mode="anchor")
            ax.set_xlim(.4, len(methods) + .6)
            combined = np.concatenate(values)
            margin = max(float(np.ptp(combined)) * .10, float(np.max(np.abs(combined))) * .01, .025)
            # The recorded natural logs are already transformed; use an unclamped linear axis.
            lower, upper = float(combined.min()) - margin, float(combined.max()) + margin
            ax.set_ylim(lower, upper)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
            ax.set_ylabel(r"$\log(\mathrm{MSE}_{\zeta})$")
            ax.set_title(letter, loc="left", fontweight="bold", pad=9)
            ax.tick_params(axis="x", length=0, pad=5)
            ax.tick_params(axis="y", direction="out", length=2.5, width=.6, pad=3)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E1E1E1", linewidth=.45)
            all_within &= bool(np.all(combined > lower) and np.all(combined < upper))
            medians_correct &= all(np.isclose(line.get_ydata()[0], np.median(v)) for line, v in zip(box["medians"], values))
        heading = "Synthetic validation — not paper results" if synthetic else "Matched-split method comparison"
        fig.text(.5, .965, heading, ha="center", va="center", fontsize=8, fontweight="bold")
        fig.text(.5, .91, f"{len(ids)} common valid splits / {len(cohort['requested_split_ids'])} requested", ha="center", fontsize=7, color="#444444")
        fig.text(.5, .10, "Median · Q1–Q3 · whiskers: 1.5 × IQR · all outliers retained", ha="center", fontsize=6.5, color="#444444")
        fig.text(.5, .06, "Overlapping partitions; spread is not an independent-sample confidence interval", ha="center", fontsize=6.0, color="#444444")
        if "CLSBM2025" in methods:
            fig.text(.5, .02, "* CLSBM: full-graph spectral representation; centers fitted on training nodes.",
                     ha="center", fontsize=6.0, color="#444444")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        artists = list(fig.texts)
        for ax in axes:
            artists += list(ax.get_xticklabels()) + list(ax.get_yticklabels()) + [ax.yaxis.label, ax._left_title]
        bounds = [artist.get_window_extent(renderer) for artist in artists if artist.get_visible() and artist.get_text()]
        canvas = fig.bbox
        text_inside = all(b.x0 >= canvas.x0 - 1 and b.y0 >= canvas.y0 - 1 and b.x1 <= canvas.x1 + 1 and b.y1 <= canvas.y1 + 1 for b in bounds)
        if not all_within or not medians_correct:
            plt.close(fig)
            raise AssertionError("Numerical figure integrity check failed")
        # Tight bounding boxes preserve arbitrary long external method labels.
        stem = output / "log_mse_comparison"
        if synthetic:
            stem = stem.with_name("synthetic_" + stem.name)
        exports = []
        for extension in ("pdf", "svg", "png"):
            destination = stem.with_suffix("." + extension)
            fig.savefig(destination, dpi=dpi, bbox_inches="tight", pad_inches=.04)
            exports.append(destination)
        plt.close(fig)

    source_fields = ["split_id", "seed", "protocol", "method", *METRICS,
                     "MSE_zeta_status", "status", *SEMANTIC_FIELDS, "trace_key"]
    _csv(output / "plot_source_data.csv", cohort["rows"], source_fields)
    _csv(output / "plot_summary.csv", tables, list(tables[0]))
    lines = ["# " + heading, "", f"Common paired splits: {len(ids)} / {len(cohort['requested_split_ids'])}.", "",
             "| Method | n | Mean log(MSE_zeta) | Median log(MSE_zeta) |", "|---|---:|---:|---:|"]
    for item in tables:
        lines.append("| " + " | ".join([item["display_name"], str(item["paired_n"])] +
                     [f"{item['log_MSE_zeta_' + statistic]:.6f}" for statistic in ("mean", "median")]) + " |")
    lines.extend(["", "All summaries use identical splits. No method-specific or best-seed filtering is performed.",
                  "The metric is the recorded per-split natural log(MSE_zeta), with lower values indicating lower posterior MSE.",
                  "Means, medians and quartiles summarize these replicate log values, not log(mean(MSE_zeta)).",
                  "Raw MSE underflow does not remove a finite log value. Exact input zeros stop plotting; no flooring is applied.",
                  "Box = Q1–Q3; center = median; whiskers = 1.5 × IQR; open dots = all outliers.",
                  "No significance tests or confidence intervals are shown. Partitions may overlap."])
    excluded_ids = sorted(set(row["split_id"] for row in cohort["excluded_split_records"]))
    if excluded_ids:
        lines.extend([f"Excluded from every method: {', '.join(excluded_ids)}.",
                      "Per-method exclusion statuses and reasons are recorded in plot_manifest.json."])
    if "CLSBM2025" in methods:
        lines.append("* CLSBM uses the full graph for its spectral representation; clustering centers are fitted on training nodes.")
    (output / "plot_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {key: value for key, value in cohort.items() if key != "rows"}
    manifest.update({"synthetic": synthetic, "purpose": heading, "panels": list(metrics), "backend": "Python/matplotlib",
                     "archetype": "quantitative grid", "metrics_sha256": _digest(Path(metrics_path)),
                     "metric_definition": "Per-split natural log posterior MSE_zeta; read directly from log_MSE_zeta",
                     "summary_definition": "Statistics of replicate log_MSE_zeta values, not log of mean MSE_zeta",
                     "axis_scale": "linear", "zero_policy": "Exact input zeros stop plotting; finite underflow logs retained; no flooring",
                     "box_definition": "Q1–Q3, median, 1.5 IQR whiskers, all outliers", "summaries": tables,
                     "no_model_fits": True, "no_human_labels_read": True, "no_metric_recomputation": True,
                     "quantitative_qa": {"all_observations_within_axes": all_within,
                                         "median_artists_match_values": medians_correct,
                                         "text_inside_untightened_canvas": text_inside,
                                         "export_uses_tight_bounding_box": True},
                     "visual_qa": "Inspect the exported PNG at final size; numerical checks do not replace visual review.",
                     "exports": {path.name: {"sha256": _digest(path), "bytes": path.stat().st_size} for path in exports}})
    (output / "plot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--synthetic", action="store_true", help="Mark synthetic validation clearly; never present it as paper results")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args(argv)
    try:
        result = plot_results(args.metrics, args.output, args.methods, synthetic=args.synthetic, dpi=args.dpi)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"No figure created: {exc}\n")
    print(json.dumps({"paired_n": result["paired_n"], "output": str(args.output), "synthetic": result["synthetic"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
