"""Evaluation-only parameter deviations from the training-label oracle.

These are deviations between estimators, not errors against unknown population
parameters. True labels are used here only after model fitting is complete.
"""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

from .model import final_from_zeta
from .runner import write_json
from .validation import completion_error, fingerprint, verify_input_identity


BLOCKS = ("alpha", "mu", "sigma", "pi")


def training_oracle(X, A, y, K):
    y = np.asarray(y)
    if y.shape != (len(X),) or not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= K)):
        raise ValueError("Training labels must be integer classes 0,...,K-1")
    counts = np.bincount(y, minlength=K)
    if np.any(counts < 2):
        raise ValueError("Training oracle undefined: every class needs at least two nodes")
    theta = final_from_zeta(X, A, np.eye(K)[y])
    eigenvalues = np.linalg.eigvalsh(theta["sigma"])
    return theta, dict(counts=counts.tolist(), min_covariance_eigenvalue=float(eigenvalues.min()),
                        positive_definite=bool(np.all(eigenvalues > 0)),
                        interior_pi=bool(np.all((theta["pi"] > 0) & (theta["pi"] < 1))))


def soft_order(log_scores, y_train, K):
    if log_scores.shape != (len(y_train), K) or np.isnan(log_scores).any():
        raise ValueError("Invalid training alignment scores")
    weights = np.exp(log_scores-logsumexp(log_scores, axis=1, keepdims=True))
    mass = np.asarray([weights[y_train == k].sum(axis=0) for k in range(K)])
    rows, cols = linear_sum_assignment(-mass)
    order = np.empty(K, int)
    order[rows] = cols
    return order


def parameter_deviations(theta, oracle, order):
    K = len(oracle["alpha"])
    if sorted(np.asarray(order).tolist()) != list(range(K)):
        raise ValueError("Alignment must be a permutation of all K components")
    for block in BLOCKS:
        if theta[block].shape != oracle[block].shape or not np.isfinite(theta[block]).all():
            raise ValueError("Mismatched or nonfinite parameter array: " + block)
    aligned = dict(alpha=theta["alpha"][order], mu=theta["mu"][order],
                   sigma=theta["sigma"][order], pi=theta["pi"][np.ix_(order, order)])
    # Historical real-data figure uses full array entries, not vech weighting.
    return {block: float(np.mean((aligned[block]-oracle[block])**2)) for block in BLOCKS}


def evaluate_parameters(prepared, runs, output, methods=("SMLE", "FINAL")):
    prepared, runs, output = Path(prepared), Path(runs), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty parameter-diagnostics output directory")
    rows, oracle_checks = [], {}
    # Discover requested split directories even when their config is damaged.
    # Config-based discovery silently drops precisely those incomplete splits.
    splits = sorted(path for path in prepared.glob("split_*") if path.is_dir())
    if not splits:
        raise ValueError("No prepared splits")
    protocols = set()
    for split in splits:
        protocol = ""
        try:
            cfg = json.loads((split/"input"/"config.json").read_text(encoding="utf-8"))
            if not isinstance(cfg, dict) or cfg.get("protocol") not in {"inductive", "transductive"}:
                raise ValueError("Specify inductive or transductive protocol")
            protocol = cfg["protocol"]
            protocols.add(protocol)
            identity = fingerprint(split/"input")
            with np.load(split/"evaluation_only.npz", allow_pickle=False) as evaluation:
                y = evaluation["y_train"]
            X = np.load(split/"input"/"X_train.npy", allow_pickle=False)
            A = np.load(split/"input"/"A_train.npy", allow_pickle=False)
            oracle, check = training_oracle(X, A, y, cfg["K"])
            oracle_checks[split.name] = check
            oracle_error = None
        except Exception as exc:
            oracle_error = f"{type(exc).__name__}: {exc}"
            oracle_checks[split.name] = dict(status="undefined", reason=oracle_error)
        for method in methods:
            row = dict(split_id=split.name, method=method, protocol=protocol, status="failed", reason="")
            try:
                if oracle_error:
                    raise ValueError(oracle_error)
                run = runs/split.name/method
                diagnostic = json.loads((run/"diagnostics.json").read_text(encoding="utf-8"))
                verify_input_identity(diagnostic, identity)
                error = completion_error(diagnostic)
                if error:
                    raise ValueError(error)
                meta = json.loads((run/"score_columns.json").read_text(encoding="utf-8"))
                if meta.get("alignment") != "soft" or meta.get("score_type") != "log_probability" or meta.get("cluster_ids") != list(range(cfg["K"])):
                    raise ValueError("Parameter comparison requires native ordered log-posterior columns")
                if method in {"SMLE", "FINAL"} and not meta.get("alignment_source", "").startswith("SMLE training posterior"):
                    raise ValueError("SMLE/FINAL must use the shared SMLE training alignment source")
                train = np.loadtxt(run/"train_scores.csv", delimiter=",", ndmin=2)
                order = soft_order(train, y, cfg["K"])
                with np.load(run/"model.npz", allow_pickle=False) as archive:
                    theta = {key: archive[key] for key in BLOCKS}
                errors = parameter_deviations(theta, oracle, order)
                if not all(np.isfinite(v) and v >= 0 for v in errors.values()):
                    raise ValueError("Nonfinite parameter deviation")
                row.update(status="complete", **errors)
                for key, value in errors.items():
                    row["log_"+key] = float(np.log(value)) if value > 0 else None
            except Exception as exc:
                row["reason"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    if len(protocols) > 1:
        raise ValueError("Do not combine different prediction protocols")
    output.mkdir(parents=True, exist_ok=True)
    fields = ["split_id", "method", "protocol", "status", "reason", *BLOCKS, *("log_"+key for key in BLOCKS)]
    with (output/"parameter_errors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader(); writer.writerows(rows)
    write_json(output/"oracle_diagnostics.json", oracle_checks)
    return rows


def plot_parameters(csv_path, output, methods=("SMLE", "FINAL"), *, synthetic=False):
    """Four-panel parameter-deviation plot; all exact zeros are reported."""
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Specify unique methods")
    selected = [row for row in rows if row["method"] in methods]
    index = {(row["split_id"], row["method"]): row for row in selected}
    if len(index) != len(selected):
        raise ValueError("Duplicate split/method rows")
    valid_protocols = {"inductive", "transductive"}
    if any(row.get("status") == "complete" and row.get("protocol") not in valid_protocols for row in selected):
        raise ValueError("Complete parameter comparisons need a valid prediction protocol")
    # A failed split may have no readable config; keep it in the requested
    # denominator without treating its unknown protocol as a second protocol.
    if len({row.get("protocol") for row in selected if row.get("protocol") in valid_protocols}) > 1:
        raise ValueError("Do not combine different protocols")
    splits = sorted({key[0] for key in index})
    common = [s for s in splits if all(index.get((s,m),{}).get("status")=="complete" for m in methods)]
    if not common:
        raise ValueError("No common complete parameter comparisons")
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    zeros = {}
    with mpl.rc_context({"font.size":8, "pdf.fonttype":42, "svg.fonttype":"none", "axes.spines.top":False, "axes.spines.right":False}):
        fig, axes = plt.subplots(1,4,figsize=(7.2,3.15))
        fig.subplots_adjust(left=.085,right=.99,bottom=.30,top=.73,wspace=.55)
        for letter,key,ax in zip("abcd",BLOCKS,axes):
            values=[]
            for method in methods:
                raw=np.array([float(index[(s,method)][key]) for s in common])
                if not np.isfinite(raw).all() or np.any(raw<0):
                    raise ValueError("Invalid parameter error")
                zeros[f"{key}/{method}"]=int((raw==0).sum())
                values.append(np.log(raw[raw>0]))
            labels=[f"{method}\nn+={len(v)}\n0={zeros[f'{key}/{method}']}" for method,v in zip(methods,values)]
            box=ax.boxplot(values,labels=labels,patch_artist=True,widths=.55,showfliers=True,
                           medianprops={"color":"black"},flierprops={"markersize":2.5})
            for patch in box["boxes"]:patch.set_facecolor("#d6d6d6")
            ax.set_title(f"{letter}  "+{"alpha":r"$\alpha$","mu":r"$\mu$","sigma":r"$\Sigma$","pi":r"$\Pi$"}[key],loc="left",fontweight="bold")
            ax.tick_params(axis="both",labelsize=7)
            ax.set_ylabel("Log squared deviation",fontsize=7)
            for j,v in enumerate(values,1):
                if len(v)==0:
                    ax.text(j,.5,"All zero",transform=ax.get_xaxis_transform(),ha="center",va="center",rotation=90,fontsize=7)
        heading="Synthetic validation — not paper results" if synthetic else "Deviation from the training-label oracle"
        fig.suptitle(heading,fontsize=10,y=.98)
        fig.text(.5,.86,f"{len(common)} common complete splits / {len(splits)} requested",ha="center",fontsize=8)
        fig.text(.5,.08,"Boxes: finite-positive subset (n+); 0: exact-zero count, log undefined",ha="center",fontsize=7)
        fig.text(.5,.025,"Full-array mean squared deviations; no zero floor or probability clipping",ha="center",fontsize=7)
        for ext in ("pdf","svg","png"):
            fig.savefig(output/("parameter_deviations."+ext),dpi=300,facecolor="white")
        plt.close(fig)
    write_json(output/"parameter_plot_manifest.json",dict(common_splits=common,methods=list(methods),
               zero_counts=zeros,synthetic=synthetic,reference="training-label oracle, not population truth"))
