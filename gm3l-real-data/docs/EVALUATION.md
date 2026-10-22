# Evaluation and figure contract

This package contains **code, not empirical results**. No real data, fitted output,
existing figure, result table, or evaluation cache is bundled. Synthetic tests
check the implementation only; they do not establish a method ranking or reproduce
the paper's numerical results.

## Label isolation and inputs

Only the evaluation stage reads human class labels. Model fitting and prediction
must not import this evaluator or open `evaluation_only.npz`. The evaluator never
fits models, chooses restarts, selects seeds, changes predictions, or opens input
features/graphs. It reads the following explicitly exported interface:

```text
prepared/split_000/
  input/config.json            # K, seed, protocol
  evaluation_only.npz           # y_train and y_test, coded 0,...,K-1
runs/split_000/METHOD/
  diagnostics.json              # completion/failure, runtime, convergence
  train_labels.csv              # one integer native cluster ID per row
  test_labels.csv               # one integer native cluster ID per row
  test_scores.csv               # continuous scores, samples × native clusters
  score_columns.json            # score/alignment semantics
  train_scores.csv              # required for soft alignment only
```

CSV files have no header and no index column. The score column order must agree
exactly with `cluster_ids`. Hard cluster IDs must appear in those columns. All
training classes must be present for training alignment to be defined. The
expected class set is determined by `K`, never by looking only at held-out labels.

`protocol` must be explicit. In particular, **full-graph transductive prediction**
and **training-frozen inductive prediction** are different experiments. Their
records are retained but cannot be merged into one summary/figure. No assumption
that using full observed X/A is equivalent to training-only inference is made.

## Score metadata: preserve native definitions

A typical GM³L/GMM export declares:

```json
{
  "cluster_ids": [0, 1, 2],
  "score_type": "log_probability",
  "alignment": "soft",
  "alignment_source": "SMLE training posterior",
  "mapping": "hungarian",
  "unseen_cluster_policy": "error"
}
```

The example has three clusters only to illustrate the schema; use the fitted
model's actual IDs and configured K. `cluster_ids` and `score_type` are required;
extra provenance fields are permitted.

| Field | Definition |
|---|---|
| `score_type` | `probability`, `log_probability`, `logits`, `log_likelihood`, `log_score`, `vote_fraction`, or native ranking scores (`decision`, `native`, `negative_distance`, `distance_score`, `confidence`) |
| `alignment` | `hard` (default) or `soft`; soft requires native training scores and a declared probabilistic/log score type |
| `mapping` | `hungarian` (default) or `majority` |
| `alignment_source` | Optional provenance; especially important when FINAL intentionally reuses the SMLE training mapping |
| `unseen_cluster_policy` | `error` (default) or explicit `training_prior` for probability/log scores |
| `score_aggregation` | Optional native-score provenance; not needed to evaluate posterior MSE or auxiliary ACC |

For `probability`, scores must be nonnegative and sum to one per row.
`vote_fraction` is a native ranking score, not automatically a probability:
DipEncoder's pairwise vote fractions sum to K/2, not one, and are never used
for MSE_zeta or soft alignment. Log score rows must have at least one finite entry. `-inf` can represent
zero probability; NaN and `+inf` inputs are rejected. Raw distance/ranking scores
must be finite. No generic softmax is applied to a distance score to invent a
posterior. `log_likelihood` and `log_score` can support a declared soft training
alignment, but are not themselves a declaration of class posterior probabilities.

### Training-only alignment

- **Hard Hungarian:** maximize overlap of native training allocations with the
  training truth. A fixed-K export gives a bijection. Rectangular Hungarian is
  also available to preserve a declared variable-K/native protocol.
- **Soft Hungarian:** normalize the declared native training log scores (or use
  probabilities) and maximize sums of training posterior membership by class.
  This can differ from hard-label contingency and is not replaced by it.
- **Unknown-K majority:** map each occupied training cluster to its most frequent
  training class; ties use the lowest class ID. Multiple clusters may map to one
  class.
- **FINAL alignment reuse:** the original GM³L FINAL evaluation reuses the SMLE
  soft training mapping. Its `train_scores.csv` therefore contains SMLE training
  scores, not a newly estimated FINAL alignment. FINAL held-out scores remain
  FINAL scores; metadata records this distinction.

Test truth is not an argument to `training_mapping`. Changing only test labels
cannot change the mapping. Unmapped hard clusters count as ACC errors under the
default policy. Under an explicitly declared `training_prior` policy, unmapped
hard predictions use the training majority class and unmapped probability mass
is redistributed with the training class prior; test frequencies are never used.

### Aligned posterior probabilities

For `probability`, `log_probability`, and `logits`, the evaluator works with
normalized log probabilities. A one-to-one training mapping only reorders the
posterior columns. For many-to-one mapping, posterior mass is combined by
log-sum-exp, equivalent to summing probabilities, without rounding tiny tails
to zero first. Unmapped score clusters or missing mapped class posteriors make
MSE unavailable unless the declared training-prior redistribution resolves them.
Distance/voting exports remain unchanged and are not converted or aggregated
into class probabilities.

## Metric definitions

| Metric | Definition |
|---|---|
| `log_MSE_zeta` | Primary per-split metric: natural log of posterior MSE, calculated directly in log space |
| `MSE_zeta` | Compatibility value: mean over held-out samples **and all K classes** of squared aligned posterior probability minus one-hot truth; reported only for `probability`, `log_probability`, or `logits` exports |
| ACC | Auxiliary diagnostic: fraction of held-out labels matching the **training-aligned** cluster prediction; does not select the comparison cohort |

For n held-out samples, the manuscript metric is

```text
MSE_zeta = sum_i sum_k (posterior_ik - 1[y_i = k])^2 / (n * K).
```

A class absent from held-out truth does not invalidate MSE: all K posterior
columns still contribute. The evaluator computes the true-class error as the
sum of wrong-class probabilities and combines squared errors in log space. It
does not compute `log` of an already rounded raw MSE and does not apply a floor.

`MSE_zeta_status` distinguishes `positive` (positive raw MSE and finite log),
`underflow` (raw MSE rounds to zero but the directly computed log remains finite),
and `zero_in_input` (all errors are exactly zero in the exported scores).
The last case is serialized as raw MSE zero and `log_MSE_zeta=null`, representing
mathematical negative infinity, not missing evaluation. Native log scores should
be exported to preserve tails lost by rounded probability exports.

Native distance, voting, and other nonposterior scores do not acquire a posterior
MSE by being scaled or normalized. Their MSE fields are null and status is
`mse_unavailable`, even when auxiliary ACC is valid. These methods cannot supply
the paper's posterior-MSE panel from the present score exports; this package
does not claim that all paper comparators have runnable posterior adapters.
Switching variable-K Hungarian to majority mapping remains a different evaluation
configuration and must be disclosed.

## Complete, paired reporting

Run from the package root, using paths chosen for your own unbundled data:

```bash
python scripts/evaluate.py \
  --prepared prepared --runs runs --output reports \
  --methods GMM GM3L_SMLE GM3L_FINAL \
  --splits 0 1 2
```

The script creates `metrics.csv`, `summary.csv`, and `summary.json`. There is one
record per **requested** split/method, including missing runs and failures:

- `complete`: valid aligned posterior MSE, including explicitly recorded input zeros.
- `failed`: the adapter did not declare successful completion.
- `missing`: a required prediction/config/label file is absent.
- `evaluation_failed`: invalid shapes, labels, metadata, probabilities, mixed
  protocols, changing score semantics, or another evaluation error.
- `mse_unavailable`: auxiliary ACC and native score semantics are retained, but
  the exports do not provide valid aligned class posterior probabilities.

All comparison summaries use the **same intersection of complete splits across
every requested method with valid posterior MSE**. ACC does not select this
cohort. Input zeros are retained, not silently dropped. The JSON reports all requested
IDs, common valid IDs, method-level status counts, and all excluded split records.
No method-specific favorable subset, best restart, seed selection, top-N ranking,
or successful-run-only silent fallback is used. An empty intersection produces an
explicit zero-sized summary with null statistics. The CLI exits with code 2 if
any requested split is outside the common complete cohort; the reports still exist.

Use `--splits` to state the full intended cohort, including directories that have
not yet been prepared. If omitted, the evaluator uses all existing prepared split
directories; it cannot infer intended-but-not-created splits. `summary.csv`
contains paired mean, median, sample standard deviation, Q1, and Q3, plus status
counts. Convergence diagnostics are retained without treating numerical completion
as a proof of global optimality.

The `log_MSE_zeta_{mean,median,sd,q1,q3}` columns summarize per-split logs over
that same cohort when all are finite. If any retained row is `zero_in_input`,
those finite-log summary fields are null instead of silently dropping the row.
Underflow and input-zero counts are reported separately. `log_mean_MSE_zeta` is
the log of the **arithmetic mean MSE**, computed with log-sum-exp and including
input zeros; it is not the arithmetic mean of per-split logs and is not the
boxplot input. An all-input-zero cohort has null `log_mean_MSE_zeta`, representing
negative infinity as indicated by its zero count.

## Figures and tables

```bash
python scripts/plot_results.py \
  --metrics reports/metrics.csv --output figures \
  --methods GMM GM3L_SMLE GM3L_FINAL
```

The Python/matplotlib panel shows **per-split natural log(MSE_zeta)** in
`log_mse_comparison.pdf`, `.svg`, and `.png`. It uses the common valid-MSE cohort
and method order, not `log_mean_MSE_zeta` or an ACC-filtered subset.
Grayscale boxes do not designate a presumed winner.
Lowercase panel labels, editable PDF/SVG text, and 600-dpi PNG are provided.

Exports include the exact paired source rows, a summary CSV and Markdown table,
and a manifest recording source hashes, paired IDs, exclusions, score semantics,
numerical checks, and export hashes. The plotter requires `log_MSE_zeta` and
`MSE_zeta_status`; legacy raw-MSE-only exports must be reevaluated. It rejects
duplicate rows, missing metrics on `complete` rows, invalid ranges, mixed
protocols, and empty cohorts. A retained `zero_in_input` row also stops plotting:
negative infinity cannot form a finite boxplot without fabrication. The error
asks for native log scores; it does not floor the metric or exclude that split.
It reads only `metrics.csv`, not labels, features, models, or cached results.

Boxes show Q1–Q3, black lines show medians, whiskers extend to the most extreme
observation within 1.5 × IQR, and open dots show **all** outliers. Axis limits cover
every observation. The spread across repeated, potentially overlapping partitions
is not an independent-sample confidence interval. No significance test or claim
of superiority is added. Inspect the exported PNG at intended print size; the
manifest's numerical checks are not a substitute for visual review.

For synthetic verification only, pass `--synthetic`; it adds **Synthetic validation
— not paper results** to the figure and manifest and prefixes figure filenames.
Synthetic fixtures/figures must stay in temporary validation directories, not the
data-free deliverable and not a manuscript results directory.

## Tests

```bash
python -m unittest discover -s tests -p 'test_evaluation.py' -v
python -m unittest discover -s tests -p 'test_plot_results.py' -v
```

These tests generate small synthetic arrays in disposable temporary directories.
They cover soft-versus-hard Hungarian, training-label isolation, posterior MSE,
missing held-out classes, unavailable posteriors, unknown-K majority, declared
prior redistribution, extreme logits, underflow, explicit input zeros,
paired-cohort failures, SVG text, and plot exports. They never import or execute
historical experiment scripts, load real
data or existing result artifacts, or validate empirical scientific conclusions.
