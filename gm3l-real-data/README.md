# GM³L real-data analysis

Code for preparing model inputs, fitting GM³L and comparison methods, evaluating
clustering, and exporting comparison figures and tables. This directory is
independent of the companion `gm3l-simulation` package.

**No study data, fitted models, existing results, API credentials or pretrained
weights are included.** The reorganization was checked using source code and
small synthetic tests only. It did not read or run the study data.

## Data access

The community telephone service data used in this study are not included in
this repository to protect personal privacy. The repository provides the
preprocessing and analysis code, together with instructions on the required
input formats, for use with authorized data. All simulation data can be
generated using the provided code.
See [data placement](data/README.md) and the complete [input contract](docs/INPUTS.md).

The primary interface accepts user-supplied inputs, not a hard-coded private
path. A combined NumPy archive has the following fields:

| Field | Required format |
| --- | --- |
| `X` | Finite numeric feature matrix, shape `n × p_embedding` |
| `A` | Complete binary symmetric matrix, shape `n × n`, diagonal zero |
| `ids` | Unique one-dimensional string IDs, aligned with matrix rows |
| `y` | Integer reference labels `0,...,K-1`, for evaluation only |

Missing must-link responses are **unknown**, not zero. Incomplete adjacency
matrices are rejected. Separate embeddings, adjacency and label archives can
be joined by ID; row order is never guessed from filenames. `K` must be set
explicitly in the experiment configuration rather than inferred from test labels.

## Layout

```text
configs/example.json           explicit experiment settings (set K before use)
data/README.md                 input placement and access requirements; no data
gm3l_real/
  data_io.py                   validation, ID alignment, input conversion
  preprocessing.py             random splits and training-only PCA
  model.py                     GMM, log-domain SMLE, FINAL and posterior prediction
  baselines.py                 FINCH and CLASSIX wrappers
  external.py                  label-free external-process interface
  runner.py                    fitting, model selection and failure records
  evaluation.py                training alignment, log(MSE_zeta), auxiliary ACC
  parameters.py                evaluation-only training-oracle comparisons
scripts/                       command-line stages listed below
external_adapters/             portable optional comparison-method bridges
docs/                          input, protocol, method and evaluation details
tests/                         small synthetic and interface tests
requirements.txt               tested core Python dependencies
```

## Environment

Use Python 3.10 and an isolated environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Run commands below from this directory. Importing the package does not read
data, fit a model or access a network service. Optional methods have separate
dependencies; they are not installed or downloaded automatically. The test
suite does not require those methods, a pretrained language model or actual data.

## Execution order

### 1. Prepare features and must-links

If features and the must-link matrix already exist, retain them as inputs;
there is no need to regenerate either. `prepare_features.py import` validates
and exports existing vectors. `prepare_features.py encode-local` can explicitly
encode a text CSV with a user-supplied **offline local model**, using the original
CLS pooling and L2-normalization recipe. No model is downloaded, and no hosted
API is called. Model identity/revision and preprocessing decisions must be
recorded by the experimenter.

`build_must_links.py` reads an existing adjacency archive or builds a symmetric
matrix from a complete ID-pair CSV. It does not infer links from reference
labels, fill unobserved responses with zero, or apply transitive closure.
The private prompts, provider credentials and raw pair responses are not bundled.

Use `--help` for these optional conversion stages. See
[preprocessing details](docs/PREPROCESSING.md) for exact commands and formats.

### 2. Configure and prepare splits

Copy `configs/example.json` to `configs/local.json`, set `K` to the prespecified
number of classes, and confirm all settings against the experiment being
reproduced. `K: null` deliberately prevents an unconfigured run. The example
preserves the original half-split, full-SVD PCA and native fitting settings;
500 replications are explicit, not executed by installation. Confirm the PCA
dimension is appropriate for the supplied features.

```bash
python scripts/prepare_data.py \
  --config configs/local.json \
  --embeddings data/embeddings.npz \
  --adjacency data/adjacency.npz \
  --evaluation data/evaluation.npz \
  --output work/prepared
```

For a combined archive, replace the three input options with
`--dataset data/dataset.npz`. Each `split_000`, `split_001`, ... contains
`input/` with only feature/graph arrays and settings, plus a separate
`evaluation_only.npz`. PCA is fitted on the training features only. Splits use
consecutive seeds and are not selected using labels, convergence or old results.

The two graph protocols are different experiments:

- `transductive`: fit parameters on training nodes; predict using the full
  unlabelled feature/graph collection, including test–test links. This is the
  full-graph real-data driver convention.
- `inductive`: fit on training nodes; predict each test node from its features
  and links to training nodes, without reading test–test links.

The runner and evaluator record the protocol. Do not combine the protocols in
one numerical comparison. [Protocol details](docs/PROTOCOL.md) explain model
selection, score alignment and historical implementation differences.

### 3. Fit GM³L and comparison methods

```bash
python scripts/run_analysis.py \
  --prepared work/prepared --runs work/runs \
  --methods GMM SMLE FINAL --threads 1
```

`SMLE` and `FINAL` share the selected training-only SMLE fit. Selection uses
the largest training composite likelihood among converged starts, not test
metrics. `FINAL` is a single closed-form update. GMM uses the original custom
EM, without a covariance ridge or a replacement sklearn estimator.

Additional built-in names are `GMM_tied`, `FINCH`, `CLASSIX` and
`CLASSIX_known_K`. Optional FINCH and CLASSIX require user-installed upstream
packages.

For the other comparison methods, see [baselines and dependencies](docs/BASELINES.md)
and `external_adapters/`. The common external interface preserves runnable
commands and output contracts without copying entire third-party repositories.
Some methods still require an author-supplied dependency/bridge; their status is
documented individually. Missing dependencies are reported, never replaced by
another algorithm. External code is trusted user code, not an OS sandbox.
The package does not claim that all paper comparators are wired into a runnable
posterior-MSE comparison; methods without native posterior scores cannot supply
that panel, and some listed methods still have no portable adapter.

The example configuration already includes optional commands for `CLSBM`,
`DEC` and `DipEncoder`; request them by name in `--methods`. CLSBM uses the core
dependencies and requires transductive graph inputs. DEC/DipEncoder require
the documented ClustPy/PyTorch environment. `{python}`, `{package}` and `{K}`
are resolved by `run_analysis.py`; no machine-specific adapter path is needed.
The per-method `input_keys` allowlist passes only features to DEC/DipEncoder.

For a subset, add `--splits 0 1 2`. Existing method output directories are never
silently overwritten. Use a new run directory when changing settings.
Every attempted split/method keeps a diagnostic, including nonconvergence and
numerical failures. Exit code 2 means at least one method was not complete.

### 4. Evaluate and export figures/tables

```bash
python scripts/evaluate.py \
  --prepared work/prepared --runs work/runs --output results/evaluation \
  --methods GMM SMLE FINAL

python scripts/plot_results.py \
  --metrics results/evaluation/metrics.csv --output results/figures \
  --methods GMM SMLE FINAL
```

Outputs include per-split metrics, a complete failure/missing-run record,
matched-split summary tables, and `log_mse_comparison.{pdf,svg,png}` plots of
per-split natural `log(MSE_zeta)`. Cluster-to-class alignment uses **training
labels only**, after fitting. `MSE_zeta` averages squared posterior errors over
both held-out samples and all K classes; ACC remains an auxiliary diagnostic.
Only native probability/log-probability/logit exports can supply this metric;
distance and voting scores are retained but report `mse_unavailable`, never
converted into probabilities. Comparisons use the same valid-MSE splits across
every requested method. Direct log-domain computation preserves finite values
when raw `MSE_zeta` underflows. An exact zero in the exported input is recorded,
not floored or silently discarded; the plotter asks for native log scores rather
than fabricating a finite boxplot value. `log_mean_MSE_zeta` is a separate summary
of the arithmetic mean and is not the plotted per-split metric.
See [evaluation details](docs/EVALUATION.md).

For the original parameter-deviation comparison (SMLE and FINAL against the
training-label oracle), run the optional evaluation-only stage:

```bash
python scripts/parameter_diagnostics.py \
  --prepared work/prepared --runs work/runs \
  --output results/parameters --methods SMLE FINAL --plot
```

This reports full-array mean squared deviations for alpha, mu, Sigma and Pi;
it is **not** estimation error against known population parameters. Singular
oracle covariances and undefined class denominators are recorded. Exact zero
deviations are not replaced by a positive floor just to take logarithms.

## Reproducibility scope

The code no longer depends on private absolute paths or cached successful
replication lists. Original model equations and declared settings are retained;
data access, the final cohort selection and optional upstream environments must
still be supplied. New split ordering and current numerical libraries need not
produce bit-for-bit legacy results. See [source provenance](docs/PROVENANCE.md).

Before public or anonymous release, complete the data-access statement, select
the project's license, verify third-party terms, and review the staged file list.
No repository or anonymous mirror is created by any script in this package.
