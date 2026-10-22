# CLSBM split adapter

`external_adapters/run_clsbm.py` is a portable extraction of the existing
`run_clsbm_transductive.py` adapter. That source is self-contained: it imports no
local algorithm module and does not require an author repository to be vendored.
The extraction was based on source code inspection, not historical data/results.
It reproduces this adapter's equations and split extension, not a claim to bundle
an entire author's repository or reproduce historical empirical measurements.

## What the method computes

For the concatenated training and test feature matrix `X`, full graph `A`, full
sample size `n`, user-specified `K`, and seed:

1. Draw `w = numpy.random.default_rng(seed).uniform()` exactly once.
2. Compute `S = w * A + X @ X.T / n`.
3. Use SciPy's full symmetric eigendecomposition, selecting `K` eigenvectors in
   descending absolute eigenvalue order. Let these vectors form `U`.
4. Construct `Q = S @ U`, without additional row normalization or regularization.
5. Fit scikit-learn `KMeans(n_clusters=K, n_init=30, random_state=seed)` on training
   rows of `Q` only.
6. Assign each test row to its nearest training centroid. Each test score is the
   negative squared distance to the corresponding centroid.

The graph and spectral representation include held-out features and test-test
edges. This is an explicit **transductive representation exception**, not a
training-only representation fit. The training/test extension consists of fitting
centroids on training nodes and assigning held-out nodes in the shared spectral
coordinates; it is not presented as an author's native inductive predictor.
PCA upstream can still be trained on training features only, but that fact does
not remove the spectral representation's use of test observations.

No real labels are loaded. Predicted cluster IDs are one-based, matching the old
adapter. `test_scores.csv` retains the original negative squared distances;
`score_type` is `decision`, and alignment is hard
Hungarian using training labels later in the separate evaluator. Distances are
not converted into probabilities and are not described as calibrated confidence.
These native scores cannot supply the manuscript's posterior-MSE panel: evaluation
retains ACC as an auxiliary diagnostic and reports `mse_unavailable`.
No artificial scores or fabricated convergence result are added. Diagnostics
report K-means iterations and explicitly do not certify a global optimum.

## Inputs, invocation and external-runner configuration

The bridge accepts exactly the five model CSV files in an isolated directory:
`X_train.csv`, `X_test.csv`, `A_train.csv`, `A_test_train.csv`, and
`A_test_test.csv`. Files must follow the shared external adapter shape/binary
graph validation. Unknown files, including label/config files, are rejected.
`K` is an explicit CLI argument and is not read from labels or a config file.
Inductive input without `A_test_test` fails rather than substituting zero edges.

```bash
python external_adapters/run_clsbm.py --input /path/to/isolated-model-csvs \
  --output /path/to/new-output --seed 1000 --K 2
```

The following is an example `method_options.external` entry for the shared
runner, not a new dataset configuration. Replace executable/script paths with
the installation's paths, and keep the literal `--K` value equal to the study's
configured component count. The external protocol deliberately does not forward
the experiment config or infer `K` from data.

```json
{
  "CLSBM": {
    "command": [
      "/path/to/python", "/path/to/package/external_adapters/run_clsbm.py",
      "--input", "{input}", "--output", "{output}",
      "--seed", "{seed}", "--K", "2"
    ],
    "timeout": 3600
  }
}
```

Select `CLSBM` as a method and use `protocol: "transductive"`. The command should
be run through the shared external runner to obtain a fresh CSV-only directory;
the normal prepared `input/` directory also contains NPY/config files and is not
itself a valid standalone CLSBM bridge input. Existing nonempty output directories
are refused. Failure writes explicit diagnostics rather than substitute results.

The adapter emits training/test predicted labels, test distance scores,
`score_columns.json`, `diagnostics.json`, and a pickle-free `model.npz` containing
the sampled weight, selected eigenvalues/basis and training centroids.

## Dependencies and documented portability changes

Required installed packages are NumPy, SciPy, scikit-learn and threadpoolctl, plus
the `gm3l_real` package located alongside the adapter. No model weights, network
access, R package, hosted service, API key or full external repository is needed.
The numerical block limits BLAS threads to one, as did the source adapter.

The source read NPY arrays and `K` from a JSON config; this extraction reads the
shared CSV-only interface and a required `--K`. It also adds stricter input
validation, no-overwrite handling, explicit failure status, and native score-type
metadata required by the shared runner. The equation, uniform draw, eigenvector
selection, K-means initialization count and distance score are unchanged.

As in the source adapter, `X_train` is used rather than the GM3L-specific
`X_train_eval`. No attempt is made to claim bitwise equality across library/BLAS
versions; eigenvector orientation, degenerate eigenspaces and K-means behavior can
vary numerically. The fixed method remains transductive even when it emits
training memberships for label alignment.

## Verification scope

`tests/test_clsbm_adapter.py` constructs tiny local arrays, checks the result
against the source equations, executes the actual CSV bridge in a subprocess,
verifies the unchanged distance-score contract, and rejects inductive input,
label-file contamination and output reuse. No historical dataset or output is
read, and no empirical accuracy/reproduction claim is derived from these tests.
