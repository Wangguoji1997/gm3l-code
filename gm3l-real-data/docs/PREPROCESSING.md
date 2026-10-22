# Preprocessing and label boundaries

## Split configuration

`scripts/prepare_data.py` reads the user-specified JSON configuration. Required
fields are `seed_base`, `replications`, `train_fraction`, `pca_components`, and `K`.
`protocol` defaults to `transductive`; `method_options` defaults to an empty object.

```json
{
  "seed_base": 1000,
  "replications": 3,
  "train_fraction": 0.5,
  "pca_components": 2,
  "K": 2,
  "protocol": "transductive",
  "method_options": {}
}
```

These small numbers only demonstrate the interface. `K` must satisfy
`2 <= K < n_train`, matching the native estimators. Choose settings for the new
study before evaluating held-out labels. `pca_components: null` disables PCA.
Otherwise the count must not exceed `min(n_train, input_feature_count)`.

For replication `r`, the seed is exactly `seed_base + r`. Indices are obtained with
`numpy.random.default_rng(seed).permutation(n)`. The first
`floor(n * train_fraction)` records form training; the remainder form testing.
There is no stratification, label-based split adjustment, class-count inference,
seed replacement, success-only selection, or access to historical split records.
Labels cannot affect the permutation, PCA or graph slicing. Fewer classes in a
particular train/test set must be handled transparently by evaluation, not by
replacing that seed.

Method options are copied unchanged into `input/config.json`: the dictionary is
retained under `method_options`, and its entries are also available at top level
for adapter compatibility. Options cannot override split/protocol fields such as
`seed`, `K`, `p`, or `protocol`. Do not put labels, raw text or private paths into
method options.

## Training-only PCA and exact coordinate distinction

PCA uses scikit-learn `PCA(..., svd_solver="full", whiten=False)`. Only training
features are passed to `fit_transform`. No scaling, imputation, whitening,
test-feature fitting, or label filtering is performed.

- `X_train`: `pca.fit_transform(raw_X_train)`; used for fitting the training model.
- `X_train_eval`: `pca.transform(raw_X_train)`; used with the transformed test
  features for full-graph posterior evaluation.
- `X_test`: `pca.transform(raw_X_test)`.

The two training coordinate arrays are mathematically equivalent but may differ
at floating-point precision. They are intentionally both retained, matching the
historical fitting/whole-graph transformation distinction. When PCA is disabled,
both training arrays are equal copies of the original training features.

## Output layout and graph protocol

```text
output/
  split_000/
    input/
      X_train.npy / X_train.csv
      X_train_eval.npy / X_train_eval.csv
      X_test.npy / X_test.csv
      A_train.npy / A_train.csv
      A_test_train.npy / A_test_train.csv
      A_test_test.npy / A_test_test.csv   [transductive only]
      config.json
    evaluation_only.npz
    pca_training_only.npz              [when PCA is enabled]
```

`A_train` is train-by-train; `A_test_train` is test-by-train. In `transductive`
mode, `A_test_test` is test-by-test and may be used during frozen-parameter
full-graph evaluation. Training still receives only training features/edges.
In `inductive` mode, `A_test_test` is not exported. Choosing this mode therefore
changes available prediction information; it is not an identical historical
transductive protocol.

`evaluation_only.npz` contains `train_indices`, `test_indices`, `y_train`,
`y_test`, `ids_train`, `ids_test`, and the ID aliases `chat_id_train`,
`chat_id_test`. No ground-truth labels or ID mapping are written to `input/`.
The PCA archive contains its training mean, components and explained variance,
not labels. Training/prediction adapters should only receive `input/`; evaluation
opens the separate labels archive after fitting. Training-only posterior
memberships are the appropriate source for any training-label mapping; do not
substitute full-graph posterior training memberships.

NPY and CSV contain the same row order, with CSV feature values written using
17 significant digits. Existing split directories are refused, including partial
ones. Use a new output directory for a new run; preprocessing does not load a
cached split or infer which previous seeds succeeded.

## Features: existing archive or explicit offline encoding

Existing features can be validated and copied while stripping texts and labels:

```bash
python scripts/prepare_features.py import \
  --embeddings /path/to/existing-embeddings.npz --output /path/to/new-embeddings.npz
```

Output contains only `X`, IDs and a small non-identifying recipe. Supply labels
separately to `prepare_data.py`. No model is imported or run on this route.

An optional, explicitly selected local path supports CSV text encoding:

```bash
python scripts/prepare_features.py encode-local \
  --texts-csv /path/to/text.csv --id-column id --text-column text \
  --model-dir /path/to/already-local-model \
  --device cpu --batch-size 4 --max-tokens 8192 \
  --output /path/to/new-embeddings.npz
```

This command requires separately installed `torch` and `transformers`. The model
directory must already exist. Both libraries' offline flags are set;
`local_files_only=True` and `trust_remote_code=False` are used. No automatic model
download, remote API, credential lookup, or service test is implemented.

Only the requested ID/text columns are used. Original text is unchanged; no query
prefix or label is encoded. The encoder applies CLS pooling
(`last_hidden_state[:, 0]`), float32 L2 normalization, length-sorted batching with
attention masks, then restores original row order. Text over the token limit
causes an error instead of truncation. Texts are not saved in the output archive.
This preserves the broad source-code recipe, not a claim that an unspecified
local model reproduces a historical model snapshot. The data owner must record
model identity/revision and package versions separately. The local model route
was not exercised during package tests.

## Cleaning boundary

The source cleaning code aggregated utterances by ID, compared full-dialogue
duplicates using NFKC normalization with whitespace removed, treated
cross-label duplicates differently from same-label duplicates, applied separately
reviewed content exclusions, and filtered classes by a fixed minimum count.
Those dataset-specific, label-informed choices are not automatically replayed.
No original exclusion file, class list, source-row list or text is included.
The user supplies the final approved observation universe and documents its
cleaning decisions; this package validates alignment rather than guessing them.

## Tests

`tests/test_preprocessing.py` constructs tiny arrays and exhaustive pair examples
at test time inside temporary directories. It verifies ID alignment, no pickle,
complete-graph validation, training-only PCA, deterministic seed enumeration,
inductive graph isolation, label isolation and refusal to reuse old split paths.
It does not read research data, call a local language model, access a service, or
assert that historical empirical results have been reproduced.
