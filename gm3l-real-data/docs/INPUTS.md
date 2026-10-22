# User-supplied inputs

No research observations, text, annotations, labels, cached embeddings, saved splits,
model weights, results, API credentials or original data locations are included in
this package. Paths below are placeholders. The package does not search for data.

## One archive

Provide a NumPy `.npz` archive with these arrays. All loaders use
`allow_pickle=False`; object arrays are forbidden.

| Key | Shape/type | Contract |
| --- | --- | --- |
| `X` | `(n, d)`, real numeric | Finite features; `d > 0`. Features are converted to float64 for splitting/PCA. |
| `A` | `(n, n)`, numeric/bool | Fully observed binary symmetric adjacency; diagonal exactly zero. |
| `ids` or `chat_id` | `(n,)`, integer or string | Nonempty, unique record IDs, in the same order as both axes of `A` and rows of `X`/`y`. IDs are normalized to strings. |
| `y` | `(n,)`, finite integer codes | Evaluation labels only, prespecified as `0,...,K-1`; not pickled. |

If both ID aliases exist, they must agree. Model component count `K` is supplied
explicitly in the configuration; it is never inferred from evaluation labels.
If raw labels are strings or use another coding, encode them with a documented
class dictionary before import. Keep that dictionary with the private evaluation
metadata; the model is not supplied with the class names.
If `complete` is supplied it must be a true scalar. If `observed` is supplied,
every off-diagonal pair must be observed. Partial annotation matrices are rejected:
unknown pairs must not be relabelled as negative edges.

```bash
python scripts/prepare_data.py --config configs/example.json \
  --dataset /path/to/dataset.npz --output /path/to/new-output
```

The config filename here is illustrative; use the actual provided example or a
user-created JSON file. No example command is run automatically.

## Separate archives, joined by ID

Alternatively provide `embeddings.npz` with `X` and IDs, and `adjacency.npz` with
`A` and its own IDs. Labels may be in `embeddings.npz`, or in a third NPZ with `y`
and IDs. The full ID sets must match. The loader reorders both adjacency axes and
evaluation labels to feature order; it never takes an intersection or assumes
positional equality. Evaluation labels in a separately specified archive are the
explicit source of truth, so no labels in the embedding archive are read.

```bash
python scripts/prepare_data.py --config configs/example.json \
  --embeddings /path/to/embeddings.npz --adjacency /path/to/adjacency.npz \
  --evaluation /path/to/labels.npz --output /path/to/new-output
```

`--dataset` cannot be combined with any of the separate-archive arguments.

## Existing adjacency or exhaustive pair labels

An existing complete adjacency can be validated and exported without any service
call:

```bash
python scripts/build_must_links.py --adjacency /path/to/existing-A.npz \
  --output /path/to/new-adjacency.npz
```

For pair annotations, supply a CSV with columns `id_i,id_j,must_link` and an NPZ
containing the complete record universe under `ids` or `chat_id`. Every unordered
pair must appear exactly once, with `must_link` exactly `0`, `1`, `Yes`, or `No`.
Unknown IDs, duplicates, self-pairs and omitted pairs are errors. Column names are
configurable with `--left-column`, `--right-column`, and `--label-column`.

```bash
python scripts/build_must_links.py --pairs /path/to/exhaustive-pairs.csv \
  --ids-npz /path/to/record-ids.npz --output /path/to/new-adjacency.npz
```

The importer adds symmetric entries, fixes the diagonal to zero and performs no
transitive closure. There is no LLM/API generation command in this package.
The historical source used pairwise binary judgments, but reading new annotations
does not establish reproduction of the historical API/model or its outputs.

## Provenance questions to settle before a real run

The source code alone does not establish the dataset's collection source, access
rights, release status, consent or de-identification, annotation origin, or whether
feature/edge construction had access to evaluation labels. The data owner must
document these before use. In particular:

1. Confirm the meaning and provenance of observations, labels and graph edges.
2. Confirm permission to process the text and store derived vectors/graphs.
3. Record any cleaning, exclusions or label revisions and when they were decided.
4. Identify the exact local embedding model snapshot and software versions, if used.
5. Document whether a complete graph was observed or constructed; partial graphs
   require a missing-edge-aware method not implemented by these input routines.

The package does not infer these facts, ship a label list or hard-code an original
sample count/class filter. It cannot promise numerical reproduction of historical
results without independently supplied data and a documented compatible protocol.
