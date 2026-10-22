# Local data directory

No data are distributed with this package. The original study's data provider,
access route, permissions and redistribution conditions must be supplied and
confirmed by the authors before release. Do not interpret the filenames below
as evidence that any particular dataset exists or is publicly downloadable.

Place authorized local inputs here, for example:

- `embeddings.npz`: numeric `X` and unique string `ids`.
- `adjacency.npz`: binary symmetric zero-diagonal `A` and matching `ids`.
- `evaluation.npz`: integer evaluation labels `y` and matching `ids`.

Alternatively use `dataset.npz` with `X`, `A`, `ids`, and `y`.
IDs are aligned before splitting; evaluation labels are exported separately
from model inputs. See [input specifications](../docs/INPUTS.md).

The ignore rules are a safeguard, not a substitute for checking a release
archive or Git staging area. Do not publish raw texts, identifiers, embeddings,
must-link responses, model checkpoints, or per-record predictions unless
their release is explicitly authorized.
