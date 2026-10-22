# Experimental protocol

The native full-graph driver supplied the main settings. Earlier test-to-training
drivers use a different prediction protocol, kept as an explicit option rather
than merged with the full-graph results.

| Stage | Native convention |
| --- | --- |
| Splits | `default_rng(seed_base + replication).permutation(n)`, first floor(n × train_fraction) nodes train; no stratification or result-based filtering |
| Features | Training-only full-SVD PCA, no whitening; fit_transform for training fit and transform for full-graph prediction coordinates |
| SMLE starts | Spectral and X-only KMeans, seed offsets 0 and 100003; pooled initial covariance, edge probabilities 0.7 diagonal/0.3 off-diagonal |
| Initializer KMeans | 10 restarts |
| SMLE selection | Highest objective among converged starts; no test labels or FINAL feasibility used to select a start |
| Native stopping | Euclidean parameter-update norm < 1e-7, maximum 500 iterations in the example; all failures retained |
| FINAL | One closed-form update from selected SMLE training posterior; no refitting on test data |
| GMM | Custom full-covariance EM, offsets 0 and 100000; tied GMM is a separately named variant |
| Numerical boundaries | No covariance ridge, probability clipping or pseudocounts; SMLE edge logit representation preserves tails |
| Alignment | Training posterior mass Hungarian mapping for GMM/SMLE; FINAL uses the same SMLE mapping |
| Primary metric | Natural log of posterior MSE averaged over held-out samples and all K classes; computed directly in log space |

## Information access

For both native graph protocols, parameter fitting only receives training X/A.
In the transductive protocol, prediction receives all unlabelled X/A, including
test–test edges; in the inductive protocol it receives only each test node's
features and test-to-training edges. True labels are excluded from the fitting
directory. Evaluation labels determine only the post-fit mapping and metrics.

Feature-only baselines do not use A. Some external graph methods compute a
full-graph representation before training the final classifier/clustering layer.
Those are declared transductive exceptions, not claimed to have identical
training information access to native GM³L. Their own adapters must disclose it.

## Differences from historical execution infrastructure

- No hard-coded private paths, dataset shape assertions, hand-selected classes,
  old successful-run lists, or saved metric reuse are carried into the runner.
  These are inputs/design decisions to be explicitly supplied, not inferred.
- A fresh run attempts every requested split and records failures. Comparisons
  use and disclose the common valid-posterior-MSE split intersection across every
  requested method; auxiliary ACC does not select the cohort.
- Supported numerical libraries can change rounding, eigenvector signs and
  KMeans paths. Current spectral initialization uses the intended SciPy API;
  historical dependency failures are not deliberately reproduced.
- Full-graph prediction is evaluated in training-then-test order. This is
  mathematically permutation equivalent but can change floating-point summation.
- A missing held-out class does not invalidate posterior MSE: all K class columns
  remain in its denominator. Unmapped class posteriors remain unavailable unless
  an explicitly declared training-prior policy resolves them; missing scores are
  not fabricated. Native distance/voting scores do not become probabilities.
- The embedding and pair-link interfaces do not establish the original cohort,
  raw-data source, prompt or service revision. Those must be confirmed separately.
