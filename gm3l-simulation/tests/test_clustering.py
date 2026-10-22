"""Stable-posterior and baseline reproducibility tests on generated toy data."""
import os

# Match the production runner before importing either NumPy or SciPy. In older
# mixed-BLAS environments, restoring a preexisting multi-thread pool between
# tests can crash in native BLAS code; the study uses one thread throughout.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp

from gm3l import clustering, data, strict_smle


class ClusteringTests(unittest.TestCase):
    def test_stable_metric_matches_direct_well_scaled_errors(self):
        rng = np.random.default_rng(619)
        logits = rng.normal(size=(80, 3))
        labels = rng.integers(0, 3, size=80)
        posterior = np.exp(logits - logsumexp(logits, axis=1)[:, None])
        expected = np.log(np.square(posterior - np.eye(3)[labels]).mean())
        self.assertAlmostEqual(clustering.log_mse_from_logits(logits, labels), expected, 13)
        self.assertAlmostEqual(clustering.log_mse_from_logits(logits + 2000, labels), expected, 12)

    def test_underflow_and_rounded_true_probability_remain_finite(self):
        for gap in (40., 1000., 10000.):
            logits = np.array([[0., -gap, -gap], [-gap, 0., -gap]])
            labels = np.array([0, 1])
            actual = clustering.log_mse_from_logits(logits, labels)
            expected = np.log(2.) - 2 * gap - 2 * np.logaddexp(0., np.log(2.) - gap)
            self.assertTrue(np.isfinite(actual))
            self.assertAlmostEqual(actual, expected, 12)
        posterior = np.exp(logits - logsumexp(logits, axis=1)[:, None])
        self.assertEqual(np.square(posterior - np.eye(3)[labels]).mean(), 0.)

    def test_metric_rejects_invalid_shapes_labels_and_logits(self):
        cases = [([], []), ([[1.]], [0]), ([[0., np.inf]], [0]),
                 ([[0., 1.]], [2]), ([[0., 1.]], [0.]),
                 ([[0., 1.]], [[0]])]
        for logits, labels in cases:
            with self.subTest(logits=logits, labels=labels):
                with self.assertRaises(ValueError):
                    clustering.log_mse_from_logits(logits, labels)

    def test_star_metric_matches_estep_and_ignores_graph_diagonal(self):
        truth = data.make_parameters()
        labels, X, A = data.simulate_data(60, truth, np.random.default_rng(917))
        order = np.array([2, 0, 1])
        posterior = strict_smle.estep(X, A, truth)['zeta'][:, order]
        expected = np.log(np.square(posterior - np.eye(3)[labels]).mean())
        actual = clustering.star_log_mse(X, A, truth, labels, order)
        self.assertAlmostEqual(actual, expected, 12)
        np.fill_diagonal(A, 0)
        self.assertAlmostEqual(clustering.star_log_mse(X, A, truth, labels, order), actual, 12)
        with self.assertRaises(ValueError):
            clustering.star_log_mse(X, A, truth, labels, [0, 0, 1])
        with self.assertRaises(ValueError):
            clustering.star_log_mse(X, A, truth, labels, order, backend="unknown")

    @unittest.skipUnless(sys.platform == 'darwin', 'macOS-only numerical comparison')
    def test_star_accelerate_matches_numpy(self):
        truth = data.make_parameters()
        labels, X, A = data.simulate_data(60, truth, np.random.default_rng(918))
        args = (X, A, truth, labels, np.arange(3))
        self.assertAlmostEqual(clustering.star_log_mse(*args, backend="accelerate"),
                               clustering.star_log_mse(*args), 10)

    def test_gmm_fixed_configuration_seed_and_labels_only_after_selection(self):
        rng = np.random.default_rng(712)
        X = rng.normal(size=(40, 2))
        labels = rng.integers(0, 3, size=40)
        created = []

        class FakeGMM:
            def __init__(self, **kwargs):
                self.start = len(created)
                created.append(kwargs)
                self.weights_ = np.array([.3, .4, .3])
                self.means_ = np.array([[-1., 0.], [0., 0.], [1., 0.]])
                self.covariances_ = np.repeat(np.eye(2)[None], 3, axis=0)
                self.converged_ = self.start != 0
                self.n_iter_ = self.start + 1
                self.lower_bound_ = -1.

            def fit(self, observed):
                np.testing.assert_array_equal(observed, X)
                return self

            def score(self, observed):
                return clustering._gmm_log_posterior(observed, self)[1]

        class GuardedLabels:
            def __array__(self, dtype=None):
                if len(created) != 10:
                    raise AssertionError("Truth labels accessed before all fits were selected")
                return np.asarray(labels, dtype=dtype)

        with patch.object(clustering, "GaussianMixture", FakeGMM):
            result = clustering.fit_gmm(X, GuardedLabels(), rep=7, n=40)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['diagnostics']['selected_start'], 1)
        self.assertEqual(result['iterations'], 2)
        self.assertEqual(len(created), 10)
        for start, kwargs in enumerate(created):
            self.assertEqual(kwargs, dict(
                n_components=3, covariance_type="full", tol=1e-6,
                reg_covar=0.0, max_iter=20000, n_init=1, init_params="kmeans",
                random_state=int(np.random.SeedSequence([2026101401, 7, 40, start]).generate_state(1)[0])))

    def test_gmm_all_failed_is_explicit_and_does_not_use_labels(self):
        with patch.object(clustering, 'GaussianMixture', side_effect=ValueError('singular covariance')):
            result = clustering.fit_gmm(np.ones((10, 2)), object(), rep=1, n=10)
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['log_mse_zeta'])
        self.assertEqual(len(result['diagnostics']['attempts']), 10)

    def test_gmm_small_fit_is_deterministic(self):
        rng = np.random.default_rng(913)
        labels = np.repeat(np.arange(3), 25)
        X = rng.normal(scale=.7, size=(len(labels), 2)) + np.array([[-3., 0.], [0., 3.], [3., 0.]])[labels]
        first = clustering.fit_gmm(X, labels, rep=2, n=len(X))
        second = clustering.fit_gmm(X, labels, rep=2, n=len(X))
        self.assertEqual(first, second)
        self.assertEqual(first['status'], 'ok')
        self.assertTrue(np.isfinite(first['log_mse_zeta']))
        self.assertAlmostEqual(first['log_mse_zeta'], np.log(first['mse_zeta']), 9)


if __name__ == '__main__':
    unittest.main()
