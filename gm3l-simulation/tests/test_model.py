"""Small mathematical and reproducibility checks without archived study files."""

import itertools
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from threadpoolctl import threadpool_limits

import gm3l
from gm3l import data, strict_smle
from gm3l.complete_data import complete_data_estimate
from gm3l.exact_spectral_init import initialize_spectral_theta
from gm3l.final_estimator import final_from_zeta


def dense_reference_generator(n, parameters, rng):
    """Original upper-triangle draw order, with dense scratch for tiny tests."""
    alpha, mu, sigma, pi = (parameters[key] for key in ('alpha', 'mu', 'sigma', 'pi'))
    labels = rng.choice(len(alpha), n, p=alpha)
    X = np.empty((n, mu.shape[1]))
    for k in range(len(alpha)):
        selected = np.flatnonzero(labels == k)
        if len(selected):
            X[selected] = rng.multivariate_normal(mu[k], sigma[k], len(selected))
    rows, columns = np.triu_indices(n, 1)
    edge = rng.binomial(1, pi[labels[rows], labels[columns]])
    A = np.eye(n)
    A[rows, columns] = edge
    A[columns, rows] = edge
    return labels, X, A


def enumerated_star_posterior(X, A, theta):
    """Sum all K**n label assignments for each star, independent of aggregation."""
    n = len(X)
    K = len(theta['alpha'])
    states = np.array(list(itertools.product(range(K), repeat=n)))
    log_nodes = np.column_stack([
        np.log(theta['alpha'][k]) + multivariate_normal.logpdf(
            X, mean=theta['mu'][k], cov=theta['sigma'][k])
        for k in range(K)
    ])
    base = log_nodes[np.arange(n)[None, :], states].sum(axis=1)
    zeta = np.zeros((n, K))
    weights = np.zeros((n, K))
    numerator = np.zeros((K, K))
    denominator = np.zeros((K, K))
    objective = 0.
    for center in range(n):
        log_joint = base.copy()
        for leaf in range(n):
            if leaf == center:
                continue
            probability = theta['pi'][states[:, center], states[:, leaf]]
            log_joint += np.log(probability if A[center, leaf] else 1-probability)
        log_marginal = logsumexp(log_joint)
        objective += log_marginal
        probabilities = np.exp(log_joint-log_marginal)
        for labels, probability in zip(states, probabilities):
            zeta[center, labels[center]] += probability
            for leaf in range(n):
                weights[leaf, labels[leaf]] += probability
                if leaf == center:
                    continue
                first, second = sorted((labels[center], labels[leaf]))
                numerator[first, second] += probability * A[center, leaf]
                denominator[first, second] += probability
    for first in range(K):
        for second in range(first+1, K):
            numerator[second, first] = numerator[first, second]
            denominator[second, first] = denominator[first, second]
    return dict(zeta=zeta, weights=weights, pi_numerator=numerator,
                pi_denominator=denominator, objective=objective)


class ModelTests(unittest.TestCase):
    def test_oracle_boundary_and_undefined_cases(self):
        labels = np.array([0, 0, 1, 1])
        x = np.array([[0., 0.], [2., 0.], [0., 2.], [0., 4.]])
        a = np.array([[99., 1., 1., 0.], [1., -99., 0., 1.],
                      [1., 0., np.nan, 0.], [0., 1., 0., np.inf]])
        estimate, diagnostics = complete_data_estimate(labels, x, a, 2)
        np.testing.assert_array_equal(estimate['alpha'], [.5, .5])
        np.testing.assert_array_equal(estimate['mu'], [[1., 0.], [0., 3.]])
        np.testing.assert_array_equal(estimate['sigma'], [[[1., 0.], [0., 0.]], [[0., 0.], [0., 1.]]])
        np.testing.assert_array_equal(estimate['pi'], [[1., .5], [.5, 0.]])
        self.assertEqual(diagnostics['directed_vs_undirected_max_abs_diff'], 0.)
        self.assertTrue(diagnostics['closed_form_estimates_defined'])
        self.assertTrue(diagnostics['psd_covariance'])
        self.assertFalse(diagnostics['pd_covariance'])
        self.assertFalse(diagnostics['global_likelihood_feasible'])
        self.assertEqual(diagnostics['classes_n_le_p'], [0, 1])
        for y, message in ((np.array([0, 0, 0, 0]), 'empty classes'),
                           (np.array([0, 0, 0, 1]), 'zero edge denominator')):
            with self.assertRaisesRegex(ValueError, message):
                complete_data_estimate(y, x, np.zeros((4, 4)), 2)

    def test_fixed_parameters_and_seeds(self):
        truth = data.make_parameters()
        np.testing.assert_array_equal(truth['alpha'], [11/30, 1/3, 3/10])
        np.testing.assert_array_equal(truth['mu'], [[-.3, -1.1, -.7, -.1, .1],
                                                  [.1, -.5, .8, .2, .6],
                                                  [.2, 1.6, -.1, -.1, -.7]])
        distance = np.abs(np.arange(5)[:, None]-np.arange(5)[None, :])
        np.testing.assert_array_equal(truth['sigma'], np.array([.3, .5, .7])[:, None, None]**distance)
        np.testing.assert_array_equal(np.diag(truth['pi']), [.70]*3)
        np.testing.assert_array_equal(truth['pi'][~np.eye(3, dtype=bool)], [.15]*6)
        self.assertEqual(data.DATA_SEED, 2026101101)
        self.assertEqual(data.N_GRID, (3200, 6400, 12800, 25600))
        truth['alpha'][0] = 0.
        self.assertEqual(data.make_parameters()['alpha'][0], 11/30)
        with self.assertRaises(ValueError):
            data.make_true_parameters(K=2, p=3)

    def test_compact_generator_matches_dense_draw_order(self):
        truth = data.make_parameters()
        for rep in (1, 2, 17):
            seed = [data.DATA_SEED, 3, 5, rep]
            expected = dense_reference_generator(80, truth, np.random.default_rng(np.random.SeedSequence(seed)))
            actual = data.simulate_data(80, truth, np.random.default_rng(np.random.SeedSequence(seed)))
            for first, second in zip(expected, actual):
                np.testing.assert_array_equal(first, second)
            self.assertEqual(actual[2].dtype, np.dtype('uint8'))
            np.testing.assert_array_equal(actual[2], actual[2].T)
        for invalid in (0, -1, 3.5, True):
            with self.assertRaises(ValueError):
                data.simulate_data(invalid, truth, np.random.default_rng(1))

    def test_estep_matches_full_label_enumeration(self):
        theta = dict(alpha=np.array([.4, .6]),
                     mu=np.array([[-.4, .3], [.7, -.2]]),
                     sigma=np.array([[[1., .2], [.2, .8]], [[.7, -.1], [-.1, 1.2]]]),
                     pi=np.array([[.7, .2], [.2, .6]]))
        X = np.array([[.2, -.1], [-.3, .8], [1.1, .2], [.6, -.5], [-.7, .4]])
        A = np.array([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [0, 1, 1, 1, 0],
                      [0, 0, 1, 1, 1], [1, 0, 0, 1, 1]], dtype=np.uint8)
        expected = enumerated_star_posterior(X, A, theta)
        actual = strict_smle.estep(X, A, theta)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12,
                                       err_msg=key)
        self.assertAlmostEqual(float(actual['weights'].sum()), len(X)**2)

    def test_final_one_hot_matches_oracle(self):
        Y, X, A = data.simulate_data(80, data.make_parameters(), np.random.default_rng(713))
        expected, _ = complete_data_estimate(Y, X, A, 3)
        actual, _ = final_from_zeta(X, A, np.eye(3)[Y])
        for key in ('alpha', 'mu', 'sigma', 'pi'):
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-13, atol=1e-13)

    def test_spectral_initializer_and_small_fit(self):
        _, X, A = data.simulate_data(160, data.make_parameters(), np.random.default_rng(1701))
        cfg = gm3l.FitConfig(max_iter=5, tolerance=1e-6)
        with threadpool_limits(limits=1):
            initial, diagnostics = initialize_spectral_theta(
                X, A, 3, 260918+1009+17*160, cfg, return_diagnostics=True)
            again = initialize_spectral_theta(X, A, 3, 260918+1009+17*160, cfg)
            for key in initial:
                np.testing.assert_array_equal(initial[key], again[key])
            result = strict_smle.fit_smle(X, A, initial, cfg)
        self.assertIsNone(result['failure'])
        self.assertGreater(result['iterations'], 0)
        self.assertLessEqual(result['iterations'], 5)
        self.assertLessEqual(diagnostics['max_relative_eigen_residual'], 1e-8)

    def test_numpy_import_does_not_load_accelerate(self):
        code = ("import sys; import gm3l; gm3l.get_backend('numpy'); "
                "assert 'gm3l.accelerate_blas' not in sys.modules; "
                "import gm3l.accelerate_blas as backend; assert backend._LIB is None")
        result = subprocess.run([sys.executable, '-B', '-c', code],
                                cwd=Path(gm3l.__file__).resolve().parent.parent,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with self.assertRaises(ValueError):
            gm3l.get_backend('unknown')

    @unittest.skipUnless(sys.platform == 'darwin', 'macOS-only numerical comparison')
    def test_optional_accelerate_matches_numpy(self):
        _, X, A = data.simulate_data(80, data.make_parameters(), np.random.default_rng(711))
        numpy_engine = gm3l.get_backend('numpy')
        mac_engine = gm3l.get_backend('accelerate')
        with threadpool_limits(limits=1):
            expected = numpy_engine.estep(X, A, data.make_parameters())
            actual = mac_engine.estep(X, A, data.make_parameters())
            for key in expected:
                np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-10)
            expected_final, _ = final_from_zeta(X, A, expected['zeta'])
            from gm3l.accelerate_blas import dgemm
            actual_final, _ = final_from_zeta(X, mac_engine._off_diagonal(A), expected['zeta'], dgemm)
            for key in expected_final:
                np.testing.assert_allclose(actual_final[key], expected_final[key], rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
