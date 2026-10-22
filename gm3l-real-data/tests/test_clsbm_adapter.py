"""CLSBM bridge tests use only locally constructed tiny arrays."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from external_adapters.run_clsbm import run
from gm3l_real.external import run_external, ExternalAdapterError


class CLSBMAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gm3l-clsbm-test-")
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(29)
        x = rng.normal(size=(10, 3))
        upper = np.triu(rng.integers(0, 2, size=(10, 10)), 1)
        a = upper + upper.T
        self.arrays = dict(X_train=x[:6], X_test=x[6:], A_train=a[:6, :6],
                           A_test_train=a[6:, :6], A_test_test=a[6:, 6:])
        self.input = self.root / "input"
        self.input.mkdir()
        for name, array in self.arrays.items():
            np.savetxt(self.input / (name + ".csv"), array, delimiter=",", fmt="%.17g")
        self.script = Path(__file__).resolve().parents[1] / "external_adapters/run_clsbm.py"

    def tearDown(self):
        self.temp.cleanup()

    def test_matches_original_adapter_equations_and_unmodified_scores(self):
        seed, K = 17, 2
        output = self.root / "result"
        run(self.input, output, seed, K)
        x = np.vstack([self.arrays["X_train"], self.arrays["X_test"]])
        a = np.block([[self.arrays["A_train"], self.arrays["A_test_train"].T],
                      [self.arrays["A_test_train"], self.arrays["A_test_test"]]])
        weight = float(np.random.default_rng(seed).uniform())
        with threadpool_limits(limits=1):
            s = weight * a + x @ x.T / len(x)
            values, vectors = eigh(s, check_finite=False)
            order = np.argsort(np.abs(values))[::-1][:K]
            q = s @ vectors[:, order]
            km = KMeans(n_clusters=K, n_init=30, random_state=seed).fit(q[:6])
            scores = -np.sum((q[6:, None] - km.cluster_centers_[None]) ** 2, axis=2)
        np.testing.assert_array_equal(np.loadtxt(output / "train_labels.csv", dtype=int), km.labels_ + 1)
        np.testing.assert_allclose(np.loadtxt(output / "test_scores.csv", delimiter=","), scores, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(np.loadtxt(output / "test_labels.csv", dtype=int), scores.argmax(axis=1) + 1)
        with np.load(output / "model.npz", allow_pickle=False) as saved:
            self.assertEqual(saved["weight"].item(), weight)
            np.testing.assert_allclose(saved["eigenvalues"], values[order])
        meta = json.loads((output / "score_columns.json").read_text())
        self.assertEqual(meta["score_type"], "decision")
        self.assertEqual(meta["alignment"], "hard")
        diagnostics = json.loads((output / "diagnostics.json").read_text())
        self.assertFalse(diagnostics["prediction_extension"]["training_only_representation"])
        self.assertFalse(diagnostics["convergence"]["global_optimum_certified"])

    def test_bridge_subprocess_preserves_contract(self):
        result = run_external([sys.executable, str(self.script), "--input", "{input}",
                               "--output", "{output}", "--seed", "{seed}", "--K", "2"],
                              self.arrays, self.root / "external", seed=17, transductive=True, timeout=30)
        self.assertEqual(result["train_labels"].shape, (6,))
        self.assertEqual(result["test_scores"].shape, (4, 2))
        self.assertEqual(result["score_metadata"]["score_type"], "decision")
        self.assertFalse(result["diagnostics"]["external_bridge"]["ground_truth_labels_exported"])

    def test_inductive_input_is_rejected_without_substituting_missing_edges(self):
        inputs = {name: value for name, value in self.arrays.items() if name != "A_test_test"}
        with self.assertRaises(ExternalAdapterError) as raised:
            run_external([sys.executable, str(self.script), "--input", "{input}",
                          "--output", "{output}", "--seed", "{seed}", "--K", "2"],
                         inputs, self.root / "inductive", seed=17, transductive=False, timeout=30)
        self.assertEqual(raised.exception.status, "failed")
        diagnostics = json.loads((self.root / "inductive/diagnostics.json").read_text())
        self.assertFalse(diagnostics["completed"])

    def test_extra_label_file_and_reuse_are_rejected(self):
        output = self.root / "result"
        run(self.input, output, 17, 2)
        with self.assertRaises(FileExistsError):
            run(self.input, output, 17, 2)
        # A prohibited label file is rejected on its filename, never opened.
        (self.input / "y_train.csv").write_text("malformed private labels\n")
        with self.assertRaisesRegex(ValueError, "only the five"):
            run(self.input, self.root / "bad_labels", 17, 2)


if __name__ == "__main__":
    unittest.main()
