"""Configuration, evaluation, and resume checks for the experiment driver."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_simulation as runner
from gm3l.data import make_parameters, simulate_data


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = runner.load_config(runner.ROOT / "configs" / "smoke.json")

    def test_paper_configuration(self):
        paper = runner.load_config(runner.ROOT / "configs" / "paper.json")
        self.assertEqual(paper["data_seed"], 2026101101)
        self.assertEqual(paper["n_grid"], [3200, 6400, 12800, 25600])
        self.assertEqual(paper["replications"], 500)

    def test_invalid_sample_grid(self):
        cfg = {**self.cfg, "n_grid": [400, 200]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                runner.load_config(path)

    def test_alignment_used_only_for_evaluation(self):
        truth = make_parameters()
        labels = np.array([0, 1, 2, 0, 1, 2])
        permutation = np.array([2, 0, 1])
        zeta = np.eye(3)[labels][:, permutation]
        theta = {"alpha": truth["alpha"][permutation], "mu": truth["mu"][permutation],
                 "sigma": truth["sigma"][permutation],
                 "pi": truth["pi"][np.ix_(permutation, permutation)]}
        original = copy.deepcopy(theta)
        metrics = runner.parameter_mse(truth, theta, runner.alignment(labels, zeta))
        self.assertTrue(all(value == 0 for value in metrics.values()))
        for key in theta:
            np.testing.assert_array_equal(theta[key], original[key])

    def test_resume_rejects_changed_configuration(self):
        cfg = {**self.cfg, "backend": "numpy", "save_estimates": False}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            runner.prepare(out, cfg)
            runner.prepare(out, cfg)
            with self.assertRaises(ValueError):
                runner.prepare(out, {**cfg, "data_seed": cfg["data_seed"] + 1})

    def test_summary_retains_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "replicates").mkdir()
            cfg = {**self.cfg, "replications": 1, "n_grid": [200]}
            rows = [dict(replication=1, n=200, estimator=method,
                         status="failed" if method == "SMLE" else "ok")
                    for method in runner.METHODS]
            runner.write_json(out / "replicates" / "rep_0001_n200.json",
                              dict(replication=1, n=200, seconds=0.1, rows=rows))
            report = runner.summarize(out, cfg)
            self.assertTrue(report["complete"])
            self.assertEqual(report["recorded"], 4)
            self.assertEqual(len(report["non_ok"]), 1)
            self.assertIn("failed", (out / "all_results.csv").read_text())

    def test_adjacency_allocation_failure_keeps_independent_methods(self):
        truth = make_parameters()
        labels, x, adjacency = simulate_data(60, truth, np.random.default_rng(41))
        cfg = {**self.cfg, "backend": "numpy"}
        with patch.object(runner.strict_smle, "_off_diagonal", side_effect=MemoryError("allocation")), \
             patch.object(runner, "fit_gmm", return_value={"status": "ok", "log_mse_zeta": -1.}) as gmm:
            rows, _ = runner.fit_one(labels, x, adjacency, truth, 1, 60, cfg)
        status = {row["estimator"]: row for row in rows}
        self.assertEqual(status["ORACLE"]["status"], "ok")
        self.assertEqual(status["SMLE"]["status"], "failed")
        self.assertIn("MemoryError", status["SMLE"]["error"])
        self.assertEqual(status["FINAL"]["status"], "source_smle_failed")
        self.assertEqual(status["GMM"]["status"], "ok")
        gmm.assert_called_once()

    def test_fit_failure_retains_diagnostics(self):
        truth = make_parameters()
        class FailingEngine:
            @staticmethod
            def fit_smle(x, adjacency, theta, config):
                return {**theta, "iterations": 3, "objective_history": [1., 2.],
                        "failure": "numerical failure"}
        with patch.object(runner, "initialize_spectral_theta", return_value=(truth, {"solver": "test"})):
            with self.assertRaises(runner.FitFailure) as caught:
                runner.fit_smle(np.zeros((9, 5)), np.zeros((9, 9)), self.cfg, 1, FailingEngine())
        self.assertEqual(caught.exception.diagnostics["iterations"], 3)
        self.assertEqual(caught.exception.diagnostics["objective_history"], [1., 2.])
        self.assertIn("last_parameters", caught.exception.diagnostics)


if __name__ == "__main__":
    unittest.main()
