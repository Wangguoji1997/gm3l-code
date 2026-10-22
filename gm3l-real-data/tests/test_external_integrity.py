"""Synthetic external-adapter regressions; no historical data or runs are read."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gm3l_real.data_io import Dataset
from gm3l_real.evaluation import evaluate_experiment
from gm3l_real.external import ExternalAdapterError, run_external
from gm3l_real.preprocessing import prepare_split
from gm3l_real.runner import run_split


class ExternalIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.inputs = dict(X_train=np.arange(8, dtype=float).reshape(4, 2),
                           X_test=np.array([[1., 2.], [3., 4.]]))
        self.command = ["synthetic-adapter", "{input}", "{output}"]

    def _process(self, diagnostics, *, score_type="decision", train_scores=None, test_scores=None):
        def emit(argv, **kwargs):
            self.assertFalse(kwargs["shell"])
            out = Path(argv[2])
            np.savetxt(out / "train_labels.csv", [0, 0, 1, 1], fmt="%d", delimiter=",")
            np.savetxt(out / "test_labels.csv", [0, 1], fmt="%d", delimiter=",")
            np.savetxt(out / "test_scores.csv", [[0., -1.], [-1., 0.]]
                       if test_scores is None else test_scores, delimiter=",")
            metadata = dict(cluster_ids=[0, 1], score_type=score_type,
                            alignment="hard" if train_scores is None else "soft", mapping="hungarian")
            if train_scores is not None:
                np.savetxt(out / "train_scores.csv", train_scores, delimiter=",")
            (out / "score_columns.json").write_text(json.dumps(metadata), encoding="utf-8")
            (out / "diagnostics.json").write_text(json.dumps(diagnostics), encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, "synthetic adapter", "")
        return emit

    def test_conflicting_or_unknown_completion_is_rejected_with_native_diagnostics(self):
        cases = [
            dict(status="nonconverged", completed=True),
            dict(status="failed", completed=True),
            dict(status="unrecognized", completed=True),
            dict(status="completed", completed=False),
            dict(status="completed", completed=True, converged=False),
            dict(status="completed", completed=True, convergence=False),
            dict(status="completed", completed=True, convergence=dict(converged=False)),
            dict(status="completed"),
            dict(completed=True),
            dict(status="completed", completed=1),
        ]
        for diagnostics in cases:
            with self.subTest(diagnostics=diagnostics), tempfile.TemporaryDirectory() as directory:
                with patch("gm3l_real.external.subprocess.run", side_effect=self._process(diagnostics)):
                    with self.assertRaises(ExternalAdapterError) as raised:
                        run_external(self.command, self.inputs, directory, 42)
                self.assertEqual(raised.exception.status, "failed")
                self.assertEqual(raised.exception.diagnostics["adapter_diagnostics"], diagnostics)
                recorded = json.loads((Path(directory) / "diagnostics.json").read_text(encoding="utf-8"))
                self.assertEqual(recorded["adapter_diagnostics"], diagnostics)
                self.assertFalse(recorded["completed"])
                self.assertIn("External adapter did not complete", recorded["error"])
                if diagnostics.get("converged") is False or "convergence" in diagnostics:
                    self.assertIn("nonconvergence", recorded["error"])

    def test_success_and_unknown_fixed_budget_convergence_are_preserved(self):
        cases = [dict(status=status, completed=True)
                 for status in ("complete", "completed", "ok", "success", "converged")]
        cases += [dict(status="complete", completed=True, convergence_verified=False),
                  dict(status="complete", completed=True, convergence=dict(converged=None, budget=10)),
                  dict(status="complete", completed=True, converged=True)]
        for diagnostics in cases:
            with self.subTest(diagnostics=diagnostics), tempfile.TemporaryDirectory() as directory:
                with patch("gm3l_real.external.subprocess.run", side_effect=self._process(diagnostics)):
                    result = run_external(self.command, self.inputs, directory, 42)
                self.assertEqual({key: value for key, value in result["diagnostics"].items()
                                  if key != "external_bridge"}, diagnostics)
                self.assertEqual(result["score_type"], "decision")

    def test_log_types_accept_exact_zero_probability_in_train_and_test(self):
        training = np.array([[0., -np.inf], [0., -np.inf], [-np.inf, 0.], [-np.inf, 0.]])
        testing = np.array([[0., -np.inf], [-np.inf, 0.]])
        for score_type in ("log_probability", "logits", "log_likelihood", "log_score"):
            with self.subTest(score_type=score_type), tempfile.TemporaryDirectory() as directory:
                process = self._process(dict(status="complete", completed=True), score_type=score_type,
                                        train_scores=training, test_scores=testing)
                with patch("gm3l_real.external.subprocess.run", side_effect=process):
                    result = run_external(self.command, self.inputs, directory, 42)
                np.testing.assert_array_equal(result["train_scores"], training)
                np.testing.assert_array_equal(result["test_scores"], testing)
                np.testing.assert_array_equal(np.loadtxt(Path(directory) / "test_scores.csv", delimiter=","), testing)

    def test_invalid_numeric_scores_and_unknown_types_are_rejected_for_both_splits(self):
        diagnostics = dict(status="complete", completed=True)
        invalid = [(kind, value) for kind in ("log_probability", "logits", "log_likelihood", "log_score")
                   for value in ([np.nan, 0.], [np.inf, 0.], [-np.inf, -np.inf])]
        invalid += [(kind, [-np.inf, 0.]) for kind in ("probability", "decision", "native")]
        invalid += [("probability", [-.1, 1.1]), ("probability", [.2, .2]), ("unknown", [0., 1.])]
        for score_type, row in invalid:
            for bad_split in ("train", "test"):
                with self.subTest(score_type=score_type, row=row, split=bad_split), tempfile.TemporaryDirectory() as directory:
                    training = np.tile([.5, .5], (4, 1))
                    testing = np.tile([.5, .5], (2, 1))
                    (training if bad_split == "train" else testing)[0] = row
                    process = self._process(diagnostics, score_type=score_type,
                                            train_scores=training, test_scores=testing)
                    with patch("gm3l_real.external.subprocess.run", side_effect=process):
                        with self.assertRaises(ExternalAdapterError) as raised:
                            run_external(self.command, self.inputs, directory, 42)
                    self.assertEqual(raised.exception.status, "invalid_output")
                    self.assertEqual(raised.exception.diagnostics["adapter_diagnostics"], diagnostics)
                    if score_type != "unknown":
                        self.assertIn(bad_split + "_scores.csv", str(raised.exception))


# This process uses only exported feature CSVs. Labels below are synthetic
# predictions derived from feature signs, never evaluation truth or archives.
SYNTHETIC_ADAPTER = """
import json
from pathlib import Path
import sys
import numpy as np

source, destination = Path(sys.argv[1]), Path(sys.argv[2])
assert sorted(p.name for p in source.iterdir()) == ["X_test.csv", "X_train.csv"]
diagnostics, score_type = json.loads(sys.argv[3]), sys.argv[4]
for split in ("train", "test"):
    features = np.loadtxt(source / ("X_" + split + ".csv"), delimiter=",", ndmin=2)
    labels = (features[:, 0] > 0).astype(int)
    scores = np.full((len(labels), 2), -np.inf)
    scores[np.arange(len(labels)), labels] = 7.0 if score_type == "logits" else 0.0
    np.savetxt(destination / (split + "_labels.csv"), labels, fmt="%d", delimiter=",")
    np.savetxt(destination / (split + "_scores.csv"), scores, delimiter=",")
metadata = dict(cluster_ids=[0, 1], score_type=score_type, alignment="soft", mapping="hungarian")
(destination / "score_columns.json").write_text(json.dumps(metadata), encoding="utf-8")
(destination / "diagnostics.json").write_text(json.dumps(diagnostics), encoding="utf-8")
"""


class ExternalPipelineIntegrityTests(unittest.TestCase):
    def _run_pipeline(self, root, diagnostics, score_type="log_probability"):
        y = np.repeat([0, 1], 6)
        dataset = Dataset(X=np.column_stack((2 * y - 1, np.arange(12))),
                          A=np.zeros((12, 12), dtype=int), ids=np.arange(12).astype(str), y=y)
        command = [sys.executable, "-c", SYNTHETIC_ADAPTER, "{input}", "{output}",
                   json.dumps(diagnostics), score_type]
        options = dict(external=dict(Synthetic=dict(command=command, input_keys=["X_train", "X_test"], timeout=30)))
        split = prepare_split(dataset, root / "prepared" / "split_000", seed=8, K=2,
                              protocol="inductive", method_options=options)
        reports = run_split(split / "input", root / "runs" / "split_000", ["Synthetic"])
        summary = evaluate_experiment(root / "prepared", root / "runs", root / "evaluation", ["Synthetic"])
        return reports["Synthetic"], summary

    def test_real_process_nonconvergence_with_completed_true_never_enters_paired_cohort(self):
        for diagnostics in (dict(status="nonconverged", completed=True),
                            dict(status="completed", completed=True, converged=False)):
            with self.subTest(diagnostics=diagnostics), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                report, summary = self._run_pipeline(root, diagnostics)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["adapter_diagnostics"], diagnostics)
                self.assertEqual(len(report["input_sha256"]), 64)
                self.assertEqual(summary["paired_n"], 0)
                self.assertEqual(summary["summary"][0]["failed_n"], 1)
                self.assertIsNone(summary["summary"][0]["ACC_mean"])
                self.assertIsNone(summary["summary"][0]["MSE_zeta_mean"])
                self.assertEqual(summary["excluded_split_records"][0]["status"], "failed")
                self.assertNotIn("ACC", summary["excluded_split_records"][0])

    def test_real_process_valid_log_scores_keep_minus_inf_and_zero_mse(self):
        diagnostics = dict(status="completed", completed=True, convergence_verified=False,
                           convergence=dict(converged=None, fixed_budget=5))
        for score_type in ("log_probability", "logits"):
            with self.subTest(score_type=score_type), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                report, summary = self._run_pipeline(root, diagnostics, score_type)
                self.assertEqual(report["status"], "complete")
                self.assertFalse(report["convergence_verified"])
                self.assertEqual(report["convergence"], diagnostics["convergence"])
                self.assertEqual(summary["paired_n"], 1)
                self.assertEqual(summary["summary"][0]["complete_n"], 1)
                self.assertEqual(summary["summary"][0]["ACC_mean"], 1.0)
                self.assertEqual(summary["summary"][0]["MSE_zeta_mean"], 0.0)
                self.assertEqual(summary["summary"][0]["MSE_zeta_zero_in_input_n"], 1)
                self.assertIsNone(summary["summary"][0]["log_MSE_zeta_mean"])
                for split in ("train", "test"):
                    scores = np.loadtxt(root / "runs" / "split_000" / "Synthetic" / (split + "_scores.csv"), delimiter=",")
                    self.assertEqual(np.isneginf(scores).sum(), len(scores))
                    np.testing.assert_array_equal(scores[np.isfinite(scores)],
                                                  np.full(len(scores), 7.0 if score_type == "logits" else 0.0))


if __name__ == "__main__":
    unittest.main()
