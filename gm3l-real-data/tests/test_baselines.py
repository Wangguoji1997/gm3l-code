"""All examples are generated here; no historical datasets or outputs are opened."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gm3l_real.baselines import fit_baseline
from gm3l_real.external import ExternalAdapterError, run_external, validate_model_inputs


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.train = np.array([[-2., 0.], [-2.1, .1], [-1.9, -.1], [2., 0.], [2.1, .1], [1.9, -.1]])
        self.test = np.array([[-2., .2], [2., -.2]])

    def test_no_sklearn_gmm(self):
        with self.assertRaises(ValueError):
            fit_baseline("GMM", self.train, self.test, 2, 3)


class ExternalTests(unittest.TestCase):
    def setUp(self):
        self.inputs = dict(X_train=np.arange(8, dtype=float).reshape(4, 2),
                           X_test=np.array([[1., 2.], [3., 4.]]))
        self.command = ["user-python", "user-adapter.py", "--input", "{input}",
                        "--output", "{output}", "--seed", "{seed}"]

    def _completed_process(self, argv, **kwargs):
        self.assertFalse(kwargs["shell"])
        self.assertFalse(kwargs["check"])
        inp, out = Path(argv[3]), Path(argv[5])
        self.assertEqual({path.name for path in inp.iterdir()}, {"X_train.csv", "X_test.csv"})
        self.assertEqual(argv[-1], "42")
        np.savetxt(out / "train_labels.csv", [0, 0, 1, 1], fmt="%d", delimiter=",")
        np.savetxt(out / "test_labels.csv", [0, 1], fmt="%d", delimiter=",")
        np.savetxt(out / "test_scores.csv", [[0., -1.], [-1., 0.]], delimiter=",")
        (out / "score_columns.json").write_text(json.dumps(dict(cluster_ids=[0, 1], score_type="decision",
            alignment="hard", mapping="hungarian")), encoding="utf-8")
        (out / "diagnostics.json").write_text(json.dumps(dict(status="completed", completed=True,
                                                             convergence_verified=False)), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "mock only", "")

    def test_allowed_export_and_roundtrip_with_mock_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "new"
            with patch("gm3l_real.external.subprocess.run", side_effect=self._completed_process):
                result = run_external(self.command, self.inputs, out, 42)
            self.assertEqual(result["test_scores"].shape, (2, 2))
            self.assertEqual(result["score_metadata"]["score_type"], "decision")
            self.assertFalse(result["diagnostics"]["convergence_verified"])
            self.assertFalse(result["diagnostics"]["external_bridge"]["ground_truth_labels_exported"])

    def test_native_score_type_is_required(self):
        def missing_score_type(argv, **kwargs):
            process = self._completed_process(argv, **kwargs)
            metadata_path = Path(argv[5]) / "score_columns.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            del metadata["score_type"]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            return process

        with tempfile.TemporaryDirectory() as directory:
            with patch("gm3l_real.external.subprocess.run", side_effect=missing_score_type):
                with self.assertRaises(ExternalAdapterError) as raised:
                    run_external(self.command, self.inputs, directory, 42)
            self.assertEqual(raised.exception.status, "invalid_output")
            self.assertIn("score_type", str(raised.exception))

    def test_y_key_is_rejected_before_launch(self):
        with patch("gm3l_real.external.subprocess.run") as process:
            with self.assertRaises(ValueError):
                run_external(self.command, {**self.inputs, "y_train": [1, 2, 1, 2]}, "unused", 42)
            process.assert_not_called()

    def test_inductive_graph_need_not_have_test_test(self):
        inputs = {**self.inputs, "A_train": np.zeros((4, 4)), "A_test_train": np.zeros((2, 4))}
        self.assertNotIn("A_test_test", validate_model_inputs(inputs))
        inputs["A_test_test"] = np.zeros((2, 2))
        with self.assertRaises(ValueError):
            validate_model_inputs(inputs)
        self.assertIn("A_test_test", validate_model_inputs(inputs, transductive=True))

    def test_missing_dependency_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("gm3l_real.external.subprocess.run", side_effect=FileNotFoundError("missing executable")):
                with self.assertRaises(ExternalAdapterError) as raised:
                    run_external(self.command, self.inputs, directory, 42)
            self.assertEqual(raised.exception.status, "missing_dependency")
            self.assertEqual(json.loads((Path(directory) / "diagnostics.json").read_text())["status"], "missing_dependency")

    def test_configured_empty_command_is_missing_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ExternalAdapterError) as raised:
                run_external([], self.inputs, directory, 42)
            self.assertEqual(raised.exception.status, "missing_dependency")

    def test_nonempty_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "keep.txt"
            path.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_external(self.command, self.inputs, directory, 42)
            self.assertEqual(path.read_text(), "preserve")

    def test_bad_output_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("gm3l_real.external.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")):
                with self.assertRaises(ExternalAdapterError) as raised:
                    run_external(self.command, self.inputs, directory, 42)
            self.assertEqual(raised.exception.status, "invalid_output")


if __name__ == "__main__":
    unittest.main()
