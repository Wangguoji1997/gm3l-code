import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from gm3l_real.parameters import (BLOCKS, evaluate_parameters, training_oracle,
                                  parameter_deviations, soft_order, plot_parameters)
from gm3l_real.validation import INPUT_NAMES, fingerprint


class ParameterTests(unittest.TestCase):
    def test_oracle_and_full_array_mse(self):
        X=np.array([[0.,0.],[2.,0.],[0.,2.],[0.,4.]])
        y=np.array([0,0,1,1])
        A=np.array([[0,1,1,0],[1,0,0,1],[1,0,0,0],[0,1,0,0]])
        oracle,diagnostics=training_oracle(X,A,y,2)
        np.testing.assert_allclose(oracle["pi"],[[1,.5],[.5,0]])
        self.assertFalse(diagnostics["positive_definite"])
        theta={key: value.copy() for key,value in oracle.items()}
        theta["sigma"][0,0,1]+=2;theta["sigma"][0,1,0]+=2
        self.assertEqual(parameter_deviations(theta,oracle,[0,1])["sigma"],1.)
        theta["mu"]=theta["mu"][:1]
        with self.assertRaises(ValueError):parameter_deviations(theta,oracle,[0,1])

    def test_soft_mapping(self):
        p=np.array([[.1,.9],[.2,.8],[.9,.1],[.8,.2]])
        np.testing.assert_array_equal(soft_order(np.log(p),np.array([0,0,1,1]),2),[1,0])

    def test_zero_plot_and_input_rejections(self):
        rows=[]
        for s in range(3):
            for m in ("SMLE","FINAL"):
                rows.append(dict(split_id=f"split_{s:03d}",method=m,protocol="transductive",status="complete",
                                 alpha=.01,mu=.02,sigma=0. if m=="FINAL" else .02,pi=.001))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/"synthetic.csv"
            def save(records):
                with path.open("w",newline="") as handle:
                    writer=csv.DictWriter(handle,list(rows[0]));writer.writeheader();writer.writerows(records)
            save(rows)
            plot_parameters(path,root/"figures",synthetic=True)
            self.assertTrue((root/"figures"/"parameter_deviations.svg").exists())
            save(rows+[rows[0]])
            with self.assertRaises(ValueError):plot_parameters(path,root/"duplicates",synthetic=True)
            mixed=[dict(row) for row in rows];mixed[0]["protocol"]="inductive";save(mixed)
            with self.assertRaises(ValueError):plot_parameters(path,root/"mixed",synthetic=True)


class ParameterIntegrityTests(unittest.TestCase):
    """Synthetic exports only; never read or overwrite research artifacts."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gm3l-synthetic-parameters-")
        self.root = Path(self.temp.name)
        self.prepared = self.root / "prepared"
        self.runs = self.root / "runs"
        self.prepare_export()

    def tearDown(self):
        self.temp.cleanup()

    def prepare_export(self, split_id="split_000", protocol="transductive"):
        split = self.prepared / split_id
        input_dir = split / "input"
        input_dir.mkdir(parents=True)
        config = dict(K=2, seed=7, protocol=protocol)
        (input_dir / "config.json").write_text(json.dumps(config))
        X = np.array([[0., 0.], [2., 0.], [0., 2.], [0., 4.]])
        A = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
        y = np.array([0, 0, 1, 1])
        for name in INPUT_NAMES:
            np.save(input_dir / (name + ".npy"), X if name.startswith("X_") else A)
        np.savez(split / "evaluation_only.npz", y_train=y, y_test=y)
        theta, _ = training_oracle(X, A, y, 2)
        for method in ("SMLE", "FINAL"):
            run = self.runs / split_id / method
            run.mkdir(parents=True)
            # Native runners do not emit a completed flag.
            diagnostic = dict(status="complete", converged=True, input_sha256=fingerprint(input_dir))
            (run / "diagnostics.json").write_text(json.dumps(diagnostic))
            metadata = dict(alignment="soft", score_type="log_probability", cluster_ids=[0, 1],
                            alignment_source="SMLE training posterior (shared with FINAL)")
            (run / "score_columns.json").write_text(json.dumps(metadata))
            np.savetxt(run / "train_scores.csv", np.log([[.9, .1], [.8, .2], [.2, .8], [.1, .9]]), delimiter=",")
            np.savez(run / "model.npz", **theta)
        return split

    def evaluate(self, name, methods=("SMLE", "FINAL")):
        return evaluate_parameters(self.prepared, self.runs, self.root / name, methods)

    def test_matching_identity_preserves_native_parameter_results(self):
        rows = self.evaluate("matching")
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["status"], "complete")
            for block in BLOCKS:
                self.assertEqual(row[block], 0.)
                self.assertIsNone(row["log_" + block])

    def test_changed_seed_or_arrays_rejects_stale_native_model(self):
        input_dir = self.prepared / "split_000" / "input"
        config_path = input_dir / "config.json"
        original_config = config_path.read_bytes()
        config = json.loads(original_config)
        config["seed"] = 999
        config_path.write_text(json.dumps(config))
        rows = self.evaluate("changed_seed")
        config_path.write_bytes(original_config)
        for name in INPUT_NAMES:
            with self.subTest(input=name):
                path = input_dir / (name + ".npy")
                original = path.read_bytes()
                values = np.load(path, allow_pickle=False)
                if name.startswith("A_"):
                    values[0, 1] = values[1, 0] = 1 - values[0, 1]
                else:
                    values.flat[0] += 1
                np.save(path, values)
                rows.extend(self.evaluate("changed_" + name))
                path.write_bytes(original)
        for row in rows:
            self.assertEqual(row["status"], "failed")
            self.assertIn("input_sha256 does not match", row["reason"])
            self.assertFalse(any(block in row for block in BLOCKS))

    def test_missing_malformed_or_wrong_hash_has_no_parameter_values(self):
        path = self.runs / "split_000" / "SMLE" / "diagnostics.json"
        for index, value in enumerate((None, "", "bad-hash", 123, "a" * 64)):
            with self.subTest(hash=value):
                diagnostic = dict(status="complete", converged=True)
                if value is not None:
                    diagnostic["input_sha256"] = value
                path.write_text(json.dumps(diagnostic))
                row = self.evaluate(f"bad_hash_{index}", ("SMLE",))[0]
                self.assertEqual(row["status"], "failed")
                self.assertIn("input_sha256", row["reason"])
                self.assertFalse(any(block in row for block in BLOCKS))

    def test_native_and_legacy_completion_compatibility(self):
        path = self.runs / "split_000" / "SMLE" / "diagnostics.json"
        identity = fingerprint(self.prepared / "split_000" / "input")
        cases = [dict(status="complete", converged=True), dict(completed=True),
                 dict(status="completed", completed=True, convergence_verified=False,
                      convergence=dict(converged=None, stopping_rule="fixed training budget"))]
        for index, diagnostic in enumerate(cases):
            with self.subTest(diagnostic=diagnostic):
                path.write_text(json.dumps(dict(diagnostic, input_sha256=identity)))
                self.assertEqual(self.evaluate(f"completion_{index}", ("SMLE",))[0]["status"], "complete")

    def test_explicit_failure_overrides_completion(self):
        path = self.runs / "split_000" / "SMLE" / "diagnostics.json"
        identity = fingerprint(self.prepared / "split_000" / "input")
        cases = [dict(status="complete", converged=False), dict(status="complete", completed=False),
                 dict(status="nonconverged", completed=True), dict(status="unknown", completed=True),
                 dict(status="complete", convergence=dict(converged=False))]
        for index, diagnostic in enumerate(cases):
            with self.subTest(diagnostic=diagnostic):
                path.write_text(json.dumps(dict(diagnostic, input_sha256=identity)))
                row = self.evaluate(f"failure_{index}", ("SMLE",))[0]
                self.assertEqual(row["status"], "failed")
                self.assertFalse(any(block in row for block in BLOCKS))

    def test_missing_config_split_is_reported_and_cli_exits_nonzero(self):
        (self.prepared / "split_001" / "input").mkdir(parents=True)
        output = self.root / "cli_missing_config"
        result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "parameter_diagnostics.py"),
                                 "--prepared", str(self.prepared), "--runs", str(self.runs),
                                 "--output", str(output), "--methods", "SMLE"],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2, result.stderr)
        with (output / "parameter_errors.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["status"], "complete")
        self.assertEqual(rows[1]["split_id"], "split_001")
        self.assertEqual(rows[1]["status"], "failed")
        self.assertIn("config.json", rows[1]["reason"])
        self.assertEqual(rows[1]["alpha"], "")

    def test_bad_config_yields_one_failure_per_requested_method(self):
        path = self.prepared / "split_000" / "input" / "config.json"
        for index, content in enumerate(("{", "[]", "{}", '{"protocol":"invalid"}',
                                         '{"protocol":"transductive"}')):
            with self.subTest(config=content):
                path.write_text(content)
                rows = self.evaluate(f"bad_config_{index}")
                self.assertEqual(len(rows), 2)
                self.assertEqual({row["method"] for row in rows}, {"SMLE", "FINAL"})
                self.assertTrue(all(row["status"] == "failed" and row["reason"] for row in rows))
                checks = json.loads((self.root / f"bad_config_{index}" / "oracle_diagnostics.json").read_text())
                self.assertEqual(checks["split_000"]["status"], "undefined")

    def test_unknown_failed_protocol_stays_in_plot_requested_count(self):
        (self.prepared / "split_001").mkdir()
        self.evaluate("plot_input")
        from matplotlib.figure import Figure
        with patch.object(Figure, "savefig"), patch.object(Figure, "text", autospec=True, side_effect=Figure.text) as text:
            plot_parameters(self.root / "plot_input" / "parameter_errors.csv", self.root / "synthetic_plot", synthetic=True)
        labels = [call.args[3] for call in text.call_args_list]
        self.assertIn("1 common complete splits / 2 requested", labels)
        manifest = json.loads((self.root / "synthetic_plot" / "parameter_plot_manifest.json").read_text())
        self.assertEqual(manifest["common_splits"], ["split_000"])
        self.assertTrue(manifest["synthetic"])

    def test_complete_rows_cannot_omit_plot_protocol(self):
        rows = self.evaluate("plot_complete")
        rows[0]["protocol"] = ""
        path = self.root / "missing_complete_protocol.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "valid prediction protocol"):
            plot_parameters(path, self.root / "invalid_plot", synthetic=True)

    def test_mixed_protocols_remain_rejected(self):
        self.prepare_export("split_001", protocol="inductive")
        with self.assertRaisesRegex(ValueError, "different prediction protocols"):
            self.evaluate("mixed_protocols")
        self.assertFalse((self.root / "mixed_protocols" / "parameter_errors.csv").exists())


if __name__ == "__main__":unittest.main()
