"""Synthetic fixtures only; these tests never read original research artifacts."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gm3l_real.evaluation import evaluate_experiment, evaluate_one, summarize_rows, training_mapping
from gm3l_real.validation import INPUT_NAMES, fingerprint


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gm3l-synthetic-evaluation-")
        self.root = Path(self.temp.name)
        self.y = np.array([0, 0, 1, 1])

    def tearDown(self):
        self.temp.cleanup()

    def export(self, target=None, *, train=None, test=None, scores=None, columns=None,
               score_type="probability", alignment="hard", mapping="hungarian",
               train_scores=None, **extra):
        target = self.root / "run" if target is None else target
        target.mkdir(parents=True, exist_ok=True)
        train = [8, 8, 3, 3] if train is None else train
        test = [8, 8, 3, 3] if test is None else test
        scores = [[.9, .1], [.8, .2], [.2, .8], [.1, .9]] if scores is None else scores
        columns = [8, 3] if columns is None else columns
        for name, values in (("train_labels", train), ("test_labels", test), ("test_scores", scores)):
            np.savetxt(target / (name + ".csv"), values, delimiter=",")
        if train_scores is not None:
            np.savetxt(target / "train_scores.csv", train_scores, delimiter=",")
        meta = dict(cluster_ids=columns, score_type=score_type, alignment=alignment,
                    mapping=mapping, **extra)
        (target / "score_columns.json").write_text(json.dumps(meta))
        diagnostics = dict(completed=True)
        prepared_input = self.root / "prepared" / target.parent.name / "input"
        if prepared_input.is_dir():
            diagnostics["input_sha256"] = fingerprint(prepared_input)
        (target / "diagnostics.json").write_text(json.dumps(diagnostics))
        return target

    def prepared(self, split="split_000", *, protocol="transductive"):
        folder = self.root / "prepared" / split
        (folder / "input").mkdir(parents=True)
        np.savez(folder / "evaluation_only.npz", y_train=self.y, y_test=self.y)
        (folder / "input" / "config.json").write_text(json.dumps(dict(K=2, seed=int(split[-3:]), protocol=protocol)))
        for name in INPUT_NAMES:
            values = np.arange(8, dtype=float).reshape(4, 2) if name.startswith("X_") else np.zeros((4, 4), dtype=int)
            np.save(folder / "input" / (name + ".npy"), values)

    def test_hard_mapping_and_all_metrics(self):
        result = evaluate_one(self.export(), self.y, self.y, 2)
        self.assertEqual(result["mapping"], {"8": 0, "3": 1})
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["ACC"], 1.0)
        self.assertNotIn("AUC", result)
        self.assertNotIn("class_auc", result)
        self.assertAlmostEqual(result["MSE_zeta"], .025)
        self.assertAlmostEqual(result["log_MSE_zeta"], np.log(.025))
        self.assertEqual(result["MSE_zeta_status"], "positive")

    def test_log_mse_avoids_cancellation_and_underflow(self):
        for score_type in ("logits", "log_probability"):
            for gap in (40., 400., 800., 5000.):
                with self.subTest(score_type=score_type, gap=gap):
                    scores = np.array([[0., -gap], [0., -gap], [-gap, 0.], [-gap, 0.]])
                    target = self.export(scores=scores, score_type=score_type)
                    result = evaluate_one(target, self.y, self.y, 2)
                    expected = -2 * np.logaddexp(0., gap)
                    self.assertAlmostEqual(result["log_MSE_zeta"], expected)
                    self.assertEqual(result["ACC"], 1.)
                    self.assertEqual(result["status"], "complete")
                    if gap == 40.:
                        self.assertAlmostEqual(result["MSE_zeta"] / np.exp(expected), 1.)
                        self.assertEqual(result["MSE_zeta_status"], "positive")
                    else:
                        self.assertEqual(result["MSE_zeta"], 0.)
                        self.assertEqual(result["MSE_zeta_status"], "underflow")

    def test_probability_mse_retains_nonzero_wrong_class_tail(self):
        # p_true has already rounded to one, but the exported wrong-class tail
        # still suffices to recover its error; no floor is introduced.
        q = np.exp(-400.)
        target = self.export(scores=[[1., q], [1., q], [q, 1.], [q, 1.]])
        result = evaluate_one(target, self.y, self.y, 2)
        self.assertAlmostEqual(result["log_MSE_zeta"], -800.)
        self.assertEqual(result["MSE_zeta_status"], "underflow")

    def test_log_mse_uses_training_column_alignment(self):
        scores = np.array([[-400., 0.], [-400., 0.], [0., -400.], [0., -400.]])
        target = self.export(scores=scores, columns=[3, 8], score_type="logits")
        result = evaluate_one(target, self.y, self.y, 2)
        self.assertEqual(result["mapping"], {"3": 1, "8": 0})
        self.assertAlmostEqual(result["log_MSE_zeta"], -800.)

    def test_log_mse_multi_class_agrees_with_direct_error(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        p = np.array([[.6, .3, .1], [.2, .7, .1], [.1, .2, .7],
                      [.1, .2, .7], [.5, .3, .2], [0., 1., 0.]])
        expected = np.mean((p - np.eye(3)[y]) ** 2)
        with np.errstate(divide="ignore"):
            log_p = np.log(p)
        for score_type, scores in (("probability", p), ("log_probability", log_p)):
            with self.subTest(score_type=score_type):
                target = self.export(train=y, test=y, columns=[0, 1, 2],
                                     scores=scores, score_type=score_type)
                result = evaluate_one(target, y, y, 3)
                self.assertAlmostEqual(result["MSE_zeta"], expected)
                self.assertAlmostEqual(result["log_MSE_zeta"], np.log(expected))

    def test_log_mse_many_to_one_and_training_prior_alignment(self):
        p = np.array([[.8, .1, .1], [.7, .2, .1], [.1, .8, .1], [.1, .7, .2]])
        cases = [
            (dict(train=[8, 7, 3, 3], test=[8, 7, 3, 3], columns=[8, 7, 3]),
             np.column_stack((p[:, :2].sum(axis=1), p[:, 2]))),
            (dict(columns=[8, 3, 99], unseen_cluster_policy="training_prior"),
             p[:, :2] + p[:, 2, None] * .5),
        ]
        for options, aligned in cases:
            for score_type, scores in (("probability", p), ("log_probability", np.log(p))):
                with self.subTest(options=options, score_type=score_type):
                    target = self.export(scores=scores, score_type=score_type,
                                         mapping="majority", **options)
                    result = evaluate_one(target, self.y, self.y, 2)
                    expected = np.mean((aligned - np.eye(2)[self.y]) ** 2)
                    self.assertAlmostEqual(result["MSE_zeta"], expected)
                    self.assertAlmostEqual(result["log_MSE_zeta"], np.log(expected))

    def test_soft_expected_overlap_not_hard_contingency(self):
        # Hard majority is identity, but confident minorities reverse soft overlap.
        y = np.array([0, 0, 0, 1, 1, 1])
        train = np.array([0, 0, 1, 1, 1, 0])
        p = np.array([[.51, .49], [.51, .49], [.01, .99], [.49, .51], [.49, .51], [.99, .01]])
        hard = training_mapping(train, y, np.array([0, 1]), 2)
        soft = training_mapping(train, y, np.array([0, 1]), 2, alignment="soft", train_probability=p)
        self.assertEqual(hard, {0: 0, 1: 1})
        self.assertEqual(soft, {0: 1, 1: 0})
        target = self.export(train=train, test=[1, 1, 0, 0], columns=[0, 1],
                             scores=np.log([[.1, .9], [.2, .8], [.8, .2], [.9, .1]]),
                             score_type="log_probability", alignment="soft",
                             train_scores=np.log(p), alignment_source="SMLE training posterior")
        result = evaluate_one(target, y, self.y, 2)
        self.assertEqual(result["mapping"], {"0": 1, "1": 0})
        self.assertAlmostEqual(result["MSE_zeta"], .025)
        self.assertEqual(result["alignment_source"], "SMLE training posterior")

    def test_mapping_does_not_use_test_truth(self):
        target = self.export()
        a = evaluate_one(target, self.y, self.y, 2)
        b = evaluate_one(target, self.y, 1 - self.y, 2)
        self.assertEqual(a["mapping"], b["mapping"])
        self.assertEqual(b["ACC"], 0.)
        self.assertNotEqual(a["MSE_zeta"], b["MSE_zeta"])

    def test_mse_uses_posteriors_not_hard_labels(self):
        target = self.export(test=[8, 8, 8, 8])
        result = evaluate_one(target, self.y, self.y, 2)
        self.assertEqual(result["ACC"], .5)
        self.assertAlmostEqual(result["MSE_zeta"], .025)

    def test_pairwise_vote_fractions_are_not_probabilities(self):
        y = np.array([0,0,1,1,2,2])
        scores = np.array([[1,.5,0],[1,0,.5],[.5,1,0],[0,1,.5],[.5,0,1],[0,.5,1]])
        target = self.export(train=y,test=y,scores=scores,columns=[0,1,2],score_type="vote_fraction")
        result = evaluate_one(target,y,y,3)
        self.assertEqual(result["status"],"mse_unavailable")
        self.assertEqual(result["ACC"],1.)
        self.assertIsNone(result["MSE_zeta"])
        self.assertIsNone(result["log_MSE_zeta"])
        self.assertIsNone(result["MSE_zeta_status"])

    def test_log_mse_invariant_to_row_offsets_and_extreme_logits(self):
        logits = np.array([[1000., -1000.], [999., -999.], [-1000., 1000.], [-999., 999.]])
        first = evaluate_one(self.export(scores=logits, score_type="logits"), self.y, self.y, 2)
        shifted = logits + np.arange(4)[:, None] * 1e4
        second = evaluate_one(self.export(scores=shifted, score_type="logits"), self.y, self.y, 2)
        self.assertEqual(first["log_MSE_zeta"], second["log_MSE_zeta"])
        self.assertEqual(first["MSE_zeta_status"], "underflow")

    def test_exact_zero_one_probabilities_are_recorded(self):
        target = self.export(scores=[[1, 0], [1, 0], [0, 1], [0, 1]])
        result = evaluate_one(target, self.y, self.y, 2)
        self.assertEqual(result["MSE_zeta"], 0.)
        self.assertIsNone(result["log_MSE_zeta"])
        self.assertEqual(result["MSE_zeta_status"], "zero_in_input")

    def test_log_probability_input_zeros_are_not_underflow(self):
        scores = [[0., -np.inf], [0., -np.inf], [-np.inf, 0.], [-np.inf, 0.]]
        result = evaluate_one(self.export(scores=scores, score_type="log_probability"), self.y, self.y, 2)
        self.assertEqual(result["MSE_zeta"], 0.)
        self.assertIsNone(result["log_MSE_zeta"])
        self.assertEqual(result["MSE_zeta_status"], "zero_in_input")
        wrong = evaluate_one(self.export(scores=scores, score_type="log_probability"), self.y, 1 - self.y, 2)
        self.assertEqual(wrong["MSE_zeta"], 1.)
        self.assertEqual(wrong["log_MSE_zeta"], 0.)
        self.assertEqual(wrong["MSE_zeta_status"], "positive")

    def test_missing_test_class_does_not_invalidate_posterior_mse(self):
        result = evaluate_one(self.export(), self.y, [0, 0, 0, 0], 2)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["reason"], "")
        self.assertIsNotNone(result["MSE_zeta"])
        self.assertTrue(np.isfinite(result["log_MSE_zeta"]))

    def test_missing_mapped_class_is_unavailable_without_imputation(self):
        target = self.export(train=[8] * 4, test=[8] * 4, scores=[[.5]] * 4,
                             columns=[8], score_type="decision")
        result = evaluate_one(target, self.y, self.y, 2)
        self.assertEqual(result["status"], "mse_unavailable")
        self.assertIsNone(result["log_MSE_zeta"])
        target = self.export(train=[8] * 4, test=[8] * 4, scores=[[1.]] * 4,
                             columns=[8], score_type="probability")
        result = evaluate_one(target, self.y, self.y, 2)
        self.assertEqual(result["status"], "mse_unavailable")
        self.assertIn("No mapped posterior probability", result["reason"])

    def test_majority_unknown_k_and_declared_native_max(self):
        target = self.export(train=[8, 7, 3, 3], test=[8, 7, 3, 3], columns=[8, 7, 3],
                             scores=[[0, -2, -8], [-2, 0, -8], [-8, -9, 0], [-9, -8, 0]],
                             score_type="decision", mapping="majority", score_aggregation="max")
        result = evaluate_one(target, self.y, self.y, 2)
        self.assertEqual(result["mapping"], {"8": 0, "7": 0, "3": 1})
        self.assertEqual(result["status"], "mse_unavailable")
        self.assertEqual(result["ACC"], 1.)
        self.assertIsNone(result["MSE_zeta"])

    def test_native_scores_do_not_gain_mse_by_aggregation(self):
        target = self.export(train=[8, 7, 3, 3], test=[8, 7, 3, 3], columns=[8, 7, 3],
                             scores=[[0, -2, -8]] * 4, score_type="decision", mapping="majority")
        result = evaluate_one(target, self.y, self.y, 2)
        self.assertEqual(result["status"], "mse_unavailable")
        self.assertIsNone(result["log_MSE_zeta"])

    def test_training_prior_policy_for_new_cluster(self):
        target = self.export(columns=[8, 3, 99], test=[8, 8, 3, 99], mapping="majority",
                             scores=[[.8, .1, .1], [.7, .2, .1], [.1, .8, .1], [.1, .7, .2]],
                             unseen_cluster_policy="training_prior")
        result = evaluate_one(target, self.y, self.y, 2)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["ACC"], .75)

    def test_distance_scores_cannot_be_probabilities(self):
        target = self.export(scores=[[-1, -2]] * 4, score_type="decision")
        result = evaluate_one(target, self.y, self.y, 2)
        self.assertEqual(result["status"], "mse_unavailable")
        self.assertIn("does not declare class posterior probabilities", result["reason"])
        self.assertIsNone(result["log_MSE_zeta"])

    def test_invalid_probabilities_rejected(self):
        target = self.export(scores=[[.8, .8]] * 4)
        with self.assertRaisesRegex(ValueError, "row-normalized"):
            evaluate_one(target, self.y, self.y, 2)

    def test_fractional_labels_not_silently_truncated(self):
        target = self.export(test=[8.5, 8, 3, 3])
        with self.assertRaisesRegex(ValueError, "finite integers"):
            evaluate_one(target, self.y, self.y, 2)

    def test_failed_diagnostics_no_predictions_needed(self):
        target = self.root / "failed"
        target.mkdir()
        (target / "diagnostics.json").write_text(json.dumps(dict(status="failed", error="synthetic failure")))
        self.assertEqual(evaluate_one(target, self.y, self.y, 2)["status"], "failed")

    def test_explicit_failure_cannot_be_overridden_by_completed(self):
        target = self.export()
        cases = [dict(status="nonconverged", completed=True),
                 dict(status="failed", completed=True),
                 dict(status="complete", completed=False),
                 dict(status="complete", converged=False),
                 dict(status="complete", convergence=dict(converged=False)),
                 dict(status="unknown_status", completed=True)]
        for diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                (target / "diagnostics.json").write_text(json.dumps(diagnostic))
                result = evaluate_one(target, self.y, self.y, 2)
                self.assertEqual(result["status"], "failed")
                self.assertNotIn("MSE_zeta", result)

    def test_native_and_fixed_budget_success_remain_compatible(self):
        target = self.export()
        cases = [dict(status="complete", converged=True), dict(completed=True),
                 dict(status="completed", completed=True, convergence_verified=False,
                      convergence=dict(converged=None, stopping_rule="fixed training budget"))]
        for diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                (target / "diagnostics.json").write_text(json.dumps(diagnostic))
                result = evaluate_one(target, self.y, self.y, 2)
                self.assertEqual(result["status"], "complete")
                self.assertAlmostEqual(result["MSE_zeta"], .025)

    def test_batch_rejects_changed_seed_and_model_inputs(self):
        self.prepared()
        input_dir = self.root / "prepared" / "split_000" / "input"
        config_path = input_dir / "config.json"
        config = json.loads(config_path.read_text())
        config["seed"] = 7
        config_path.write_text(json.dumps(config))
        self.export(self.root / "runs" / "split_000" / "A")
        baseline = evaluate_experiment(self.root / "prepared", self.root / "runs", self.root / "valid", ["A"])
        self.assertEqual(baseline["paired_n"], 1)
        config["seed"] = 999
        config_path.write_text(json.dumps(config))
        for suffix in ("seed", "arrays"):
            if suffix == "arrays":
                config["seed"] = 7
                config_path.write_text(json.dumps(config))
                np.save(input_dir / "X_test.npy", np.full((4, 2), 100.))
            summary = evaluate_experiment(self.root / "prepared", self.root / "runs", self.root / suffix, ["A"])
            self.assertEqual(summary["paired_n"], 0)
            self.assertEqual(summary["summary"][0]["evaluation_failed_n"], 1)
            with (self.root / suffix / "metrics.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["status"], "evaluation_failed")
            self.assertIn("input_sha256 does not match", row["reason"])

    def test_batch_rejects_missing_or_invalid_input_hash(self):
        self.prepared()
        target = self.export(self.root / "runs" / "split_000" / "A")
        for index, value in enumerate((None, "bad-hash", 123, "a" * 64)):
            diagnostics = dict(status="complete", completed=True)
            if value is not None:
                diagnostics["input_sha256"] = value
            (target / "diagnostics.json").write_text(json.dumps(diagnostics))
            summary = evaluate_experiment(self.root / "prepared", self.root / "runs", self.root / f"hash_{index}", ["A"])
            self.assertEqual(summary["paired_n"], 0)
            self.assertEqual(summary["summary"][0]["evaluation_failed_n"], 1)

    def test_batch_rejects_different_prepared_directory(self):
        self.prepared()
        self.export(self.root / "runs" / "split_000" / "A")
        other = self.root / "other_prepared" / "split_000"
        (other / "input").mkdir(parents=True)
        original = self.root / "prepared" / "split_000"
        for file in (original / "input").iterdir():
            (other / "input" / file.name).write_bytes(file.read_bytes())
        (other / "evaluation_only.npz").write_bytes((original / "evaluation_only.npz").read_bytes())
        np.save(other / "input" / "A_train.npy", np.ones((4, 4), dtype=int) - np.eye(4, dtype=int))
        summary = evaluate_experiment(other.parent, self.root / "runs", self.root / "wrong_directory", ["A"])
        self.assertEqual(summary["paired_n"], 0)
        self.assertEqual(summary["summary"][0]["evaluation_failed_n"], 1)

    def test_nonconverged_run_is_excluded_from_common_cohort(self):
        self.prepared()
        target = self.export(self.root / "runs" / "split_000" / "A")
        diagnostic = json.loads((target / "diagnostics.json").read_text())
        diagnostic.update(status="nonconverged", converged=False)
        (target / "diagnostics.json").write_text(json.dumps(diagnostic))
        summary = evaluate_experiment(self.root / "prepared", self.root / "runs", self.root / "nonconverged", ["A"])
        self.assertEqual(summary["paired_n"], 0)
        self.assertEqual(summary["summary"][0]["complete_n"], 0)
        self.assertEqual(summary["summary"][0]["failed_n"], 1)

    def test_batch_records_missing_runs_and_common_cohort(self):
        for split in ("split_000", "split_001"):
            self.prepared(split)
            self.export(self.root / "runs" / split / "A")
        self.export(self.root / "runs" / "split_000" / "B")
        summary = evaluate_experiment(self.root / "prepared", self.root / "runs", self.root / "out", ["A", "B"], [0, 1, 2])
        self.assertEqual(summary["common_valid_split_ids"], ["split_000"])
        self.assertEqual([s["paired_n"] for s in summary["summary"]], [1, 1])
        self.assertEqual(summary["summary"][1]["missing_n"], 2)
        with (self.root / "out" / "metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(r["status"] == "missing" for r in rows), 3)
        self.assertNotIn("AUC", rows[0])
        self.assertNotIn("class_auc", rows[0])
        self.assertNotIn("auc_transform", rows[0])

    def test_batch_exports_stable_log_mse_and_json_safe_input_zero(self):
        for index, gap in enumerate((400., 800.)):
            split = f"split_{index:03d}"
            self.prepared(split)
            scores = [[0., -gap], [0., -gap], [-gap, 0.], [-gap, 0.]]
            self.export(self.root / "runs" / split / "A", scores=scores, score_type="logits")
            self.export(self.root / "runs" / split / "B", scores=[[1., 0.]] * 2 + [[0., 1.]] * 2)
        summary = evaluate_experiment(self.root / "prepared", self.root / "runs", self.root / "out", ["A", "B"])
        self.assertEqual(summary["common_valid_split_ids"], ["split_000", "split_001"])
        a, b = summary["summary"]
        self.assertAlmostEqual(a["log_mean_MSE_zeta"], np.logaddexp(-800., -1600.) - np.log(2.))
        self.assertEqual(a["MSE_zeta_underflow_n"], 2)
        self.assertEqual(a["MSE_zeta_zero_in_input_n"], 0)
        self.assertIsNone(b["log_mean_MSE_zeta"])
        self.assertEqual(b["MSE_zeta_underflow_n"], 0)
        self.assertEqual(b["MSE_zeta_zero_in_input_n"], 2)
        with (self.root / "out" / "metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([float(row["log_MSE_zeta"]) for row in rows if row["method"] == "A"], [-800., -1600.])
        self.assertTrue(all(row["log_MSE_zeta"] == "" for row in rows if row["method"] == "B"))
        # No non-standard NaN/Infinity tokens can leak into JSON.
        def reject_nonfinite(value):
            self.fail(f"Non-finite JSON token: {value}")
        saved = json.loads((self.root / "out" / "summary.json").read_text(), parse_constant=reject_nonfinite)
        self.assertEqual(saved["summary"], summary["summary"])

    def test_log_mean_mse_includes_input_zero_and_shared_cohort(self):
        rows = [
            dict(split_id="split_000", method="A", status="complete", ACC=1.,
                 MSE_zeta=0., log_MSE_zeta=-800., MSE_zeta_status="underflow"),
            dict(split_id="split_001", method="A", status="complete", ACC=1.,
                 MSE_zeta=0., log_MSE_zeta=None, MSE_zeta_status="zero_in_input"),
            dict(split_id="split_002", method="A", status="complete", ACC=1.,
                 MSE_zeta=.1, log_MSE_zeta=float(np.log(.1)), MSE_zeta_status="positive"),
            dict(split_id="split_000", method="B", status="complete", ACC=1.,
                 MSE_zeta=.1, log_MSE_zeta=float(np.log(.1)), MSE_zeta_status="positive"),
            dict(split_id="split_001", method="B", status="complete", ACC=1.,
                 MSE_zeta=.1, log_MSE_zeta=float(np.log(.1)), MSE_zeta_status="positive"),
            dict(split_id="split_002", method="B", status="failed"),
        ]
        summary = summarize_rows(rows, ["A", "B"], ["split_000", "split_001", "split_002"])
        self.assertEqual(summary["paired_n"], 2)
        self.assertAlmostEqual(summary["summary"][0]["log_mean_MSE_zeta"], -800. - np.log(2.))
        self.assertAlmostEqual(summary["summary"][1]["log_mean_MSE_zeta"], np.log(.1))
        self.assertIsNone(summary["summary"][0]["log_MSE_zeta_median"])

    def test_mse_unavailable_method_cannot_be_in_paper_cohort(self):
        self.prepared()
        self.export(self.root / "runs" / "split_000" / "P")
        self.export(self.root / "runs" / "split_000" / "D", scores=[[-1., -2.]] * 4, score_type="decision")
        summary = evaluate_experiment(self.root / "prepared", self.root / "runs", self.root / "out", ["P", "D"])
        self.assertEqual(summary["paired_n"], 0)
        self.assertEqual(summary["summary"][1]["mse_unavailable_n"], 1)
        self.assertIsNone(summary["summary"][0]["log_MSE_zeta_mean"])

    def test_cohort_does_not_require_auxiliary_accuracy(self):
        row = dict(split_id="split_000", method="A", status="complete",
                   MSE_zeta=.1, log_MSE_zeta=float(np.log(.1)), MSE_zeta_status="positive")
        summary = summarize_rows([row], ["A"], ["split_000"])
        self.assertEqual(summary["paired_n"], 1)
        self.assertIsNone(summary["summary"][0]["ACC_mean"])
        self.assertAlmostEqual(summary["summary"][0]["log_MSE_zeta_mean"], np.log(.1))

    def test_mixed_protocols_never_aggregated(self):
        for index, protocol in enumerate(("inductive", "transductive")):
            split = f"split_{index:03d}"
            self.prepared(split, protocol=protocol)
            self.export(self.root / "runs" / split / "A")
        summary = evaluate_experiment(self.root / "prepared", self.root / "runs", self.root / "out", ["A"])
        self.assertEqual(summary["paired_n"], 0)
        self.assertEqual(summary["summary"][0]["evaluation_failed_n"], 2)

    def test_duplicate_summary_rows_rejected(self):
        row = dict(split_id="split_000", method="A", status="complete", ACC=1.)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize_rows([row, row], ["A"], ["split_000"])


if __name__ == "__main__":
    unittest.main()
