"""Synthetic-only plot checks; all figures go to disposable temporary folders."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_results.py"
SPEC = importlib.util.spec_from_file_location("plot_results", SCRIPT)
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


class PlotResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gm3l-synthetic-plot-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def rows(self):
        return [dict(split_id=f"split_{split:03d}", method=method, status="complete", seed=split,
                     protocol="synthetic", log_MSE_zeta=-2.0 - split * .2 - m * .1,
                     MSE_zeta_status="positive", score_type="posterior",
                     alignment="training_hungarian", alignment_source="own_training",
                     mapping_rule="hungarian", unseen_cluster_policy="error",
                     score_aggregation="sum", trace_key=f"synthetic/{split}/{method}")
                for split in range(4) for m, method in enumerate(("Method_A", "Method_B", "Method_C"))]

    def write_rows(self, rows=None, fields=None):
        rows = self.rows() if rows is None else rows
        fields = list(dict.fromkeys(key for row in rows for key in row)) if fields is None else fields
        path = self.root / "synthetic_metrics.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_log_mse_exports_and_tables(self):
        path = self.write_rows()
        output = self.root / "plot"
        manifest = PLOT.plot_results(path, output, synthetic=True, dpi=150)
        self.assertEqual(manifest["paired_n"], 4)
        self.assertEqual(manifest["panels"], ["log_MSE_zeta"])
        self.assertEqual(PLOT.METRICS, ("log_MSE_zeta",))
        self.assertTrue(manifest["synthetic"])
        self.assertEqual(manifest["axis_scale"], "linear")
        self.assertEqual(manifest["metrics_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        for extension in ("pdf", "svg", "png"):
            exported = output / ("synthetic_log_mse_comparison." + extension)
            self.assertGreater(exported.stat().st_size, 1000)
            self.assertEqual(manifest["exports"][exported.name]["sha256"], hashlib.sha256(exported.read_bytes()).hexdigest())
        svg = (output / "synthetic_log_mse_comparison.svg").read_text()
        self.assertIn("Synthetic validation", svg)
        self.assertIn("<text", svg)
        self.assertIn(r"\log(\mathrm{MSE}_{\zeta})", svg)
        self.assertNotIn("AUC", svg)
        self.assertTrue(manifest["quantitative_qa"]["median_artists_match_values"])
        self.assertTrue(manifest["quantitative_qa"]["all_observations_within_axes"])
        with (output / "plot_source_data.csv").open() as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, [
                "split_id", "seed", "protocol", "method", "log_MSE_zeta", "MSE_zeta_status", "status",
                *PLOT.SEMANTIC_FIELDS, "trace_key",
            ])
            source = list(reader)
        self.assertEqual(len(source), 12)
        self.assertEqual(source[0]["trace_key"], "synthetic/0/Method_A")
        with (output / "plot_summary.csv").open() as stream:
            summaries = list(csv.DictReader(stream))
        self.assertAlmostEqual(float(summaries[0]["log_MSE_zeta_mean"]), -2.3)
        self.assertAlmostEqual(float(summaries[0]["log_MSE_zeta_median"]), -2.3)
        self.assertFalse(any("AUC" in key or "ACC" in key or key == "log_mean_MSE_zeta" for key in summaries[0]))
        self.assertEqual(json.loads((output / "plot_manifest.json").read_text()), manifest)

    def test_reads_recorded_logs_without_recomputing_raw_mse(self):
        rows = self.rows()
        for row in rows:
            row.update(MSE_zeta=0.0, log_MSE_zeta=-900 - int(row["seed"]),
                       MSE_zeta_status="underflow", log_mean_MSE_zeta=1234.0, ACC="not required")
        manifest = PLOT.plot_results(self.write_rows(rows), self.root / "underflow", synthetic=True, dpi=150)
        self.assertAlmostEqual(manifest["summaries"][0]["log_MSE_zeta_mean"], -901.5)
        self.assertTrue(manifest["quantitative_qa"]["all_observations_within_axes"])
        self.assertEqual(manifest["paired_n"], 4)

    def test_summary_is_mean_of_logs_not_log_of_mean(self):
        rows = [dict(split_id=str(i), method="A", status="complete", log_MSE_zeta=value,
                     MSE_zeta_status="positive", MSE_zeta=math.exp(value)) for i, value in enumerate((-1., -9.))]
        manifest = PLOT.plot_results(self.write_rows(rows), self.root / "mean", synthetic=True, dpi=150)
        summary = manifest["summaries"][0]
        self.assertEqual(summary["log_MSE_zeta_mean"], -5.)
        self.assertEqual(summary["log_MSE_zeta_median"], -5.)
        self.assertNotAlmostEqual(summary["log_MSE_zeta_mean"], math.log((math.exp(-1.) + math.exp(-9.)) / 2))

    def test_requested_method_order_preserved(self):
        requested = ["Method_C", "Method_A"]
        cohort = PLOT.read_matched_metrics(self.write_rows(), requested)
        self.assertEqual(cohort["methods"], requested)
        self.assertEqual([row["method"] for row in cohort["rows"][:2]], requested)

    def test_incomplete_split_excluded_for_every_method_and_disclosed(self):
        rows = self.rows()
        rows[-1].update(status="mse_unavailable", log_MSE_zeta="", MSE_zeta_status="",
                        reason="Nonposterior score export")
        path = self.write_rows(rows)
        cohort = PLOT.read_matched_metrics(path)
        self.assertEqual(cohort["paired_n"], 3)
        self.assertEqual(len(cohort["rows"]), 9)
        self.assertEqual(len(cohort["excluded_split_records"]), 3)
        self.assertEqual(cohort["excluded_split_records"][-1]["reason"], "Nonposterior score export")
        manifest = PLOT.plot_results(path, self.root / "paired", synthetic=True, dpi=150)
        self.assertEqual(manifest["excluded_split_records"], cohort["excluded_split_records"])
        self.assertIn("Excluded from every method: split_003", (self.root / "paired" / "plot_summary.md").read_text())

    def test_missing_method_cell_is_reported(self):
        cohort = PLOT.read_matched_metrics(self.write_rows(self.rows()[:-1]))
        self.assertEqual(cohort["paired_n"], 3)
        missing = [row for row in cohort["excluded_split_records"] if row["status"] == "missing"]
        self.assertEqual([(row["split_id"], row["method"]) for row in missing], [("split_003", "Method_C")])

    def test_duplicate_records_fail(self):
        rows = self.rows()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            PLOT.read_matched_metrics(self.write_rows(rows + rows[:1]))

    def test_empty_cohort_no_figure(self):
        rows = self.rows()
        for row in rows:
            row["status"] = "mse_unavailable"
        with self.assertRaisesRegex(ValueError, "No jointly complete"):
            PLOT.plot_results(self.write_rows(rows), self.root / "none", synthetic=True)
        self.assertFalse((self.root / "none").exists())

    def test_invalid_logs_rejected_before_output_created(self):
        for invalid in ("nan", "inf", "-inf", "0.001", "", "not a number"):
            with self.subTest(log=invalid):
                rows = self.rows()
                rows[0]["log_MSE_zeta"] = invalid
                with self.assertRaisesRegex(ValueError, "log_MSE_zeta"):
                    PLOT.plot_results(self.write_rows(rows), self.root / "invalid", synthetic=True)
                self.assertFalse((self.root / "invalid").exists())

    def test_exact_zero_is_not_floored_or_dropped_even_in_excluded_split(self):
        for recorded_log in ("", "-inf", "-1000"):
            with self.subTest(recorded_log=recorded_log):
                rows = self.rows()
                rows[0].update(log_MSE_zeta=recorded_log, MSE_zeta_status="zero_in_input", MSE_zeta=0)
                rows[1]["status"] = "mse_unavailable"
                with self.assertRaisesRegex(ValueError, "zero_in_input.*split=split_000.*method=Method_A"):
                    PLOT.plot_results(self.write_rows(rows), self.root / "zero", synthetic=True)
                self.assertFalse((self.root / "zero").exists())

    def test_missing_new_columns_rejects_stale_csv(self):
        for missing in ("log_MSE_zeta", "MSE_zeta_status"):
            with self.subTest(missing=missing):
                rows = self.rows()
                rows[0]["AUC"] = .8
                rows[0]["log_mean_MSE_zeta"] = -1.0
                fields = [key for key in rows[0] if key != missing]
                with self.assertRaisesRegex(ValueError, "missing columns.*" + missing):
                    PLOT.read_matched_metrics(self.write_rows(rows, fields))

    def test_invalid_mse_status_rejected(self):
        rows = self.rows()
        rows[0]["MSE_zeta_status"] = "unknown"
        with self.assertRaisesRegex(ValueError, "invalid MSE_zeta_status"):
            PLOT.read_matched_metrics(self.write_rows(rows))

    def test_zero_log_is_valid_and_constant_values_render(self):
        rows = self.rows()
        for row in rows:
            row["log_MSE_zeta"] = 0.0
        manifest = PLOT.plot_results(self.write_rows(rows), self.root / "constant", synthetic=True, dpi=150)
        self.assertTrue(manifest["quantitative_qa"]["all_observations_within_axes"])

    def test_score_semantics_cannot_change_across_paired_splits(self):
        for field in PLOT.SEMANTIC_FIELDS:
            with self.subTest(field=field):
                rows = self.rows()
                rows[0][field] = "changed"
                with self.assertRaisesRegex(ValueError, "Score semantics change"):
                    PLOT.read_matched_metrics(self.write_rows(rows))

    def test_mixed_protocols_rejected(self):
        rows = self.rows()
        rows[0]["protocol"] = "different"
        with self.assertRaisesRegex(ValueError, "different protocols"):
            PLOT.read_matched_metrics(self.write_rows(rows))

    def test_invalid_method_selection_rejected(self):
        for methods in ([], ["Method_A", "Method_A"], ["missing"]):
            with self.subTest(methods=methods), self.assertRaisesRegex(ValueError, "Methods must be unique"):
                PLOT.read_matched_metrics(self.write_rows(), methods)

    def test_removed_auc_flag_and_api_fail(self):
        path = self.write_rows()
        with self.assertRaises(TypeError):
            PLOT.plot_results(path, self.root / "obsolete", auc_only=True)
        with contextlib.redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit) as error:
            PLOT.main(["--metrics", str(path), "--output", str(self.root / "obsolete"), "--auc-only"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("unrecognized arguments: --auc-only", stderr.getvalue())

    def test_dpi_validation(self):
        with self.assertRaisesRegex(ValueError, "dpi"):
            PLOT.plot_results(self.write_rows(), self.root / "invalid_dpi", dpi=72)


if __name__ == "__main__":
    unittest.main()
