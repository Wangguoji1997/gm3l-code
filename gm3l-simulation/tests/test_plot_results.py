"""Input validation tests; these do not render mock scientific figures."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_results.py"
_SPEC = importlib.util.spec_from_file_location("gm3l_plot_results", _SCRIPT)
plot_results = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(plot_results)


class PlotResultTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="gm3l-plot-test-")
        self.directory = Path(self.temporary.name)
        self.config = {"replications": 1, "n_grid": [20]}
        self.rows = [dict(replication=1, n=20, estimator=method, status="ok",
                          mse_alpha=.1, mse_mu=.2, mse_sigma=.3, mse_pi=.4,
                          log_mse_zeta=-10000. if method != "GMM" else -2.)
                     for method in ("SMLE", "ORACLE", "FINAL", "GMM")]

    def tearDown(self):
        self.temporary.cleanup()

    def write_inputs(self):
        (self.directory / "configuration.json").write_text(json.dumps(self.config), encoding="utf-8")
        with (self.directory / "all_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def load(self):
        self.write_inputs()
        return plot_results.load_results(self.directory)

    def test_complete_table_preserves_stable_logs_and_all_values(self):
        config, ns, groups, source = self.load()
        self.assertEqual(config, self.config)
        self.assertEqual(ns, [20])
        self.assertEqual(len(groups), 15)
        self.assertEqual(len(source), 15)
        self.assertEqual(groups["zeta", 20, "SMLE"], [-10000.])
        self.assertEqual(groups["zeta", 20, "GMM"], [-2.])
        self.assertEqual(groups["alpha", 20, "ORACLE"], [float(np.log(.1))])
        self.assertNotIn(("zeta", 20, "ORACLE"), groups)
        self.assertNotIn(("mu", 20, "GMM"), groups)

    def test_missing_cell_is_rejected(self):
        self.rows.pop()
        with self.assertRaisesRegex(ValueError, "Incomplete results"):
            self.load()

    def test_duplicate_cell_is_rejected(self):
        self.rows.append(self.rows[0].copy())
        with self.assertRaisesRegex(ValueError, "Duplicate result cell"):
            self.load()

    def test_non_ok_cell_is_rejected_instead_of_dropped(self):
        self.rows[0]["status"] = "not_converged"
        with self.assertRaisesRegex(ValueError, "Non-ok result cell"):
            self.load()

    def test_unexpected_replication_is_rejected(self):
        self.rows[0]["replication"] = 2
        with self.assertRaisesRegex(ValueError, "Unexpected result cell"):
            self.load()

    def test_zero_parameter_mse_is_not_floored(self):
        self.rows[0]["mse_alpha"] = 0
        with self.assertRaisesRegex(ValueError, "Nonpositive/nonfinite parameter MSE"):
            self.load()

    def test_nonfinite_stable_log_mse_is_not_clipped(self):
        self.rows[0]["log_mse_zeta"] = float("-inf")
        with self.assertRaisesRegex(ValueError, "Invalid log posterior MSE"):
            self.load()

    def test_incomplete_grid_is_rejected(self):
        self.config["n_grid"] = [20, 40]
        with self.assertRaisesRegex(ValueError, "Incomplete results"):
            self.load()

    def test_invalid_grid_is_rejected(self):
        self.config["n_grid"] = [20, 20]
        with self.assertRaisesRegex(ValueError, "configuration.n_grid"):
            self.load()

    def test_missing_column_is_rejected(self):
        for row in self.rows:
            row.pop("log_mse_zeta")
        with self.assertRaisesRegex(ValueError, "Missing result columns"):
            self.load()


if __name__ == "__main__":
    unittest.main()
