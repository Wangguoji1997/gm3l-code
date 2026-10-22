"""Synthetic checks for shared run identity and export validation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gm3l_real.validation import (INPUT_NAMES, completion_error, fingerprint,
                                  validate_scores, verify_input_identity)


class ValidationTests(unittest.TestCase):
    def test_fingerprint_matches_existing_run_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(dict(protocol="inductive", seed=7)))
            for name in INPUT_NAMES[:-1]:
                np.save(root / (name + ".npy"), np.zeros((2, 2)))
            old = hashlib.sha256()
            for path in [root / "config.json"] + [root / (key + ".npy") for key in INPUT_NAMES[:-1]]:
                old.update(path.name.encode())
                old.update(path.read_bytes())
            self.assertEqual(fingerprint(root), old.hexdigest())
            verify_input_identity(dict(input_sha256=old.hexdigest()), fingerprint(root))
            # Evaluation-only artifacts are deliberately never opened by the hash.
            (root / "evaluation_only.npz").write_bytes(b"not model input")
            self.assertEqual(fingerprint(root), old.hexdigest())
            (root / "config.json").write_text(json.dumps(dict(protocol="inductive", seed=999)))
            with self.assertRaisesRegex(ValueError, "does not match"):
                verify_input_identity(dict(input_sha256=old.hexdigest()), fingerprint(root))

    def test_each_model_input_is_bound_to_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(dict(protocol="transductive", seed=7)))
            for name in INPUT_NAMES:
                np.save(root / (name + ".npy"), np.zeros((2, 2)))
            initial = fingerprint(root)
            for name in INPUT_NAMES:
                with self.subTest(name=name):
                    np.save(root / (name + ".npy"), np.ones((2, 2)))
                    self.assertNotEqual(fingerprint(root), initial)
                    np.save(root / (name + ".npy"), np.zeros((2, 2)))
                    self.assertEqual(fingerprint(root), initial)

    def test_completion_contracts(self):
        self.assertIsNone(completion_error(dict(status="complete")))
        self.assertIsNone(completion_error(dict(completed=True)))
        self.assertIsNotNone(completion_error(dict(status="complete"), require_completed=True))
        self.assertIsNotNone(completion_error(dict(completed=True), require_completed=True))
        self.assertIsNone(completion_error(dict(status="completed", completed=True,
                                                convergence_verified=False), require_completed=True))
        for values in ({}, dict(completed=1), dict(status="complete", completed=False),
                       dict(status="nonconverged", completed=True),
                       dict(status="complete", convergence=False)):
            with self.subTest(values=values):
                self.assertIsNotNone(completion_error(values))

    def test_score_validation_preserves_log_zero(self):
        values = np.array([[0., -np.inf], [-np.inf, 0.]])
        for kind in ("log_probability", "logits", "log_score", "log_likelihood"):
            with self.subTest(kind=kind):
                np.testing.assert_array_equal(validate_scores(values, 2, 2, kind, "scores"), values)
        for kind, invalid in (("probability", values), ("decision", values),
                              ("log_probability", [[-np.inf, -np.inf], [0., -1.]]),
                              ("logits", [[np.inf, 0.], [0., -1.]]),
                              ("log_probability", [[np.nan, 0.], [0., -1.]])):
            with self.subTest(kind=kind, invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_scores(invalid, 2, 2, kind, "scores")


if __name__ == "__main__":
    unittest.main()
