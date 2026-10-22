"""Contract tests use tiny arrays constructed here, never existing research data."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from sklearn.decomposition import PCA

from gm3l_real.data_io import Dataset, adjacency_from_pairs, load_dataset, load_embeddings, load_adjacency, save_archive
from gm3l_real.preprocessing import prepare_split, prepare_splits, split_indices


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gm3l_contract_")
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(9)
        X = rng.normal(size=(12, 4))
        upper = np.triu(rng.integers(0, 2, size=(12, 12)), 1)
        self.dataset = Dataset(X=X, A=(upper + upper.T).astype(np.uint8),
                               ids=np.asarray([f"item-{i}" for i in range(12)]),
                               y=np.arange(12) % 2)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, name, **arrays):
        return save_archive(self.root / name, **arrays)

    def test_combined_npz_and_separate_id_alignment(self):
        d = self.dataset
        combined = self.save("dataset.npz", X=d.X, A=d.A, ids=d.ids, y=d.y)
        loaded = load_dataset(combined)
        np.testing.assert_array_equal(loaded.X, d.X)
        permutation = np.arange(12)[::-1]
        x = self.save("embedding.npz", X=d.X, chat_id=d.ids)
        a = self.save("adjacency.npz", A=d.A[np.ix_(permutation, permutation)], ids=d.ids[permutation])
        y = self.save("evaluation.npz", y=d.y[permutation], ids=d.ids[permutation])
        aligned = load_dataset(embeddings_path=x, adjacency_path=a, evaluation_path=y)
        np.testing.assert_array_equal(aligned.A, d.A)
        np.testing.assert_array_equal(aligned.y, d.y)

    def test_id_mismatch_duplicates_and_alias_conflict_are_rejected(self):
        d = self.dataset
        x = self.save("x.npz", X=d.X, ids=d.ids, y=d.y)
        a = self.save("a.npz", A=d.A, ids=np.asarray([f"other-{i}" for i in range(12)]))
        with self.assertRaisesRegex(ValueError, "ID sets differ"):
            load_dataset(embeddings_path=x, adjacency_path=a)
        dup = self.save("dup.npz", X=d.X, ids=np.repeat("same", 12))
        with self.assertRaisesRegex(ValueError, "unique"):
            load_embeddings(dup)
        aliases = self.save("aliases.npz", X=d.X, ids=d.ids, chat_id=d.ids[::-1])
        with self.assertRaisesRegex(ValueError, "disagree"):
            load_embeddings(aliases)

    def test_object_arrays_never_loaded(self):
        # Deliberately malformed tiny local fixture; not an external pickle.
        path = self.root / "unsafe.npz"
        np.savez(path, X=self.dataset.X, ids=np.asarray(self.dataset.ids, dtype=object))
        with self.assertRaises(ValueError):
            load_embeddings(path)
        with self.assertRaisesRegex(ValueError, "Object arrays"):
            self.save("forbidden.npz", values=np.asarray(["x"], dtype=object))

    def test_partial_or_invalid_graph_rejected(self):
        d = self.dataset
        partial = self.save("partial.npz", A=d.A, ids=d.ids, complete=np.asarray(False))
        with self.assertRaisesRegex(ValueError, "Partial"):
            load_adjacency(partial)
        observed = np.ones((12, 12), dtype=bool)
        observed[0, 1] = False
        unobserved = self.save("unobserved.npz", A=d.A, ids=d.ids, observed=observed)
        with self.assertRaisesRegex(ValueError, "Partial"):
            load_adjacency(unobserved)
        for name, graph in (("nonbinary", d.A.astype(float) + 0.5),
                            ("diagonal", d.A + np.eye(12, dtype=np.uint8))):
            path = self.save(name + ".npz", A=graph, ids=d.ids)
            with self.assertRaises(ValueError):
                load_adjacency(path)
        asymmetric = d.A.copy()
        asymmetric[0, 1] = 1 - asymmetric[1, 0]
        with self.assertRaisesRegex(ValueError, "symmetric"):
            load_adjacency(self.save("asymmetric.npz", A=asymmetric, ids=d.ids))

    def test_training_only_pca_and_eval_coordinates(self):
        d = self.dataset
        split = prepare_split(d, self.root / "split", seed=12, train_fraction=.5,
                              pca_components=3, K=2, method_options={"max_iter": 7})
        train, test = split_indices(12, seed=12, train_fraction=.5)
        pca = PCA(n_components=3, svd_solver="full")
        expected_train = pca.fit_transform(d.X[train])
        np.testing.assert_allclose(np.load(split / "input/X_train.npy"), expected_train)
        np.testing.assert_allclose(np.load(split / "input/X_train_eval.npy"), pca.transform(d.X[train]))
        np.testing.assert_allclose(np.load(split / "input/X_test.npy"), pca.transform(d.X[test]))
        with np.load(split / "pca_training_only.npz", allow_pickle=False) as stored:
            np.testing.assert_allclose(stored["mean"], d.X[train].mean(axis=0))
        cfg = json.loads((split / "input/config.json").read_text())
        self.assertEqual(cfg["max_iter"], 7)
        self.assertEqual(cfg["method_options"], {"max_iter": 7})
        self.assertEqual(cfg["protocol"], "transductive")
        np.testing.assert_array_equal(np.load(split / "input/A_test_train.npy"), d.A[np.ix_(test, train)])

    def test_test_features_and_labels_do_not_affect_training_preprocessing(self):
        d = self.dataset
        train, test = split_indices(12, seed=23, train_fraction=.5)
        changed = d.X.copy()
        changed[test] += 10000
        altered = Dataset(X=changed, A=d.A, ids=d.ids, y=1-d.y)
        one = prepare_split(d, self.root / "one", seed=23, K=2, pca_components=2)
        two = prepare_split(altered, self.root / "two", seed=23, K=2, pca_components=2)
        np.testing.assert_array_equal(np.load(one / "input/X_train.npy"), np.load(two / "input/X_train.npy"))
        np.testing.assert_array_equal(np.load(one / "input/A_train.npy"), np.load(two / "input/A_train.npy"))
        self.assertEqual((one / "input/config.json").read_bytes(), (two / "input/config.json").read_bytes())

    def test_label_isolation_and_inductive_graph(self):
        split = prepare_split(self.dataset, self.root / "inductive", seed=4, K=2, protocol="inductive")
        expected = {f"{name}.{ext}" for name in ["X_train", "X_train_eval", "X_test", "A_train", "A_test_train"] for ext in ["npy", "csv"]}
        expected.add("config.json")
        self.assertEqual({path.name for path in (split / "input").iterdir()}, expected)
        self.assertFalse((split / "input/A_test_test.npy").exists())
        with np.load(split / "evaluation_only.npz", allow_pickle=False) as evaluation:
            np.testing.assert_array_equal(evaluation["y_train"], self.dataset.y[evaluation["train_indices"]])
            np.testing.assert_array_equal(evaluation["y_test"], self.dataset.y[evaluation["test_indices"]])
        with self.assertRaises(FileExistsError):
            prepare_split(self.dataset, split, seed=4, K=2)

    def test_fixed_attempt_sequence_and_explicit_k(self):
        config = dict(seed_base=31, replications=3, train_fraction=.5, pca_components=2, K=2)
        paths = prepare_splits(self.dataset, self.root / "runs", config)
        self.assertEqual([path.name for path in paths], ["split_000", "split_001", "split_002"])
        for rep, path in enumerate(paths):
            cfg = json.loads((path / "input/config.json").read_text())
            self.assertEqual(cfg["seed"], 31 + rep)
        with self.assertRaises(FileExistsError):
            prepare_splits(self.dataset, self.root / "runs", config)
        with self.assertRaisesRegex(ValueError, "K must be"):
            prepare_split(self.dataset, self.root / "missing_k", seed=4)
        with self.assertRaisesRegex(ValueError, "override"):
            prepare_split(self.dataset, self.root / "reserved", seed=4, K=2, method_options={"K": 3})
        for invalid_k in (1, 6):
            with self.assertRaisesRegex(ValueError, "2 <= K < n_train"):
                prepare_split(self.dataset, self.root / f"invalid_k_{invalid_k}", seed=4, K=invalid_k)

    def test_runner_allowlist_does_not_open_evaluation_or_extra_label_file(self):
        from gm3l_real.runner import load_inputs
        split = prepare_split(self.dataset, self.root / "allowlist", seed=14, K=2)
        # Poison files in this synthetic test ensure no wildcard-loading occurs.
        (split / "evaluation_only.npz").write_bytes(b"not an archive")
        (split / "input/y_train.npy").write_bytes(b"not a numpy array")
        load = np.load
        opened = []

        def guarded_load(path, *args, **kwargs):
            path = Path(path)
            self.assertEqual(path.parent, split / "input")
            self.assertTrue(path.name.startswith(("X_", "A_")))
            self.assertIs(kwargs.get("allow_pickle"), False)
            opened.append(path.name)
            return load(path, *args, **kwargs)

        with patch("gm3l_real.runner.np.load", side_effect=guarded_load):
            inputs, _ = load_inputs(split / "input")
        self.assertEqual(set(opened), {name + ".npy" for name in inputs})
        self.assertNotIn("y_train", inputs)

    def test_full_graph_uses_eval_coordinates_but_mapping_uses_training_coordinates(self):
        from gm3l_real.model import split_posteriors
        train = np.asarray([[1.0], [2.0], [3.0]])
        train_eval = train + 0.125  # Deliberately distinguish the two interfaces.
        test = np.asarray([[4.0], [5.0]])
        inputs = dict(X_train=train, X_train_eval=train_eval, X_test=test,
                      A_train=np.zeros((3, 3)), A_test_train=np.ones((2, 3)),
                      A_test_test=np.zeros((2, 2)))
        calls = []

        def capture(query, reference, adjacency, theta, *, same_nodes=False):
            calls.append((query.copy(), reference.copy(), adjacency.copy(), same_nodes))
            return np.zeros((len(query), 2))

        with patch("gm3l_real.model.star_log_posterior", side_effect=capture):
            mapped, predicted = split_posteriors(inputs, {}, "transductive")
        np.testing.assert_array_equal(calls[0][0], train)
        np.testing.assert_array_equal(calls[0][1], train)
        np.testing.assert_array_equal(calls[1][0], np.concatenate([train_eval, test]))
        np.testing.assert_array_equal(calls[1][1], np.concatenate([train_eval, test]))
        self.assertEqual(mapped.shape, (3, 2))
        self.assertEqual(predicted.shape, (2, 2))
        calls.clear()
        del inputs["A_test_test"]
        with patch("gm3l_real.model.star_log_posterior", side_effect=capture):
            split_posteriors(inputs, {}, "inductive")
        np.testing.assert_array_equal(calls[1][0], test)
        np.testing.assert_array_equal(calls[1][1], train)
        self.assertFalse(calls[1][3])

    def test_complete_pairs_no_transitive_closure(self):
        ids = np.asarray(["a", "b", "c"])
        path = self.root / "pairs.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id_i", "id_j", "must_link"])
            writer.writerows([["a", "b", "Yes"], ["b", "c", "1"], ["a", "c", "No"]])
        A = adjacency_from_pairs(path, ids)
        np.testing.assert_array_equal(A, [[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id_i", "id_j", "must_link"])
            writer.writerow(["a", "b", "1"])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            adjacency_from_pairs(path, ids)


if __name__ == "__main__":
    unittest.main()
