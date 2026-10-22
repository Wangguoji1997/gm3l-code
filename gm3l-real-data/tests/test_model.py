"""Small generated-array tests; no experiment data or result files are used."""
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit, logsumexp

from gm3l_real.model import (FitConfig, initialize_theta, setup_adjacency, estep_log,
    log_gaussian_density, final_from_zeta, gaussian_update, star_log_posterior,
    split_posteriors, fit_smle, fit_gmm, gaussian_log_posterior)
from gm3l_real.data_io import Dataset
from gm3l_real.preprocessing import prepare_split
from gm3l_real.runner import run_split, load_inputs


class ModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(125)
        self.X = rng.normal(size=(5, 2))
        self.A = np.array([[0,1,0,1,0], [1,0,0,0,1], [0,0,0,1,0], [1,0,1,0,1], [0,1,0,1,0]])
        self.theta = dict(alpha=np.array([0.4, 0.6]), mu=np.array([[-0.7,0.2], [0.4,0.6]]),
                          sigma=np.array([[[1.1,0.1],[0.1,0.8]], [[0.9,0.2],[0.2,1.2]]]),
                          pi=np.array([[0.75,0.25],[0.25,0.65]]))
        self.theta["pi_logit"] = np.log(self.theta["pi"])-np.log1p(-self.theta["pi"])

    def test_estep_against_full_latent_enumeration(self):
        n, K = len(self.X), 2
        node = np.log(self.theta["alpha"])[None, :]+log_gaussian_density(self.X, self.theta["mu"], self.theta["sigma"])
        assignments = np.array(list(itertools.product(range(K), repeat=n)))
        expected_weights = np.zeros((n, K))
        expected_zeta = np.zeros((n, K))
        successes, failures = np.zeros((K,K)), np.zeros((K,K))
        objective = 0.
        for i in range(n):
            logprob = np.array([node[np.arange(n), y].sum() for y in assignments])
            for j in range(n):
                if i == j:
                    continue
                prob = self.theta["pi"][assignments[:, i], assignments[:, j]]
                logprob += np.log(prob if self.A[i,j] else 1-prob)
            normalizer = logsumexp(logprob)
            objective += normalizer
            weights = np.exp(logprob-normalizer)
            for index, y in enumerate(assignments):
                w = weights[index]
                expected_zeta[i,y[i]] += w
                expected_weights[np.arange(n), y] += w
                for j in range(n):
                    if i != j:
                        r,s = sorted((y[i],y[j]))
                        (successes if self.A[i,j] else failures)[r,s] += w
        B,D = setup_adjacency(self.A)
        actual = estep_log(self.X,B,D,self.theta)
        np.testing.assert_allclose(actual["weights"], expected_weights, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(actual["zeta"], expected_zeta, rtol=1e-12, atol=1e-13)
        self.assertAlmostEqual(actual["objective"], objective, places=11)
        rows,cols = np.triu_indices(K)
        np.testing.assert_allclose(expit(actual["next_pi_logit"])[rows,cols],
                                   (successes/(successes+failures+np.tril(np.ones((K,K)),-1)))[rows,cols], rtol=1e-12)
        np.testing.assert_allclose(star_log_posterior(self.X,self.X,self.A,self.theta,same_nodes=True), actual["log_zeta"], atol=1e-12)

    def test_final_uses_distinct_ordered_nodes(self):
        z = np.array([[0.8,0.2],[0.4,0.6],[0.7,0.3],[0.2,0.8],[0.1,0.9]])
        theta = final_from_zeta(self.X,self.A,z)
        num,den = np.zeros((2,2)),np.zeros((2,2))
        for i in range(5):
            for j in range(5):
                if i != j:
                    w = np.outer(z[i],z[j])
                    num += w*self.A[i,j]
                    den += w
        np.testing.assert_allclose(theta["pi"], num/den)
        np.testing.assert_allclose(theta["alpha"].sum(),1)
        self.assertTrue(np.all(np.linalg.eigvalsh(theta["sigma"])>0))

    def test_tied_covariance_update(self):
        z = np.full((5,2),0.5)
        theta = gaussian_update(self.X,z,tied=True)
        covariance = np.cov(self.X.T,bias=True)
        np.testing.assert_allclose(theta["sigma"],np.stack([covariance,covariance]))

    def test_input_and_graph_validation(self):
        bad = self.A.copy(); bad[0,1]=0
        with self.assertRaises(ValueError):
            setup_adjacency(bad)
        bad = self.A.astype(float);bad[0,1]=np.nan
        with self.assertRaises(ValueError):
            setup_adjacency(bad)

    def test_protocol_changes_only_allowed_neighbors(self):
        data = dict(X_train=self.X[:3],X_train_eval=self.X[:3],X_test=self.X[3:],
                    A_train=self.A[:3,:3],A_test_train=self.A[3:,:3],A_test_test=self.A[3:,3:])
        tr,te = split_posteriors(data,self.theta,"inductive")
        np.testing.assert_allclose(te,star_log_posterior(self.X[3:],self.X[:3],self.A[3:,:3],self.theta))
        tr2,te2 = split_posteriors(data,self.theta,"transductive")
        np.testing.assert_allclose(tr,tr2)
        np.testing.assert_allclose(te2,star_log_posterior(self.X,self.X,self.A,self.theta,same_nodes=True)[3:])

    def test_fit_objective_and_shapes(self):
        rng=np.random.default_rng(321)
        y=np.repeat([0,1],30)
        X=rng.normal(size=(60,2))+np.where(y[:,None]==0,-2,2)
        A=np.zeros((60,60),int);r,c=np.triu_indices(60,1)
        A[r,c]=rng.binomial(1,np.where(y[r]==y[c],0.8,0.2));A[c,r]=A[r,c]
        cfg=FitConfig(max_iter=150,tolerance=1e-5)
        initial=initialize_theta(X,A,2,5,cfg,"x_kmeans")
        fit=fit_smle(X,A,initial,cfg)
        self.assertIsNone(fit["failure"])
        self.assertTrue(fit["converged"])
        self.assertGreaterEqual(np.diff(fit["objective_history"]).min(),-1e-7)
        gmm=fit_gmm(X,2,5,cfg)
        self.assertTrue(gmm["converged"])
        logp,_=gaussian_log_posterior(X,gmm)
        np.testing.assert_allclose(np.exp(logp).sum(1),1)


class PipelineTests(unittest.TestCase):
    def test_synthetic_pipeline_and_label_boundary(self):
        from gm3l_real.evaluation import evaluate_experiment
        from gm3l_real.parameters import evaluate_parameters
        rng=np.random.default_rng(812)
        y=np.repeat([0,1],30)
        X=rng.normal(size=(60,3))+np.where(y[:,None]==0,-2,2)
        A=np.zeros((60,60),int);r,c=np.triu_indices(60,1)
        A[r,c]=rng.binomial(1,np.where(y[r]==y[c],0.8,0.2));A[c,r]=A[r,c]
        dataset=Dataset(X=X,A=A,ids=np.arange(60).astype(str),y=y)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            options=dict(smle=dict(max_iter=250,tolerance=1e-5,seed_offsets=[0],initializers=["x_kmeans"]),
                         gmm=dict(max_iter=250,tolerance=1e-5,seed_offsets=[0]))
            split=prepare_split(dataset,root/"prepared"/"split_000",seed=8,K=2,pca_components=2,method_options=options)
            evaluation=split/"evaluation_only.npz"
            hidden=split/"not_accessible.npz"
            evaluation.rename(hidden)
            reports=run_split(split/"input",root/"runs"/"split_000",["SMLE","FINAL","GMM"])
            hidden.rename(evaluation)
            self.assertTrue(all(r["status"]=="complete" for r in reports.values()),reports)
            summary=evaluate_experiment(root/"prepared",root/"runs",root/"evaluation",["SMLE","FINAL","GMM"])
            self.assertEqual(summary["paired_n"],1)
            self.assertTrue((root/"evaluation"/"metrics.csv").exists())
            parameter_rows=evaluate_parameters(root/"prepared",root/"runs",root/"parameters")
            self.assertTrue(all(row["status"]=="complete" for row in parameter_rows),parameter_rows)
            with self.assertRaises(FileExistsError):
                run_split(split/"input",root/"runs"/"split_000",["GMM"])


if __name__ == "__main__":
    unittest.main()
