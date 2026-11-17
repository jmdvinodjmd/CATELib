"""PCA substitute-confounder outcome regression used by the legacy DEC baseline.

Upstream attribution and MIT notices: see THIRD_PARTY_NOTICES.md.

Paper: https://arxiv.org/abs/1805.06826
Code source: https://github.com/ZhaozhiQIAN/Single-Cause-Perturbation-NeurIPS-2021
Local reference: baselines/dec.py; scp_utils.py::generate_dataset_bmc.

Stages: fit PCA to the training treatment matrix, append its scores to X, then
fit outcome regression on augmented X and T. This reproduces the local
benchmark's PCA construction, not the complete Bayesian deconfounder framework.
PCA alone does NOT guarantee identification of unmeasured confounding.
Compared with the legacy script, PCA is fit on training patients only. At
prediction, scores are computed from the requested treatment vector, matching
the local benchmark convention. This is an explicit modelling convention, not
a claim that intervening on T should change a pre-treatment confounder.
Supports numeric mixed treatments; multi-output regression is an extension.
"""
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from catelib.methods.s_learner import SLearner
from catelib.training import arrays


class Deconfounder(BaseEstimator):
    def __init__(self, estimator=None, n_components=1, outcome_types=None, random_state=0):
        self.estimator = estimator
        self.n_components = n_components
        self.outcome_types = outcome_types
        self.random_state = random_state

    def _augment(self, X, treatment):
        X, T = np.asarray(X), np.asarray(treatment)
        if T.ndim == 1:
            T = np.broadcast_to(T, (len(X), len(T)))
        return np.column_stack((X, self.pca_.transform(T))), T

    def fit(self, X, Y, *, treatment, validation_data=None):
        X, Y, T = arrays(X, Y, treatment)
        if validation_data is None:
            a, b = train_test_split(np.arange(len(X)), test_size=0.2, random_state=self.random_state)
            validation_data = X[b], Y[b], T[b]
            X, Y, T = X[a], Y[a], T[a]
        Xv, Yv, Tv = arrays(*validation_data)
        self.pca_ = PCA(n_components=self.n_components, random_state=self.random_state).fit(T)
        Xa, _ = self._augment(X, T)
        Xva, _ = self._augment(Xv, Tv)
        template = self.estimator if self.estimator is not None else SLearner()
        self.model_ = clone(template).set_params(outcome_types=self.outcome_types)
        self.model_.fit(Xa, Y, treatment=T, validation_data=(Xva, Yv, Tv))
        self.history_ = self.model_.history_
        return self

    def predict(self, X, *, treatment):
        check_is_fitted(self, 'model_')
        Xa, T = self._augment(X, treatment)
        return self.model_.predict(Xa, treatment=T)

    def effect(self, X, *, treatment, reference):
        return self.predict(X, treatment=treatment) - self.predict(X, treatment=reference)
