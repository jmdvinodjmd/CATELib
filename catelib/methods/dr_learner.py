"""Cross-fitted multi-arm doubly robust outcome regression.

Paper: https://arxiv.org/abs/2004.14497 (Kennedy, doubly robust CATE estimation).
Reference library: https://github.com/py-why/EconML (not imported here).

Independent implementation: encode binary combinations as categorical arms,
cross-fit outcome means and multinomial propensities, form each arm's AIPW
pseudo-outcome, then regress it on X. Differences of these regressions estimate
arm contrasts. Uses ridge nuisances/final regressions and logistic propensities
to keep the baseline inexpensive and inspectable. Propensity clipping is
explicit. Binary outcome means are clipped at prediction. Requires observed
arms and overlap; it cannot identify a missing arm. Not a forest DR learner.
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from catelib.training import arrays


class DRLearner(BaseEstimator):
    def __init__(self, n_splits=2, alpha=1.0, min_propensity=0.01,
                 outcome_types=None, random_state=0):
        self.n_splits = n_splits
        self.alpha = alpha
        self.min_propensity = min_propensity
        self.outcome_types = outcome_types
        self.random_state = random_state

    def fit(self, X, Y, *, treatment, validation_data=None):
        X, Y, T = arrays(X, Y, treatment)
        if not np.isin(T, [0, 1]).all() or not np.isfinite(Y).all():
            raise ValueError('DR requires binary treatments and complete outcome labels.')
        if not 0 < self.min_propensity < 1:
            raise ValueError('min_propensity must be between zero and one.')
        self.arms_, ids, counts = np.unique(T, axis=0, return_inverse=True, return_counts=True)
        if len(counts) < 2 or min(counts) < self.n_splits or self.n_splits < 2:
            raise ValueError('Every observed arm needs at least n_splits patients, with at least two arms.')
        pseudo = np.empty((len(X), len(counts), Y.shape[1]))
        splitter = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        for train, held in splitter.split(X, ids):
            propensity = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500,
                                       random_state=self.random_state)).fit(X[train], ids[train])
            probabilities = propensity.predict_proba(X[held])
            for arm in range(len(counts)):
                selected = train[ids[train] == arm]
                nuisance = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha)).fit(X[selected], Y[selected])
                mean = nuisance.predict(X[held])
                if mean.ndim == 1:
                    mean = mean[:, None]
                weight = (ids[held] == arm)/np.maximum(probabilities[:, arm], self.min_propensity)
                pseudo[held, arm] = mean + weight[:, None]*(Y[held]-mean)
        self.models_ = [make_pipeline(StandardScaler(), Ridge(alpha=self.alpha)).fit(X, pseudo[:, arm])
                        for arm in range(len(counts))]
        self.outcome_types_ = tuple(self.outcome_types or ('continuous',)*Y.shape[1])
        self.n_outcomes_ = Y.shape[1]
        return self

    def predict(self, X, *, treatment):
        check_is_fitted(self, 'models_')
        X, T = np.asarray(X), np.asarray(treatment)
        if T.ndim == 1:
            T = np.broadcast_to(T, (len(X), len(T)))
        if T.shape != (len(X), self.arms_.shape[1]):
            raise ValueError('Incorrect treatment shape.')
        result = np.empty((len(X), self.n_outcomes_))
        for arm in np.unique(T, axis=0):
            match = np.flatnonzero(np.all(self.arms_ == arm, axis=1))
            if not len(match):
                raise ValueError(f'Unsupported treatment arm: {arm.tolist()}')
            selected = np.all(T == arm, axis=1)
            result[selected] = self.models_[match[0]].predict(X[selected]).reshape(selected.sum(), -1)
        for j, kind in enumerate(self.outcome_types_):
            if kind == 'binary':
                result[:, j] = result[:, j].clip(0, 1)
        return result

    def effect(self, X, *, treatment, reference):
        return self.predict(X, treatment=treatment)-self.predict(X, treatment=reference)
