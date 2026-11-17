"""Explicit treatment-feature regression: an interpretable structural control.

Related S-learner framework: https://arxiv.org/abs/1706.03461
Ridge implementation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

This is our benchmark control, not a claimed reproduction of a named paper.
Fit ridge regression on covariates, treatment main/pair terms, and their
covariate interactions. order=1 assumes additive treatment effects conditional
on X; order=2 permits pair interactions. Continuous doses remain continuous.
Extrapolation relies on this explicit response-surface specification. Binary
outcomes are probability regressions clipped to [0,1], not logistic regression.
"""
from itertools import combinations
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from catelib.training import arrays


class CompositionalRegression(BaseEstimator):
    def __init__(self, order=2, alpha=1.0, outcome_types=None):
        self.order = order
        self.alpha = alpha
        self.outcome_types = outcome_types

    def _features(self, X, T):
        X, T = np.asarray(X, float), np.asarray(T, float)
        if T.ndim == 1:
            T = np.broadcast_to(T, (len(X), len(T)))
        if T.shape != (len(X), self.n_treatments_):
            raise ValueError('Incorrect treatment shape.')
        terms = np.column_stack([np.prod(T[:, subset], axis=1) for subset in self.terms_])
        return np.column_stack((X, terms, (X[:, :, None] * terms[:, None, :]).reshape(len(X), -1)))

    def fit(self, X, Y, *, treatment, validation_data=None):
        X, Y, T = arrays(X, Y, treatment)
        if self.order not in (1, 2):
            raise ValueError('order must be 1 or 2.')
        self.n_treatments_ = T.shape[1]
        self.terms_ = [s for degree in range(1, self.order+1)
                       for s in combinations(range(T.shape[1]), degree)]
        self.outcome_types_ = tuple(self.outcome_types or ('continuous',)*Y.shape[1])
        if len(self.outcome_types_) != Y.shape[1]:
            raise ValueError('Specify one type per outcome.')
        self.scaler_ = StandardScaler().fit(self._features(X, T))
        features = self.scaler_.transform(self._features(X, T))
        self.models_ = []
        for column in Y.T:
            observed = np.isfinite(column)
            if not observed.any():
                raise ValueError('Every outcome needs training labels.')
            self.models_.append(Ridge(alpha=self.alpha).fit(features[observed], column[observed]))
        return self

    def predict(self, X, *, treatment):
        check_is_fitted(self, 'models_')
        features = self.scaler_.transform(self._features(X, treatment))
        result = np.column_stack([m.predict(features) for m in self.models_])
        for j, kind in enumerate(self.outcome_types_):
            if kind == 'binary':
                result[:, j] = result[:, j].clip(0, 1)
        return result

    def effect(self, X, *, treatment, reference):
        return self.predict(X, treatment=treatment) - self.predict(X, treatment=reference)
