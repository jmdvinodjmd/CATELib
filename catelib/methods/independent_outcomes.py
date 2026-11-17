"""Fit separate copies of an estimator to isolate outcome sharing.

Paper (xS comparison): https://arxiv.org/abs/2502.08282
Local reference: run_xSLearner.py.

With SLearner as the base estimator this is the paper's independent-outcome
xS baseline, NOT the imputed-effect X-learner. Each outcome gets its own fitted
network, preprocessing and early stopping, using common patient splits. Mixed
interventions are supported whenever the base estimator supports them. This
wrapper changes outcome sharing only; it does not relax causal assumptions.
"""
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from catelib.training import arrays
from catelib.methods.s_learner import SLearner


class IndependentOutcomes(BaseEstimator):
    def __init__(self, estimator=None, outcome_types=None, random_state=0):
        self.estimator = estimator
        self.outcome_types = outcome_types
        self.random_state = random_state

    def fit(self, X, Y, *, treatment, validation_data=None):
        X, Y, T = arrays(X, Y, treatment)
        if validation_data is None:
            train, val = train_test_split(np.arange(len(X)), test_size=0.2,
                                         random_state=self.random_state)
            validation_data = (X[val], Y[val], T[val])
            X, Y, T = X[train], Y[train], T[train]
        Xv, Yv, Tv = arrays(*validation_data)
        kinds = self.outcome_types or ('continuous',) * Y.shape[1]
        if len(kinds) != Y.shape[1] or Yv.shape[1] != Y.shape[1]:
            raise ValueError('Outcome counts must agree.')
        self.models_ = []
        template = self.estimator if self.estimator is not None else SLearner()
        for j, kind in enumerate(kinds):
            model = clone(template).set_params(outcome_types=(kind,))
            model.fit(X, Y[:, j:j+1], treatment=T,
                      validation_data=(Xv, Yv[:, j:j+1], Tv))
            self.models_.append(model)
        self.history_ = [model.history_ for model in self.models_]
        return self

    def predict(self, X, *, treatment):
        check_is_fitted(self, 'models_')
        return np.concatenate([m.predict(X, treatment=treatment) for m in self.models_], axis=1)

    def effect(self, X, *, treatment, reference):
        return self.predict(X, treatment=treatment) - self.predict(X, treatment=reference)
