"""Separate outcome regression per observed binary treatment combination.

Paper: https://arxiv.org/abs/1706.03461
Base network: src/models/s_learner.py (local implementation).

This is the discrete-arm T-learner extended by treating each combination as
an arm. It deliberately rejects continuous doses: fitting a separate model to
each unique dose is not a useful continuous-treatment baseline. Unobserved
arms raise an error rather than returning arbitrary untrained predictions.
Each arm needs training observations. Arms with held-out validation rows use
early stopping. If validation rows are absent, the local neural estimator uses
the full declared epoch budget and the final weights, monitoring training loss
only. It never borrows target labels from the test set. No cross-arm sharing is
performed; arm-specific fitting details are saved in history_.
"""
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from catelib.methods.s_learner import SLearner
from catelib.training import arrays, fit_estimator


class TLearner(BaseEstimator):
    def __init__(self, estimator=None, outcome_types=None, random_state=0):
        self.estimator = estimator
        self.outcome_types = outcome_types
        self.random_state = random_state

    def fit(self, X, Y, *, treatment, validation_data=None):
        X, Y, T = arrays(X, Y, treatment)
        if not np.isin(T, [0, 1]).all():
            raise ValueError('TLearner supports binary intervention combinations only.')
        if validation_data is None:
            a, b = train_test_split(np.arange(len(X)), test_size=0.2, random_state=self.random_state)
            validation_data = X[b], Y[b], T[b]
            X, Y, T = X[a], Y[a], T[a]
        Xv, Yv, Tv = arrays(*validation_data)
        self.models_ = {}
        self.history_ = []
        self.n_treatments_, self.n_outcomes_ = T.shape[1], Y.shape[1]
        template = self.estimator if self.estimator is not None else SLearner()
        for arm in np.unique(T, axis=0):
            train = np.all(T == arm, axis=1)
            val = np.all(Tv == arm, axis=1)
            model = clone(template).set_params(outcome_types=self.outcome_types)
            arm_treatment = np.zeros((train.sum(), 1))
            if val.any():
                model.fit(X[train], Y[train], treatment=arm_treatment,
                          validation_data=(Xv[val], Yv[val], np.zeros((val.sum(), 1))))
            elif hasattr(model, '_build_network'):
                # Train all observed target patients for the declared budget.
                # Training loss is logged, but does not select an early epoch.
                fit_estimator(model, X[train], Y[train], arm_treatment,
                              validation_data=(X[train], Y[train], arm_treatment), fixed_epochs=True)
            else:
                model.fit(X[train], Y[train], treatment=arm_treatment)
            self.history_.append(dict(arm=arm.tolist(), training_count=int(train.sum()),
                                      validation_count=int(val.sum()),
                                      selection='heldout_validation' if val.any() else 'fixed_epochs_no_validation',
                                      history=getattr(model, 'history_', [])))
            self.models_[tuple(arm)] = model
        if not self.models_:
            raise ValueError('No observed treatment arms could be fitted.')
        return self

    def predict(self, X, *, treatment):
        check_is_fitted(self, 'models_')
        X, T = np.asarray(X), np.asarray(treatment)
        if T.ndim == 1:
            T = np.broadcast_to(T, (len(X), len(T)))
        if T.shape != (len(X), self.n_treatments_):
            raise ValueError('Incorrect treatment shape.')
        result = np.empty((len(X), self.n_outcomes_))
        for arm in np.unique(T, axis=0):
            if tuple(arm) not in self.models_:
                raise ValueError(f'Unsupported treatment arm: {arm.tolist()}')
            selected = np.all(T == arm, axis=1)
            result[selected] = self.models_[tuple(arm)].predict(X[selected], treatment=[0])
        return result

    def effect(self, X, *, treatment, reference):
        return self.predict(X, treatment=treatment) - self.predict(X, treatment=reference)
