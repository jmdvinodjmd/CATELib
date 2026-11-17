"""Shared neural estimator behaviour; networks and losses belong to methods."""
from sklearn.base import BaseEstimator
from catelib.training import fit_estimator, predict_estimator


class NeuralEstimator(BaseEstimator):
    """Methods provide _build_network and optional loss hooks, not a fixed MLP."""

    def fit(self, X, Y, *, treatment, validation_data=None):
        return fit_estimator(self, X, Y, treatment, validation_data)

    def predict(self, X, *, treatment):
        return predict_estimator(self, X, treatment)

    def effect(self, X, *, treatment, reference):
        return self.predict(X, treatment=treatment) - self.predict(X, treatment=reference)
