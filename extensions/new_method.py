"""Example: register an additive treatment-feature ridge estimator.

Direct use: AdditiveRegression(alpha=1).fit(X, Y, treatment=T).
CLI: --extension extensions.new_method --method additive
A new neural architecture would instead subclass NeuralEstimator, implement its
own _build_network, and declare explicit sklearn constructor parameters.
"""
from catelib.methods.compositional_regression import CompositionalRegression
from catelib.run import METHODS


class AdditiveRegression(CompositionalRegression):
    def __init__(self, alpha=1.0, outcome_types=None):
        super().__init__(order=1, alpha=alpha, outcome_types=outcome_types)


METHODS['additive'] = AdditiveRegression
