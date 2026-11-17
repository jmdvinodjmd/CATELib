"""Joint-outcome S-learner: concatenate covariates and intervention components.

Meta-learner paper: https://arxiv.org/abs/1706.03461
Composite-outcome adaptation: https://arxiv.org/abs/2502.08282
Local source: src/models/baseline_estimators.py and src/utils.py (MLP_Model).

One supervised network learns E[Y|X,T] and is queried at alternative treatment
vectors. Continuous doses are not discretised. This multi-output adaptation
shares hidden features across outcomes; IndependentOutcomes removes that
sharing. Causal interpretation requires consistency, conditional exchangeability
and adequate joint support. Smooth predictions outside support are not evidence
of identification. This implementation uses train-only scaling and Adam rather
than claiming numerical reproduction of the original experiments.
"""
import torch
from torch import nn
from catelib.methods.base import NeuralEstimator



class OutcomeNetwork(nn.Module):
    def __init__(self, n_features, n_treatments, n_outcomes, hidden_sizes):
        super().__init__()
        layers = []
        width = n_features + n_treatments
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, n_outcomes))
        self.layers = nn.Sequential(*layers)

    def forward(self, X, treatment):
        return self.layers(torch.cat((X, treatment), dim=1))


class SLearner(NeuralEstimator):
    """Mixed simultaneous interventions; predict returns (patients, outcomes).

    Binary and continuous treatment columns are used as supplied. For encoded
    categorical interventions, callers must supply valid category combinations.
    Continuous outcomes are standardised using training labels only.
    """
    def __init__(self, hidden_sizes=(16,), outcome_types=None, learning_rate=0.001,
                 weight_decay=0.0, batch_size=64, max_epochs=50, patience=10,
                 random_state=0, device='cpu'):
        self.hidden_sizes = hidden_sizes
        self.outcome_types = outcome_types
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self.device = device

    def _build_network(self):
        return OutcomeNetwork(self.n_features_in_, self.n_treatments_,
                              self.n_outcomes_, self.hidden_sizes)

