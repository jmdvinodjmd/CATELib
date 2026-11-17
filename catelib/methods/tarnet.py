"""TARNet/CFR with one outcome head per observed binary combination.

Upstream attribution and MIT notices: see THIRD_PARTY_NOTICES.md.

Paper: https://arxiv.org/abs/1606.03976
Original code: https://github.com/clinicalml/cfrnet
This is a local multi-combination adaptation, not a port of the binary code.

A shared representation feeds separate arm heads. balancing=0 gives TARNet;
positive balancing adds mean pairwise W1 representation penalties (CFR-W).
Training updates the factual head only. Missing arms are rejected at prediction
rather than using untrained heads. Binary treatments only; multiple outcomes
share the representation in this explicit extension. Use IndependentOutcomes
when comparing separately fitted scalar-output models.
"""
import numpy as np
import torch
from torch import nn
from catelib.methods.s_learner import SLearner
from catelib.methods.cihsi_net import mlp, transport_cost
from catelib.training import outcome_loss
from itertools import combinations


class ArmNetwork(nn.Module):
    def __init__(self, inputs, outcomes, hidden_sizes, width, alpha, arms):
        super().__init__()
        self.shared = mlp(inputs, hidden_sizes, width)
        self.heads = nn.ModuleList([nn.Linear(width, outcomes) for _ in arms])
        self.register_buffer('arms', torch.as_tensor(arms, dtype=torch.float32))
        self.alpha = alpha

    def outcome(self, representation, T):
        matches = (T[:, None, :] == self.arms[None, :, :]).all(-1)
        if not matches.any(1).all():
            raise ValueError('Unsupported treatment arm for TARNet/CFR.')
        predictions = torch.stack([head(representation) for head in self.heads], dim=1)
        return (predictions*matches[:, :, None]).sum(1)

    def forward(self, X, T):
        return self.outcome(self.shared(X), T)

    def training_loss(self, X, Y, T, kinds):
        representation = self.shared(X)
        loss = outcome_loss(self.outcome(representation, T), Y, kinds)
        if loss is None or self.alpha == 0:
            return loss
        _, inverse = torch.unique(T, dim=0, return_inverse=True)
        groups = [representation[inverse == j] for j in range(int(inverse.max())+1)]
        penalties = [transport_cost(a, b) for a, b in combinations(groups, 2)]
        return loss + (self.alpha*sum(penalties)/len(penalties) if penalties else 0)


class TARNet(SLearner):
    def __init__(self, hidden_sizes=(16,), representation_dim=8, balancing=0.0,
                 outcome_types=None, learning_rate=0.001, weight_decay=0.0,
                 batch_size=64, max_epochs=50, patience=10, random_state=0, device='cpu'):
        super().__init__(hidden_sizes, outcome_types, learning_rate, weight_decay,
                         batch_size, max_epochs, patience, random_state, device)
        self.representation_dim = representation_dim
        self.balancing = balancing

    def fit(self, X, Y, *, treatment, validation_data=None):
        if not np.isin(treatment, [0, 1]).all():
            raise ValueError('TARNet/CFR require binary treatments.')
        if validation_data is None:
            from sklearn.model_selection import train_test_split
            from catelib.training import arrays
            X, Y, treatment = arrays(X, Y, treatment)
            a, b = train_test_split(np.arange(len(X)), test_size=.2, random_state=self.random_state)
            validation_data = X[b], Y[b], treatment[b]
            X, Y, treatment = X[a], Y[a], treatment[a]
        self.observed_arms_ = np.unique(treatment, axis=0)
        return super().fit(X, Y, treatment=treatment, validation_data=validation_data)

    def _build_network(self):
        return ArmNetwork(self.n_features_in_, self.n_outcomes_, self.hidden_sizes,
                           self.representation_dim, self.balancing, self.observed_arms_)
