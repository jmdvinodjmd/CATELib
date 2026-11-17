"""NCoRE: treatment-specific branches with conditional interaction modules.

Paper: https://arxiv.org/abs/2103.11175 (Section 3 and Figure 3).
Official code: no public author repository verified in this implementation pass.
This is an independent implementation of the unbalanced architecture, not
NCoRE (balanced): the paper's approximate batch-matching variant is not used.

Shared covariate features feed one branch per base treatment. Each branch is
conditionally transformed by other active treatments, in component-index order.
Active branch predictions are averaged. The no-treatment outcome uses a
separate control head (an explicit extension for a valid empty intervention).
Inactive branches receive no factual gradient. Binary simultaneous treatments
only; mixed outcomes are an extension of scalar prediction. Exact author-code
parity and published accuracy have not been verified.
"""
import numpy as np
import torch
from torch import nn
from catelib.methods.s_learner import SLearner


class RelationNetwork(nn.Module):
    def __init__(self, inputs, treatments, outcomes, hidden_sizes):
        super().__init__()
        width = hidden_sizes[-1] if hidden_sizes else 16
        layers = []
        for size in hidden_sizes:
            layers.extend((nn.Linear(inputs, size), nn.ReLU()))
            inputs = size
        if not hidden_sizes:
            layers.extend((nn.Linear(inputs, width), nn.ReLU()))
        self.shared = nn.Sequential(*layers)
        self.branches = nn.ModuleList([nn.Linear(width, width) for _ in range(treatments)])
        self.interactions = nn.ModuleList([
            nn.ModuleList([nn.Linear(width, width) for _ in range(treatments)])
            for _ in range(treatments)])
        self.outputs = nn.ModuleList([nn.Linear(width, outcomes) for _ in range(treatments)])
        self.control = nn.Linear(width, outcomes)

    def forward(self, X, T):
        shared = self.shared(X)
        predictions = []
        for arm, branch in enumerate(self.branches):
            hidden = torch.relu(branch(shared))
            for other, interaction in enumerate(self.interactions[arm]):
                if other != arm:
                    changed = torch.relu(interaction(hidden))
                    hidden = torch.where(T[:, other:other+1].bool(), changed, hidden)
            predictions.append(self.outputs[arm](hidden)*T[:, arm:arm+1])
        total = torch.stack(predictions).sum(0)
        count = T.sum(1, keepdim=True)
        return torch.where(count > 0, total/count.clamp_min(1), self.control(shared))


class NCoRE(SLearner):
    def _build_network(self):
        return RelationNetwork(self.n_features_in_, self.n_treatments_, self.n_outcomes_, self.hidden_sizes)

    def fit(self, X, Y, *, treatment, validation_data=None):
        if not np.isin(treatment, [0, 1]).all():
            raise ValueError('NCoRE requires binary treatments.')
        return super().fit(X, Y, treatment=treatment, validation_data=validation_data)

    def predict(self, X, *, treatment):
        if not np.isin(treatment, [0, 1]).all():
            raise ValueError('NCoRE requires binary treatments.')
        return super().predict(X, treatment=treatment)
