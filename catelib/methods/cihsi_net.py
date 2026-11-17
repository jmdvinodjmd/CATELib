"""CIHSI-Net: treatment embedding plus barycentric FGW balancing.

Paper: https://arxiv.org/abs/2608.22024 (equations 4-6 and 10-11, Appendix C).
OT library: https://pythonot.github.io/
Official author code: not verified; this is an independent paper-based version.

Three networks learn covariate representations, treatment embeddings and the
outcome. Binary patterns are balanced against a free-support W2 barycenter.
FGW uses Euclidean cross/within costs, eta on the feature term, and squared
differences of within distances. Factual loss uses inverse training-pattern
frequency. This implementation freezes each solved transport plan/barycenter
during differentiation (alternating optimisation, not differentiation through
the OT solver). Missing batch patterns contribute no balancing term, retaining
2^-K weights on present patterns; this is an explicit missing-pattern adaptation.

Binary simultaneous treatments only. Multiple-output and binary-outcome modes
are local extensions; use IndependentOutcomes for separate single-output fits.
Early stopping uses factual validation loss, not the regularised objective.
No published results or author-code numerical parity have been reproduced.
"""
from itertools import combinations
import numpy as np
import torch
from torch import nn
import ot
from catelib.methods.s_learner import SLearner
from catelib.training import fit_estimator


def mlp(inputs, hidden_sizes, outputs):
    layers = []
    for width in hidden_sizes:
        layers.extend((nn.Linear(inputs, width), nn.LeakyReLU(0.01)))
        inputs = width
    layers.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*layers)


def transport_cost(a, b, eta=1.0):
    """Differentiate costs with the solved coupling held fixed."""
    cross = torch.cdist(a, b)
    pa, pb = np.ones(len(a))/len(a), np.ones(len(b))/len(b)
    cost = cross.detach().cpu().double().numpy()
    if eta == 1:
        plan = ot.emd(pa, pb, cost)
        return (cross * torch.as_tensor(plan, dtype=a.dtype, device=a.device)).sum()
    ca, cb = torch.cdist(a, a), torch.cdist(b, b)
    plan = ot.gromov.fused_gromov_wasserstein(
        cost, ca.detach().cpu().double().numpy(), cb.detach().cpu().double().numpy(),
        pa, pb, loss_fun='square_loss', alpha=1-eta)
    coupling = torch.as_tensor(plan, dtype=a.dtype, device=a.device)
    # Contract the four-index GW objective without allocating a 4D tensor.
    structure = ca.square().mean() + cb.square().mean() - 2*(ca @ coupling @ cb.T * coupling).sum()
    return eta*(cross*coupling).sum() + (1-eta)*structure


class BalancedNetwork(nn.Module):
    def __init__(self, inputs, treatments, outcomes, hidden_sizes, representation_dim,
                 embedding_dim, alpha, eta, barycenter_size, ot_iterations, mode):
        super().__init__()
        self.representation = mlp(inputs, hidden_sizes, representation_dim)
        self.normalisation = nn.BatchNorm1d(representation_dim, affine=False)
        self.embedding = mlp(treatments, hidden_sizes, embedding_dim)
        self.outcome = mlp(representation_dim+embedding_dim, hidden_sizes, outcomes)
        self.alpha, self.eta, self.mode = alpha, eta, mode
        self.barycenter_size, self.ot_iterations = barycenter_size, ot_iterations
        self.treatments = treatments

    def configure_training(self, T):
        patterns, counts = torch.unique(T, dim=0, return_counts=True)
        self.register_buffer('patterns', patterns)
        self.register_buffer('inverse_frequency', len(T)/counts.float())

    def represent(self, X):
        representation = self.representation(X)
        if self.mode == 'barycenter':
            if self.training and len(X) == 1:
                return nn.functional.batch_norm(representation, self.normalisation.running_mean,
                                                self.normalisation.running_var, training=False)
            representation = self.normalisation(representation)
        return representation

    def forward(self, X, T):
        return self.outcome(torch.cat((self.represent(X), self.embedding(T)), dim=1))

    def training_loss(self, X, Y, T, kinds):
        representation = self.represent(X)
        scores = self.outcome(torch.cat((representation, self.embedding(T)), dim=1))
        groups, inverse = torch.unique(T, dim=0, return_inverse=True)
        distributions = [representation[inverse == j] for j in range(len(groups))]
        weights = torch.ones(len(X), device=X.device)
        if self.mode == 'barycenter':
            matches = (T[:, None, :] == self.patterns[None, :, :]).all(-1)
            weights = matches.float() @ self.inverse_frequency
        factual = scores.sum()*0
        for j, kind in enumerate(kinds):
            observed = ~torch.isnan(Y[:, j])
            if observed.any():
                prediction = scores[:, j].sigmoid() if kind == 'binary' else scores[:, j]
                factual = factual + (weights[observed]*(prediction[observed]-Y[observed, j]).square()).mean()
        if len(distributions) < 2 or self.alpha == 0:
            return factual
        if self.mode == 'pairwise':
            penalties = [transport_cost(a, b) for a, b in combinations(distributions, 2)]
            balance = sum(penalties)/len(penalties)
        else:
            measures = [r.detach().cpu().double().numpy() for r in distributions]
            masses = [np.ones(len(r))/len(r) for r in measures]
            size = min(self.barycenter_size, len(X))
            initial = representation.detach().cpu().double().numpy()[np.linspace(0, len(X)-1, size, dtype=int)]
            barycenter = ot.lp.free_support_barycenter(measures, masses, initial,
                                                       numItermax=self.ot_iterations)
            anchor = torch.as_tensor(barycenter, dtype=X.dtype, device=X.device)
            balance = sum(transport_cost(r, anchor, self.eta) for r in distributions)/(2**self.treatments)
        return factual + self.alpha*balance


class CIHSINet(SLearner):
    def __init__(self, hidden_sizes=(16,), representation_dim=8, embedding_dim=4,
                 alpha=1.0, eta=0.6, barycenter_size=8, ot_iterations=20,
                 outcome_types=None, learning_rate=0.001, weight_decay=1e-5,
                 batch_size=64, max_epochs=50, patience=10, random_state=0, device='cpu'):
        super().__init__(hidden_sizes, outcome_types, learning_rate, weight_decay,
                         batch_size, max_epochs, patience, random_state, device)
        self.representation_dim = representation_dim
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.eta = eta
        self.barycenter_size = barycenter_size
        self.ot_iterations = ot_iterations

    def _build_network(self):
        return BalancedNetwork(self.n_features_in_, self.n_treatments_, self.n_outcomes_,
                               self.hidden_sizes, self.representation_dim, self.embedding_dim,
                               self.alpha, self.eta, self.barycenter_size, self.ot_iterations, 'barycenter')

    def fit(self, X, Y, *, treatment, validation_data=None):
        if not np.isin(treatment, [0, 1]).all():
            raise ValueError('CIHSI/CISI-Net require binary treatment components.')
        if not 0 < self.eta <= 1 or self.alpha < 0 or min(self.barycenter_size, self.ot_iterations) < 1:
            raise ValueError('Invalid balancing settings.')
        return fit_estimator(self, X, Y, treatment, validation_data)

    def predict(self, X, *, treatment):
        if not np.isin(treatment, [0, 1]).all():
            raise ValueError('CIHSI/CISI-Net require binary treatment components.')
        return super().predict(X, treatment=treatment)
