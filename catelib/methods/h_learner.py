"""H-learner with component-conditioned, generated outcome networks.

Paper: https://arxiv.org/abs/2502.08282
Paper's advertised code: https://github.com/jmdvinodjmd/HLearner
Local reference: src/models/hypernet_estimators.py and hypernetworks.py.

Treatments can be simultaneous binary, continuous or encoded categorical
components. A shared hypernetwork generates a covariate-to-outcome predictor
for each requested intervention and outcome identity. Factual supervised
loss trains the embeddings and hypernetwork jointly. Predicting an unobserved
combination is computationally possible, but causal interpretation still needs
exchangeability and support or justified structural extrapolation assumptions.
This is a local refactor, not a verified upstream port: it fixes legacy bias
unpacking and uses explicit train-only scaling and configurable Adam training.

Preserves the original 'all' architecture: linear treatment embedding, optional
outcome embedding, ReLU hypernetwork and generated ReLU target MLP. Continuous
doses are embedded directly, with no treatment enumeration or discretisation.
"""
import torch
from torch import nn
from catelib.methods.base import NeuralEstimator



class HyperOutcomeNetwork(nn.Module):
    def __init__(self, n_features, n_treatments, n_outcomes, hidden_sizes,
                 hypernet_sizes, treatment_embedding_dim, outcome_embedding_dim):
        super().__init__()
        self.n_outcomes = n_outcomes
        self.target_sizes = (n_features, *hidden_sizes, 1)
        self.treatment_embedding = nn.Linear(n_treatments, treatment_embedding_dim, bias=False)
        embedding_width = treatment_embedding_dim
        if n_outcomes > 1:
            self.outcome_embedding = nn.Embedding(n_outcomes, outcome_embedding_dim)
            embedding_width += outcome_embedding_dim
        parameter_count = sum((a + 1) * b for a, b in
                              zip(self.target_sizes[:-1], self.target_sizes[1:]))
        layers = []
        for hidden in hypernet_sizes:
            layers.extend([nn.Linear(embedding_width, hidden), nn.ReLU()])
            embedding_width = hidden
        layers.append(nn.Linear(embedding_width, parameter_count))
        self.hypernetwork = nn.Sequential(*layers)

    def target_forward(self, X, parameters):
        """Apply each patient's generated weights; avoid a Python patient loop."""
        values, offset = X, 0
        pairs = list(zip(self.target_sizes[:-1], self.target_sizes[1:]))
        for layer, (n_in, n_out) in enumerate(pairs):
            end = offset + n_in * n_out
            weights = parameters[:, offset:end].reshape(-1, n_out, n_in)
            bias = parameters[:, end:end + n_out]
            values = torch.bmm(weights, values.unsqueeze(-1)).squeeze(-1) + bias
            if layer < len(pairs) - 1:
                values = torch.relu(values)
            offset = end + n_out
        return values

    def forward(self, X, treatment):
        treatment_features = self.treatment_embedding(treatment)
        outputs = []
        for outcome in range(self.n_outcomes):
            embedding = treatment_features
            if self.n_outcomes > 1:
                outcome_features = self.outcome_embedding.weight[outcome].expand(len(X), -1)
                embedding = torch.cat((embedding, outcome_features), dim=1)
            parameters = self.hypernetwork(embedding)
            outputs.append(self.target_forward(X, parameters))
        return torch.cat(outputs, dim=1)


class HLearner(NeuralEstimator):
    """Generate outcome-model weights for arbitrary simultaneous interventions.

    Defaults are deliberately small for local experiments. Outcome embeddings
    identify observed outcomes; they do not support a wholly unseen outcome.
    """
    def __init__(self, hidden_sizes=(16,), hypernet_sizes=(16,),
                 treatment_embedding_dim=4, outcome_embedding_dim=4,
                 outcome_types=None, learning_rate=0.001, weight_decay=0.0,
                 batch_size=64, max_epochs=50, patience=10, random_state=0,
                 device='cpu'):
        self.hidden_sizes = hidden_sizes
        self.hypernet_sizes = hypernet_sizes
        self.treatment_embedding_dim = treatment_embedding_dim
        self.outcome_embedding_dim = outcome_embedding_dim
        self.outcome_types = outcome_types
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self.device = device

    def _build_network(self):
        return HyperOutcomeNetwork(self.n_features_in_, self.n_treatments_,
                                   self.n_outcomes_, self.hidden_sizes,
                                   self.hypernet_sizes, self.treatment_embedding_dim,
                                   self.outcome_embedding_dim)

