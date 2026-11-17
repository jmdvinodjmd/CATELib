"""Task-embedding causal-effect VAE (independent implementation).

Paper: https://doi.org/10.1145/3308558.3313744
Title: Multiple Treatment Effect Estimation using Deep Generative Model with
Task Embedding (Saini et al., WWW 2019).
Official code: not verified. Related CEVAE code: https://github.com/AMLab-Amsterdam/CEVAE

Shared linear treatment embeddings enter q(y|x,t), q(z|x,t,y), and p(y|z,t).
Training uses a factual-data ELBO (Gaussian p(z), Gaussian covariates,
Bernoulli treatments, outcome-specific likelihood) plus auxiliary treatment
and outcome likelihoods. Prediction integrates q(t|x), q(y|x,t), q(z|x,t,y),
then evaluates the decoder under the requested intervention in marginal mode.
The default paper mode instead replaces t in both encoder and decoder with the
requested intervention, as described in Section 3.2. Marginal mode must be
recorded separately as TECE-VAE-marginal.

Continuous covariates and binary treatments only in this version; Gaussian
covariate likelihood is not a categorical/proxy likelihood implementation.
Multiple outputs assume conditional factorisation and are an extension; use
IndependentOutcomes for scalar fits. Missing outcomes are rejected. Latent
confounder recovery depends on model assumptions, not just fitting a VAE.
Fixed Monte Carlo draws make evaluation deterministic and comparable across
interventions. No author-code or published-result parity is claimed.
"""
import numpy as np
import torch
from torch import nn
from catelib.methods.s_learner import SLearner


def network(inputs, hidden_sizes, outputs):
    layers = []
    for width in hidden_sizes:
        layers.extend((nn.Linear(inputs, width), nn.ELU()))
        inputs = width
    layers.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*layers)


class TaskEmbeddingVAE(nn.Module):
    def __init__(self, p, k, m, hidden_sizes, latent_dim, embedding_dim,
                 outcome_types, prediction_samples, random_state, inference):
        super().__init__()
        self.k, self.m, self.latent_dim = k, m, latent_dim
        self.kinds = outcome_types
        self.inference = inference
        self.embedding = nn.Linear(k, embedding_dim, bias=False)
        self.qt = network(p, hidden_sizes, k)
        self.qy = network(p+embedding_dim, hidden_sizes, m)
        self.qz = network(p+embedding_dim+m, hidden_sizes, 2*latent_dim)
        self.px = network(latent_dim, hidden_sizes, 2*p)
        self.pt = network(latent_dim, hidden_sizes, k)
        self.py = network(latent_dim+embedding_dim, hidden_sizes, m)
        generator = torch.Generator().manual_seed(random_state)
        self.register_buffer('uniform_t', torch.rand(prediction_samples, k, generator=generator))
        self.register_buffer('uniform_y', torch.rand(prediction_samples, m, generator=generator))
        self.register_buffer('noise_y', torch.randn(prediction_samples, m, generator=generator))
        self.register_buffer('noise_z', torch.randn(prediction_samples, latent_dim, generator=generator))

    def posterior(self, X, T, Y):
        mu, logvar = self.qz(torch.cat((X, self.embedding(T), Y), dim=1)).chunk(2, dim=1)
        return mu, logvar.clamp(-10, 10)

    def outcome_nll(self, scores, Y):
        values = []
        for j, kind in enumerate(self.kinds):
            if kind == 'binary':
                values.append(nn.functional.binary_cross_entropy_with_logits(scores[:, j], Y[:, j]))
            else:
                values.append(0.5*(scores[:, j]-Y[:, j]).square().mean())
        return sum(values)

    def training_loss(self, X, Y, T, kinds):
        mu, logvar = self.posterior(X, T, Y)
        z = mu + (logvar*0.5).exp()*torch.randn_like(mu)
        xmean, xlogvar = self.px(z).chunk(2, dim=1)
        xlogvar = xlogvar.clamp(-10, 10)
        xloss = 0.5*(xlogvar + (X-xmean).square()*(-xlogvar).exp()).sum(1).mean()
        tloss = nn.functional.binary_cross_entropy_with_logits(self.pt(z), T, reduction='none').sum(1).mean()
        yloss = self.outcome_nll(self.py(torch.cat((z, self.embedding(T)), dim=1)), Y)
        kl = -0.5*(1+logvar-mu.square()-logvar.exp()).sum(1).mean()
        auxiliary = nn.functional.binary_cross_entropy_with_logits(self.qt(X), T, reduction='none').sum(1).mean()
        auxiliary = auxiliary + self.outcome_nll(self.qy(torch.cat((X, self.embedding(T)), dim=1)), Y)
        return xloss + tloss + yloss + kl + auxiliary

    def forward(self, X, intervention):
        treatment_probability = self.qt(X).sigmoid()
        predictions = []
        for s in range(len(self.uniform_t)):
            T = intervention if self.inference == 'paper' else (treatment_probability > self.uniform_t[s]).to(X.dtype)
            yscore = self.qy(torch.cat((X, self.embedding(T)), dim=1))
            draws = []
            for j, kind in enumerate(self.kinds):
                if kind == 'binary':
                    draws.append((yscore[:, j].sigmoid() > self.uniform_y[s, j]).to(X.dtype))
                else:
                    draws.append(yscore[:, j]+self.noise_y[s, j])
            mu, logvar = self.posterior(X, T, torch.stack(draws, dim=1))
            z = mu + (0.5*logvar).exp()*self.noise_z[s]
            scores = self.py(torch.cat((z, self.embedding(intervention)), dim=1))
            means = torch.stack([scores[:, j].sigmoid() if kind == 'binary' else scores[:, j]
                                 for j, kind in enumerate(self.kinds)], dim=1)
            predictions.append(means)
        result = torch.stack(predictions).mean(0)
        # Shared prediction helper expects logits for binary outcomes.
        return torch.stack([torch.logit(result[:, j].clamp(1e-6, 1-1e-6))
                            if kind == 'binary' else result[:, j]
                            for j, kind in enumerate(self.kinds)], dim=1)


class TECEVAE(SLearner):
    def __init__(self, hidden_sizes=(16,), latent_dim=4, embedding_dim=4,
                 prediction_samples=8, outcome_types=None, learning_rate=0.001,
                 weight_decay=0.0, batch_size=64, max_epochs=50, patience=10,
                 random_state=0, device='cpu', inference='paper'):
        super().__init__(hidden_sizes, outcome_types, learning_rate, weight_decay,
                         batch_size, max_epochs, patience, random_state, device)
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.prediction_samples = prediction_samples
        self.inference = inference

    def _build_network(self):
        return TaskEmbeddingVAE(self.n_features_in_, self.n_treatments_, self.n_outcomes_,
                                self.hidden_sizes, self.latent_dim, self.embedding_dim,
                                self.outcome_types_, self.prediction_samples, self.random_state, self.inference)

    def fit(self, X, Y, *, treatment, validation_data=None):
        if not np.isin(treatment, [0, 1]).all():
            raise ValueError('TECE-VAE requires binary treatment components.')
        if not np.isfinite(Y).all() or (validation_data is not None and not np.isfinite(validation_data[1]).all()):
            raise ValueError('TECE-VAE currently requires complete outcome labels.')
        if self.prediction_samples < 1:
            raise ValueError('prediction_samples must be positive.')
        if self.inference not in ('paper', 'marginal'):
            raise ValueError('inference must be paper or marginal.')
        return super().fit(X, Y, treatment=treatment, validation_data=validation_data)

    def predict(self, X, *, treatment):
        if not np.isin(treatment, [0, 1]).all():
            raise ValueError('TECE-VAE requires binary treatment components.')
        return super().predict(X, treatment=treatment)
