"""Variational sample reweighting (VSR), with explicit three-stage fitting.

Upstream attribution and MIT notices: see THIRD_PARTY_NOTICES.md.

Source implementation: https://github.com/ZhaozhiQIAN/Single-Cause-Perturbation-NeurIPS-2021
Local references: baselines/vsr.py and TreatmentVAE/DensityRatioNetwork in
baselines/scp_utils.py. The original paper is cited in the H-learner paper:
https://arxiv.org/abs/2502.08282

1. A treatment VAE encodes binary bundles into a latent Gaussian.
2. A classifier distinguishes (X, posterior latent) from (X, prior latent).
3. Their density ratio weights a supervised outcome model.

This is a corrected local reimplementation: standard Bernoulli reconstruction
fixes the legacy reversed probability ordering; nuisance fits see training data
only; configurable Adam replaces hard-coded SGD. Weight clipping is explicit.
It does not establish identification of arbitrary hidden confounders. Binary
interventions only. Multiple outcomes use the chosen outcome estimator.
"""
import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from catelib.methods.s_learner import SLearner
from catelib.training import arrays, train_stage, fit_estimator


class TreatmentVAE(nn.Module):
    def __init__(self, treatments, latent):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(treatments, 2*latent), nn.ReLU())
        self.mu = nn.Linear(2*latent, latent)
        self.logvar = nn.Linear(2*latent, latent)
        self.decoder = nn.Linear(latent, treatments)

    def encode(self, T):
        hidden = self.encoder(T)
        return self.mu(hidden), self.logvar(hidden).clamp(-10, 10)

    def forward(self, T):
        mu, logvar = self.encode(T)
        z = mu + (0.5*logvar).exp()*torch.randn_like(mu) if self.training else mu
        return self.decoder(z), mu, logvar

    def loss(self, prediction, T):
        scores, mu, logvar = prediction
        reconstruction = nn.functional.binary_cross_entropy_with_logits(scores, T)
        kl = -0.5*(1+logvar-mu.square()-logvar.exp()).sum(1).mean()
        return reconstruction + kl


class DensityRatioNetwork(nn.Module):
    def __init__(self, inputs, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, width), nn.ReLU(), nn.Linear(width, 2))

    def forward(self, X):
        return self.layers(X)

    def loss(self, scores, labels):
        return nn.functional.cross_entropy(scores, labels.long().flatten())


class VSR(BaseEstimator):
    def __init__(self, estimator=None, outcome_types=None, latent_dim=2, width=16,
                 max_epochs=50, batch_size=64, learning_rate=0.001, patience=10,
                 max_weight=20.0, random_state=0, device='cpu'):
        self.estimator = estimator
        self.outcome_types = outcome_types
        self.latent_dim = latent_dim
        self.width = width
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.max_weight = max_weight
        self.random_state = random_state
        self.device = device

    def fit(self, X, Y, *, treatment, validation_data=None):
        X, Y, T = arrays(X, Y, treatment)
        if not np.isin(T, [0, 1]).all():
            raise ValueError('VSR requires binary treatments.')
        if self.max_weight <= 0:
            raise ValueError('max_weight must be positive.')
        if validation_data is None:
            a, b = train_test_split(np.arange(len(X)), test_size=0.2, random_state=self.random_state)
            validation_data = X[b], Y[b], T[b]
            X, Y, T = X[a], Y[a], T[a]
        Xv, Yv, Tv = arrays(*validation_data)
        self.scaler_ = StandardScaler().fit(X)
        Xs, Xvs = self.scaler_.transform(X), self.scaler_.transform(Xv)
        options = dict(epochs=self.max_epochs, batch_size=self.batch_size,
                       learning_rate=self.learning_rate, patience=self.patience,
                       random_state=self.random_state, device=self.device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_state)
            self.vae_, vae_history = train_stage(TreatmentVAE(T.shape[1], self.latent_dim),
                                                 T, T, (Tv, Tv), **options)
            def ratio_data(covariates, treatments):
                with torch.no_grad():
                    t = torch.as_tensor(treatments, dtype=torch.float32, device=self.device)
                    mu, logvar = self.vae_.encode(t)
                    z = mu + (logvar*0.5).exp()*torch.randn_like(mu)
                    prior = torch.randn_like(z)
                positive = np.column_stack((covariates, z.cpu().numpy()))
                negative = np.column_stack((covariates, prior.cpu().numpy()))
                return np.concatenate((positive, negative)), np.r_[np.ones(len(t)), np.zeros(len(t))]
            rx, ry = ratio_data(Xs, T)
            rvx, rvy = ratio_data(Xvs, Tv)
            self.ratio_, ratio_history = train_stage(DensityRatioNetwork(rx.shape[1], self.width),
                                                       rx, ry, (rvx, rvy), **options)
            def weights(data, n):
                with torch.no_grad():
                    scores = self.ratio_(torch.as_tensor(data[:n], dtype=torch.float32, device=self.device))
                    log_ratio = (scores[:, 0]-scores[:, 1]).cpu().numpy()
                return log_ratio
            logw = weights(rx, len(X))
            centre = logw.mean()
            w = np.exp(np.clip(logw-centre, -30, np.log(self.max_weight)))
            wv = np.exp(np.clip(weights(rvx, len(Xv))-centre, -30, np.log(self.max_weight)))
        template = self.estimator if self.estimator is not None else SLearner()
        self.model_ = clone(template).set_params(outcome_types=self.outcome_types)
        fit_estimator(self.model_, X, Y, T, (Xv, Yv, Tv), w, wv)
        self.history_ = dict(vae=vae_history, ratio=ratio_history, outcome=self.model_.history_)
        self.training_weights_ = w
        return self

    def predict(self, X, *, treatment):
        check_is_fitted(self, 'model_')
        if not np.isin(treatment, [0, 1]).all():
            raise ValueError('VSR requires binary treatments.')
        return self.model_.predict(X, treatment=treatment)

    def effect(self, X, *, treatment, reference):
        return self.predict(X, treatment=treatment) - self.predict(X, treatment=reference)
