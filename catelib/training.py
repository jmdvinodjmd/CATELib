"""Small, explicit training loop shared by the S- and H-learners.

Networks return raw outcome scores. Continuous outcomes use MSE; binary
outcomes use binary cross entropy with logits. No experiment I/O happens here.
"""
from copy import deepcopy

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from torch import nn


def arrays(X, Y, treatment):
    X = np.asarray(X, dtype=np.float32)
    T = np.asarray(treatment, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    if Y.ndim == 1:
        Y = Y[:, None]
    if X.ndim != 2 or T.ndim != 2 or Y.ndim != 2:
        raise ValueError('Use X=(n,p), treatment=(n,K), Y=(n,M).')
    if not (len(X) == len(T) == len(Y)) or not len(X):
        raise ValueError('X, treatment and Y must have the same nonzero row count.')
    if not np.isfinite(X).all() or not np.isfinite(T).all() or np.isinf(Y).any():
        raise ValueError('X/treatment must be finite; only Y may contain NaNs.')
    return X, Y, T


def outcome_loss(scores, Y, outcome_types, weights=None):
    """Sum of observed-entry mean losses, skipping entirely missing outcomes."""
    losses = []
    for j, kind in enumerate(outcome_types):
        observed = ~torch.isnan(Y[:, j])
        if observed.any():
            if kind == 'binary':
                loss = nn.functional.binary_cross_entropy_with_logits(
                    scores[observed, j], Y[observed, j], reduction='none')
            else:
                loss = nn.functional.mse_loss(scores[observed, j], Y[observed, j], reduction='none')
            if weights is not None:
                loss = loss * weights[observed]
            losses.append(loss.mean())
    return sum(losses) if losses else None


def fit_estimator(estimator, X, Y, treatment, validation_data=None,
                  sample_weight=None, validation_weight=None, fixed_epochs=False):
    X, Y, T = arrays(X, Y, treatment)
    if estimator.max_epochs < 1 or estimator.batch_size < 1 or estimator.patience < 1:
        raise ValueError('max_epochs, batch_size and patience must be positive.')
    if estimator.learning_rate <= 0 or estimator.weight_decay < 0:
        raise ValueError('learning_rate must be positive and weight_decay nonnegative.')
    kinds = tuple(estimator.outcome_types or ('continuous',) * Y.shape[1])
    if len(kinds) != Y.shape[1] or any(k not in ('continuous', 'binary') for k in kinds):
        raise ValueError('Specify one continuous/binary outcome type per Y column.')
    if validation_data is None:
        if sample_weight is not None or validation_weight is not None:
            raise ValueError('Weighted fitting requires explicit validation_data.')
        train, val = train_test_split(np.arange(len(X)), test_size=0.2,
                                     random_state=estimator.random_state)
        Xv, Yv, Tv = X[val], Y[val], T[val]
        X, Y, T = X[train], Y[train], T[train]
    else:
        Xv, Yv, Tv = arrays(*validation_data)
    if (Xv.shape[1], Yv.shape[1], Tv.shape[1]) != (X.shape[1], Y.shape[1], T.shape[1]):
        raise ValueError('Training and validation column counts must agree.')
    for j, kind in enumerate(kinds):
        if np.isnan(Y[:, j]).all() or np.isnan(Yv[:, j]).all():
            raise ValueError('Each outcome needs observed training and validation labels.')
        if kind == 'binary':
            labels = np.concatenate((Y[:, j], Yv[:, j]))
            if not np.isin(labels[~np.isnan(labels)], [0, 1]).all():
                raise ValueError('Binary outcomes must have labels 0 or 1.')

    estimator.n_features_in_ = X.shape[1]
    estimator.n_treatments_ = T.shape[1]
    estimator.n_outcomes_ = Y.shape[1]
    estimator.outcome_types_ = kinds
    estimator.x_scaler_ = StandardScaler().fit(X)
    estimator.y_mean_ = np.zeros(Y.shape[1], dtype=np.float32)
    estimator.y_scale_ = np.ones(Y.shape[1], dtype=np.float32)
    for j, kind in enumerate(kinds):
        if kind == 'continuous':
            estimator.y_mean_[j] = np.nanmean(Y[:, j])
            estimator.y_scale_[j] = max(float(np.nanstd(Y[:, j])), 1e-6)
    Y = (Y - estimator.y_mean_) / estimator.y_scale_
    Yv = (Yv - estimator.y_mean_) / estimator.y_scale_
    device = torch.device(estimator.device)
    train_tensors = [torch.as_tensor(a, dtype=torch.float32, device=device)
                     for a in (estimator.x_scaler_.transform(X), Y, T)]
    val_tensors = [torch.as_tensor(a, dtype=torch.float32, device=device)
                   for a in (estimator.x_scaler_.transform(Xv), Yv, Tv)]
    rng = np.random.default_rng(estimator.random_state)
    train_weight = torch.ones(len(X), device=device) if sample_weight is None else torch.as_tensor(sample_weight, dtype=torch.float32, device=device)
    val_weight = torch.ones(len(Xv), device=device) if validation_weight is None else torch.as_tensor(validation_weight, dtype=torch.float32, device=device)
    for weights, count in ((train_weight, len(X)), (val_weight, len(Xv))):
        if weights.shape != (count,) or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('Weights must be finite nonnegative row vectors.')
    # Preserve the caller's CPU RNG state. CPU is the inexpensive default.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(estimator.random_state)
        estimator.network_ = estimator._build_network().to(device)
        if hasattr(estimator.network_, 'configure_training'):
            estimator.network_.configure_training(train_tensors[2])
        optimizer = torch.optim.Adam(estimator.network_.parameters(),
                                     lr=estimator.learning_rate,
                                     weight_decay=estimator.weight_decay)
        best_loss, stale, best_state = float('inf'), 0, None
        estimator.history_ = []
        for epoch in range(estimator.max_epochs):
            estimator.network_.train()
            order = rng.permutation(len(X))
            for start in range(0, len(X), estimator.batch_size):
                index = torch.as_tensor(order[start:start + estimator.batch_size], device=device)
                xb, yb, tb = [a[index] for a in train_tensors]
                optimizer.zero_grad()
                if hasattr(estimator.network_, 'training_loss'):
                    loss = estimator.network_.training_loss(xb, yb, tb, kinds)
                else:
                    loss = outcome_loss(estimator.network_(xb, tb), yb, kinds, train_weight[index])
                if loss is None:
                    continue
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss.')
                loss.backward()
                nn.utils.clip_grad_norm_(estimator.network_.parameters(), 10.0)
                optimizer.step()
            estimator.network_.eval()
            with torch.no_grad():
                xt, yt, tt = train_tensors
                xv, yv, tv = val_tensors
                train_loss = float(outcome_loss(estimator.network_(xt, tt), yt, kinds, train_weight))
                val_loss = float(outcome_loss(estimator.network_(xv, tv), yv, kinds, val_weight))
            if not np.isfinite(val_loss):
                raise FloatingPointError('Nonfinite validation loss.')
            estimator.history_.append(dict(epoch=epoch + 1, train_loss=train_loss,
                                           validation_loss=val_loss))
            if fixed_epochs or val_loss < best_loss:
                best_loss, stale = val_loss, 0
                best_state = deepcopy(estimator.network_.state_dict())
                estimator.best_epoch_ = epoch + 1
            else:
                stale += 1
            if not fixed_epochs and stale >= estimator.patience:
                break
        estimator.network_.load_state_dict(best_state)
    estimator.network_.eval()
    estimator.best_validation_loss_ = best_loss
    estimator.validation_strategy_ = 'fixed_epochs_training_monitor' if fixed_epochs else 'heldout_early_stopping'
    return estimator


def predict_estimator(estimator, X, treatment):
    check_is_fitted(estimator, 'network_')
    X = np.asarray(X, dtype=np.float32)
    T = np.asarray(treatment, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != estimator.n_features_in_:
        raise ValueError('X must have the fitted number of feature columns.')
    # A single intervention vector applies to every patient. Other broadcasting
    # is rejected so a transposed treatment matrix cannot silently pass.
    if T.ndim == 1:
        T = np.broadcast_to(T, (len(X), len(T))).copy()
    if T.shape != (len(X), estimator.n_treatments_):
        raise ValueError('treatment must be (K,) or (n,K).')
    if not np.isfinite(X).all() or not np.isfinite(T).all():
        raise ValueError('Prediction inputs must be finite.')
    if not len(X):
        return np.empty((0, estimator.n_outcomes_), dtype=np.float32)
    device = next(estimator.network_.parameters()).device
    estimator.network_.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(X), estimator.batch_size):
            xb = estimator.x_scaler_.transform(X[start:start + estimator.batch_size])
            tb = T[start:start + estimator.batch_size]
            scores = estimator.network_(torch.as_tensor(xb, dtype=torch.float32, device=device),
                                        torch.as_tensor(tb, dtype=torch.float32, device=device))
            for j, kind in enumerate(estimator.outcome_types_):
                if kind == 'binary':
                    scores[:, j] = scores[:, j].sigmoid()
            predictions.append(scores.cpu().numpy())
    return np.concatenate(predictions) * estimator.y_scale_ + estimator.y_mean_


def train_stage(network, X, Y, validation_data, *, epochs, batch_size,
                learning_rate, patience, random_state, device='cpu'):
    """Train a nuisance network whose loss(prediction, target) defines its stage.

    This keeps optimisation out of model definitions. Input arrays are already
    preprocessed by the estimator using training statistics only.
    """
    network = network.to(device)
    X, Y = [torch.as_tensor(a, dtype=torch.float32, device=device) for a in (X, Y)]
    Xv, Yv = [torch.as_tensor(a, dtype=torch.float32, device=device) for a in validation_data]
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    rng = np.random.default_rng(random_state)
    best, state, stale, history = float('inf'), None, 0, []
    for epoch in range(epochs):
        network.train()
        order = rng.permutation(len(X))
        for start in range(0, len(X), batch_size):
            index = torch.as_tensor(order[start:start+batch_size], device=device)
            optimizer.zero_grad()
            loss = network.loss(network(X[index]), Y[index])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite nuisance loss.')
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.no_grad():
            score = float(network.loss(network(Xv), Yv))
        if not np.isfinite(score):
            raise FloatingPointError('Nonfinite nuisance validation loss.')
        history.append(dict(epoch=epoch+1, validation_loss=score))
        if score < best:
            best, state, stale = score, deepcopy(network.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    if state is None:
        raise ValueError('No successful training epoch.')
    network.load_state_dict(state)
    network.eval()
    return network, history
