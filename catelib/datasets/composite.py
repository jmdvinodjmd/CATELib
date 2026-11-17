"""Small empirical DGP with mixed interventions and explicit causal truth.

This is a NEW, versioned generator, not a reproduction of synthetic_dgp.py.
It retains Gaussian covariates, confounded assignment, nonlinear outcomes and
editable feature blocks, but evaluates every intervention at the requested T.
There is no neural network or estimator inside this generator.
"""
from itertools import combinations, product

import numpy as np
from scipy.special import expit


class CompositeDGP:
    """Binary/continuous simultaneous interventions with main and pair effects.

    Continuous doses lie in (0,1) observationally and may be queried in [0,1].
    outcome_types can mix continuous and binary outcomes. feature_blocks gives
    the common covariate count followed by one count per treatment component.
    Mechanism and sampling seeds are separate so changing n keeps the DGP fixed.
    """
    version = 'composite_v2'

    def __init__(self, treatment_types=('binary', 'continuous', 'binary'),
                 outcome_types=('continuous', 'binary'), feature_blocks=None,
                 confounding=0.5, interaction_strength=0.5, noise_sd=0.1,
                 mechanism_seed=0, outcome_sharing=0.0, response_family='nonlinear'):
        self.treatment_types = tuple(treatment_types)
        self.outcome_types = tuple(outcome_types)
        if not self.treatment_types or not self.outcome_types:
            raise ValueError('At least one treatment and outcome are required.')
        if any(t not in ('binary', 'continuous') for t in self.treatment_types + self.outcome_types):
            raise ValueError('Types must be binary or continuous.')
        k, m = len(self.treatment_types), len(self.outcome_types)
        self.feature_blocks = tuple(feature_blocks if feature_blocks is not None else (2,) + (1,) * k)
        if len(self.feature_blocks) != k + 1 or min(self.feature_blocks) < 0 or sum(self.feature_blocks) < 1:
            raise ValueError('feature_blocks needs a common block and one block per treatment.')
        self.n_features = sum(self.feature_blocks)
        self.confounding = confounding
        self.interaction_strength = interaction_strength
        self.noise_sd = noise_sd
        self.mechanism_seed = mechanism_seed
        self.outcome_sharing = outcome_sharing
        self.response_family = response_family
        if not 0 <= outcome_sharing <= 1 or response_family not in ('linear', 'nonlinear', 'higher_order', 'unstructured'):
            raise ValueError('Invalid sharing fraction or response family.')
        if noise_sd < 0:
            raise ValueError('noise_sd must be nonnegative.')
        rng = np.random.default_rng(mechanism_seed)
        p = self.n_features
        self.assignment_weights = np.zeros((p, k))
        boundaries = np.cumsum(self.feature_blocks)
        for j in range(k):
            indices = np.r_[np.arange(self.feature_blocks[0]),
                            np.arange(boundaries[j], boundaries[j + 1])]
            self.assignment_weights[indices, j] = rng.normal(size=len(indices)) / np.sqrt(max(1, len(indices)))
        self.baseline_weights = rng.normal(0, 0.4 / np.sqrt(p), (p, m))
        self.main_intercepts = rng.normal(0.4, 0.3, (k, m))
        self.main_weights = rng.normal(0, 0.4 / np.sqrt(p), (p, k, m))
        self.pairs = tuple(combinations(range(k), 2))
        self.pair_intercepts = rng.normal(0, 0.3, (len(self.pairs), m))
        self.pair_weights = rng.normal(0, 0.2 / np.sqrt(p), (p, len(self.pairs), m))
        # Separate streams preserve outcome 0's mechanism as extra outcomes are
        # added. Shared coefficients specify a DGP property, not an estimator.
        shapes = {'baseline_weights': (p,), 'main_intercepts': (k,),
                  'main_weights': (p, k), 'pair_intercepts': (len(self.pairs),),
                  'pair_weights': (p, len(self.pairs))}
        scales = {'baseline_weights': .4/np.sqrt(p), 'main_intercepts': .3,
                  'main_weights': .4/np.sqrt(p), 'pair_intercepts': .3,
                  'pair_weights': .2/np.sqrt(p)}
        shared_rng = np.random.default_rng(mechanism_seed+10000)
        shared = {name: shared_rng.normal(size=shape) for name, shape in shapes.items()}
        for j in range(m):
            outcome_rng = np.random.default_rng(mechanism_seed+20000+j)
            for name, shape in shapes.items():
                coefficient = (np.sqrt(outcome_sharing)*shared[name] +
                               np.sqrt(1-outcome_sharing)*outcome_rng.normal(size=shape))*scales[name]
                getattr(self, name)[..., j] = coefficient + (0.4 if name == 'main_intercepts' else 0)
        if response_family == 'unstructured':
            if any(kind != 'binary' for kind in self.treatment_types) or k > 10:
                raise ValueError('Unstructured stress test supports at most 10 binary components.')
            self.arm_weights = np.stack([np.random.default_rng(mechanism_seed+30000+j).normal(
                scale=.4/np.sqrt(p+1), size=(2**k, p+1)) for j in range(m)], axis=-1)

    def mean(self, X, treatment):
        """Noiseless conditional potential-outcome means, shape (n,M)."""
        X = np.asarray(X, dtype=float)
        T = np.asarray(treatment, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError('Incorrect covariate shape.')
        if T.ndim == 1:
            T = np.broadcast_to(T, (len(X), len(T)))
        if T.shape != (len(X), len(self.treatment_types)):
            raise ValueError('Use treatment=(K,) or (n,K).')
        if not np.isfinite(X).all() or not np.isfinite(T).all():
            raise ValueError('Inputs must be finite.')
        for j, kind in enumerate(self.treatment_types):
            if kind == 'binary' and not np.isin(T[:, j], [0, 1]).all():
                raise ValueError('Binary components must be 0 or 1.')
            if kind == 'continuous' and ((T[:, j] < 0) | (T[:, j] > 1)).any():
                raise ValueError('This DGP defines continuous doses in [0,1].')
        nonlinear_X = X if self.response_family == 'linear' else np.tanh(X)
        scores = 0.2 + X @ self.baseline_weights
        main_effects = self.main_intercepts + np.einsum('np,pkm->nkm', nonlinear_X, self.main_weights)
        scores += np.einsum('nk,nkm->nm', T, main_effects)
        for pair, (a, b) in enumerate(self.pairs):
            effect = self.pair_intercepts[pair] + nonlinear_X @ self.pair_weights[:, pair, :]
            scores += self.interaction_strength * (T[:, a] * T[:, b])[:, None] * effect
        if self.response_family == 'higher_order' and T.shape[1] >= 3:
            scores += self.interaction_strength*np.prod(T[:, :3], axis=1)[:, None]*(0.5+np.tanh(X[:, :1]))
        if self.response_family == 'unstructured':
            indices = (T @ (2**np.arange(T.shape[1]))).astype(int)
            scores = np.einsum('np,npm->nm', np.column_stack((np.ones(len(X)), np.tanh(X))), self.arm_weights[indices])
        for j, kind in enumerate(self.outcome_types):
            if kind == 'binary':
                scores[:, j] = expit(scores[:, j])
        return scores

    def sample(self, n=240, sample_seed=1, covariates=None):
        """Return X, factual T, factual Y, and factual conditional means."""
        if n < 1:
            raise ValueError('n must be positive.')
        rng = np.random.default_rng(sample_seed)
        X = rng.normal(size=(n, self.n_features)) if covariates is None else np.asarray(covariates, float)
        if X.shape != (n, self.n_features) or not np.isfinite(X).all():
            raise ValueError('Covariates must be finite with shape (n,sum(feature_blocks)).')
        assignment = self.confounding * X @ self.assignment_weights
        T = np.empty_like(assignment)
        for j, kind in enumerate(self.treatment_types):
            if kind == 'binary':
                T[:, j] = rng.binomial(1, expit(assignment[:, j]))
            else:
                T[:, j] = expit(assignment[:, j] + rng.normal(size=n))
        mu = self.mean(X, T)
        Y = mu.copy()
        for j, kind in enumerate(self.outcome_types):
            if kind == 'binary':
                Y[:, j] = rng.binomial(1, mu[:, j])
            else:
                Y[:, j] += rng.normal(0, self.noise_sd, n)
        return X, T, Y, mu

    def intervention_grid(self, doses=(0.25, 0.75)):
        """A small evaluation grid only; continuous training doses stay continuous."""
        choices = [(0.0, 1.0) if t == 'binary' else doses for t in self.treatment_types]
        return np.asarray(list(product(*choices)), dtype=float)


def make_splits(T, treatment_types, seed=0, holdout_pattern=None,
                target_count=None, train_size=None, dose_interval=None):
    """Common 60/20/20 patient split, optionally remove a binary stratum.

    In mixed settings a stratum spans ALL continuous doses. Removed patients
    are excluded from fitting/validation, not silently moved into the test set.
    """
    n = len(T)
    if n < 20:
        raise ValueError('Use at least 20 patients for this pilot split.')
    order = np.random.default_rng(seed).permutation(n)
    train, validation, test = np.split(order, [int(0.6 * n), int(0.8 * n)])
    excluded = np.empty(0, dtype=int)
    if holdout_pattern is not None:
        binary = [j for j, kind in enumerate(treatment_types) if kind == 'binary']
        pattern = np.asarray(holdout_pattern)
        if not binary or pattern.shape != (len(binary),) or not np.isin(pattern, [0, 1]).all():
            raise ValueError('holdout_pattern needs one 0/1 entry per binary treatment.')
        withheld = np.all(T[:, binary] == pattern, axis=1)
        count = 0 if target_count is None else target_count
        if count < 0 or count > np.sum(withheld[train]):
            raise ValueError('Requested target count is negative or exceeds available training patients.')
        target_rows = train[withheld[train]]
        excluded = np.concatenate((target_rows[count:], validation[withheld[validation]]))
        train = np.concatenate((train[~withheld[train]], target_rows[:count]))
        # Keep all target labels out of validation, even at positive few-shot
        # counts, so changing tuning settings cannot add hidden target labels.
        validation = validation[~withheld[validation]]
    elif target_count is not None:
        raise ValueError('target_count requires a binary holdout/target pattern.')
    if dose_interval is not None:
        lower, upper = dose_interval
        if not 0 <= lower < upper <= 1:
            raise ValueError('dose_interval must satisfy 0 <= lower < upper <= 1.')
        columns = [j for j, kind in enumerate(treatment_types) if kind == 'continuous']
        if not columns:
            raise ValueError('A dose interval requires continuous treatments.')
        eligible = np.all((T[:, columns] >= lower) & (T[:, columns] <= upper), axis=1)
        excluded = np.r_[excluded, train[~eligible[train]], validation[~eligible[validation]]]
        train, validation = train[eligible[train]], validation[eligible[validation]]
    if train_size is not None:
        if train_size < 2 or train_size > len(train):
            raise ValueError('train_size must be between 2 and the available training count.')
        excluded = np.r_[excluded, train[train_size:]]
        train = train[:train_size]
    if min(len(train), len(validation), len(test)) < 2:
        raise ValueError('Too few patients remain in a split.')
    return dict(train=train, validation=validation, test=test, excluded=excluded)
