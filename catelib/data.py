"""Patient data and splitting, independent of estimators and simulations."""
from dataclasses import dataclass
import numpy as np
from catelib.datasets.composite import make_splits


@dataclass
class Dataset:
    X: np.ndarray
    Y: np.ndarray
    treatment: np.ndarray
    treatment_types: tuple
    outcome_types: tuple
    truth: object = None

    def subset(self, indices):
        """Training API tuple: X, Y, T (not the simulator's X, T, Y)."""
        return self.X[indices], self.Y[indices], self.treatment[indices]


def split_data(data, *, seed=2, fractions=(0.6, 0.2, 0.2), indices=None, **restrictions):
    """Split observational or simulated patients identically; honour explicit indices.

    Support restrictions retain the original study's 60/20/20 procedure. Arbitrary
    split fractions are available without restrictions, or supply explicit indices.
    """
    if indices is not None:
        if restrictions:
            raise ValueError('Explicit splits cannot be combined with support restrictions.')
        splits = {name: np.asarray(indices[name], dtype=int) for name in ('train','validation','test')}
    elif tuple(fractions) == (0.6, 0.2, 0.2):
        splits = make_splits(data.treatment, data.treatment_types, seed=seed, **restrictions)
    else:
        if restrictions:
            raise ValueError('Custom fractions with support restrictions require explicit indices.')
        f = np.asarray(fractions)
        if f.shape != (3,) or (f <= 0).any() or not np.isclose(f.sum(), 1):
            raise ValueError('Three positive fractions must sum to one.')
        order = np.random.default_rng(seed).permutation(len(data.X))
        parts = np.split(order, (np.cumsum(f)[:2]*len(order)).astype(int))
        splits = dict(zip(('train','validation','test'),parts))
    rows = np.concatenate([splits[k] for k in ('train','validation','test')])
    if any(len(splits[k]) < 2 for k in ('train','validation','test')):
        raise ValueError('Each split requires at least two patients.')
    if len(np.unique(rows)) != len(rows) or (rows < 0).any() or (rows >= len(data.X)).any():
        raise ValueError('Patient splits must be disjoint, unique and in bounds.')
    return splits
