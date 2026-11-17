"""Tiny tests of honest scoring, candidate resume and sparse-arm fitting."""
import numpy as np
import pytest
from sklearn.base import BaseEstimator
from catelib.tuning import _search as tune, validation_score
from catelib.methods.s_learner import SLearner
from catelib.methods.t_learner import TLearner


class ConstantEstimator(BaseEstimator):
    calls = 0

    def __init__(self, offset=0., outcome_types=('continuous',)):
        self.offset = offset
        self.outcome_types = outcome_types

    def fit(self, X, Y, *, treatment, validation_data=None):
        type(self).calls += 1
        return self

    def predict(self, X, *, treatment):
        return np.full((len(X), 1), self.offset)


def test_candidate_resume_and_validation_only_selection(tmp_path):
    X = np.arange(8.).reshape(-1, 1)
    Y = np.tile([[0.], [1.]], (4, 1)); T = np.zeros((8,1))
    validation = (X[:4], np.zeros((4,1)), T[:4])
    grid = [{'offset':[1.]}, {'offset':[0.]}]
    ConstantEstimator.calls = 0
    tune(ConstantEstimator(), X, Y, T, validation, grid, cache_dir=tmp_path, max_candidates=1)
    model, search = tune(ConstantEstimator(), X, Y, T, validation, grid, cache_dir=tmp_path)
    assert ConstantEstimator.calls == 2 and model.offset == 0
    assert search[0]['reused'] and not search[1]['reused']
    model, search = tune(ConstantEstimator(), X, Y, T, validation, grid, cache_dir=tmp_path)
    assert ConstantEstimator.calls == 2 and all(row['reused'] for row in search)
    # Changing validation labels invalidates the search and changes the winner.
    changed = (X[:4], np.ones((4,1)), T[:4])
    model, _ = tune(ConstantEstimator(), X, Y, T, changed, grid, cache_dir=tmp_path)
    assert ConstantEstimator.calls == 4 and model.offset == 1


def test_nonfinite_predictions_cannot_win_by_omission():
    Y = np.array([[0.], [1.]])
    with pytest.raises(ValueError, match='Nonfinite predictions'):
        validation_score(np.array([[0.], [np.nan]]), Y, Y, ('continuous',))
    score, _ = validation_score(np.array([[0.], [np.nan]]), Y,
                                np.array([[0.], [np.nan]]), ('continuous',))
    assert score == 0


def test_tlearner_fits_observed_arm_without_validation_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12,2)); T = np.r_[np.zeros(8),np.ones(4)][:,None]
    Y = (X[:, :1]+T)
    model = TLearner(SLearner(hidden_sizes=(2,), max_epochs=2, patience=1,
                             outcome_types=('continuous',)), outcome_types=('continuous',))
    model.fit(X,Y,treatment=T,validation_data=(X[:3],Y[:3],T[:3]))
    assert np.isfinite(model.predict(X[:3],treatment=[1])).all()
    fallback=model.models_[(1.,)]
    assert len(fallback.history_)==2 and fallback.best_epoch_==2
    assert fallback.validation_strategy_=='fixed_epochs_training_monitor'
    assert model.history_[1]['validation_count']==0
    with pytest.raises(ValueError, match='Unsupported treatment arm'):
        model.predict(X[:3],treatment=[2])
