"""Small checks of experiment comparisons and resumable artifacts."""
import numpy as np
from catelib.datasets.composite import CompositeDGP, make_splits
from catelib.results import finish_model, completed_model, save_model, load_model


def test_fixed_test_pool_and_exact_target_support():
    dgp = CompositeDGP(treatment_types=('binary','binary'))
    X, T, _, _ = dgp.sample(200)
    small = make_splits(T, dgp.treatment_types, train_size=20)
    large = make_splits(T, dgp.treatment_types, train_size=40)
    np.testing.assert_array_equal(small['train'], large['train'][:20])
    np.testing.assert_array_equal(small['test'], large['test'])
    zero = make_splits(T, dgp.treatment_types, holdout_pattern=[1,1])
    few = make_splits(T, dgp.treatment_types, holdout_pattern=[1,1], target_count=5)
    assert np.all(T[few['train']]==1, axis=1).sum() == 5
    assert not np.all(T[few['validation']]==1, axis=1).any()
    np.testing.assert_array_equal(zero['test'], few['test'])
    assert set(few['train'])-set(zero['train']) == set(few['train'][-5:])


def test_extra_outcomes_do_not_change_first_mechanism():
    one = CompositeDGP(outcome_types=('continuous',), outcome_sharing=.8)
    many = CompositeDGP(outcome_types=('continuous',)*4, outcome_sharing=.8)
    X, T, Y, mu = one.sample(40)
    X2, T2, Y2, mu2 = many.sample(40)
    for a,b in [(X,X2),(T,T2),(Y[:,0],Y2[:,0]),(mu[:,0],mu2[:,0])]:
        np.testing.assert_allclose(a,b)


def test_model_completion_detects_corruption_and_checkpoint_survives(tmp_path):
    save_model(tmp_path/'h_model.joblib', {'fitted': True})
    metric = tmp_path/'h_metrics.json'
    metric.write_text('[]')
    finish_model(tmp_path, 'h', {'model':'h','status':'ok'})
    assert completed_model(tmp_path, 'h')['status'] == 'ok'
    metric.write_text('[1]')
    assert completed_model(tmp_path, 'h') is None
    assert load_model(tmp_path/'h_model.joblib') == {'fitted': True}
    (tmp_path/'h_model.joblib').write_bytes(b'corrupted')
    assert load_model(tmp_path/'h_model.joblib') is None
