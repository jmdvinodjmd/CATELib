"""Small correctness checks, not a benchmark or hyperparameter search."""
import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.base import clone

from catelib.datasets.composite import CompositeDGP, make_splits
from catelib.metrics.effects import evaluate
from catelib.methods.h_learner import HLearner, HyperOutcomeNetwork
from catelib.methods.s_learner import SLearner, OutcomeNetwork
from catelib.training import outcome_loss

torch.set_num_threads(1)


def test_dgp_truth_and_mixed_support():
    dgp = CompositeDGP(noise_sd=0, outcome_types=('continuous',), mechanism_seed=4)
    X, T, Y, mu = dgp.sample(100, 9)
    np.testing.assert_allclose(Y, dgp.mean(X, T))
    assert len(np.unique(T[:, 1])) == 100  # no rounding to a dose grid
    assert not np.allclose(dgp.mean(X, [0, 0.2, 0]), dgp.mean(X, [0, 0.8, 0]))
    # Query truth depends on X and requested intervention, not factual assignment.
    np.testing.assert_allclose(dgp.mean(X[:1], [1, 0.37, 0]),
                               dgp.mean(X, [1, 0.37, 0])[:1])
    splits = make_splits(T, dgp.treatment_types, holdout_pattern=[1, 1])
    used = np.concatenate(list(splits.values()))
    assert len(np.unique(used)) == len(T) == len(used)
    for key in ('train', 'validation'):
        assert not np.any(np.all(T[splits[key]][:, [0, 2]] == [1, 1], axis=1))


@pytest.mark.parametrize('estimator', [SLearner, HLearner])
def test_fit_predict_contract(estimator):
    dgp = CompositeDGP()
    X, T, Y, _ = dgp.sample(60)
    Y[0, 1] = np.nan
    kwargs = dict(hidden_sizes=(4,), max_epochs=2, batch_size=16,
                  outcome_types=dgp.outcome_types, random_state=12)
    if estimator is HLearner:
        kwargs['hypernet_sizes'] = (4,)
    model = estimator(**kwargs)
    assert clone(model).get_params() == model.get_params()
    assert model.fit(X[:40], Y[:40], treatment=T[:40],
                     validation_data=(X[40:], Y[40:], T[40:])) is model
    prediction = model.predict(X[:3], treatment=[1, 0.371, 0])
    assert prediction.shape == (3, 2) and np.isfinite(prediction).all()
    assert np.all((prediction[:, 1] >= 0) & (prediction[:, 1] <= 1))
    np.testing.assert_array_equal(prediction, model.predict(X[:3], treatment=[1, 0.371, 0]))
    np.testing.assert_array_equal(model.effect(X[:3], treatment=[1, 0.4, 0],
                                              reference=[1, 0.4, 0]), np.zeros((3, 2)))
    assert model.predict(X[:1], treatment=T[:1]).shape == (1, 2)
    assert len(model.history_) == 2
    np.testing.assert_allclose(model.x_scaler_.mean_, X[:40].mean(axis=0), atol=1e-7)
    with pytest.raises(ValueError):
        model.predict(X[:3], treatment=[1, 0])


def test_generated_target_matches_legacy_after_bias_fix():
    # Original shapeWeights selects one bias scalar (wt[-width]), broadcasting
    # it over the layer. Correct that slice in this in-memory reference only;
    # keep the on-disk legacy implementation unchanged for provenance.
    source = Path('src/models/hn_utils.py').read_text()
    source = source.replace('wt[-layers[j][i]]', 'wt[-layers[j][i]:]')
    namespace = {}
    exec(compile(source, 'legacy_target_with_bias_fix', 'exec'), namespace)
    MLPFunctional = namespace['MLPFunctional']
    torch.manual_seed(3)
    net = HyperOutcomeNetwork(3, 2, 1, (4, 2), (4,), 2, 2)
    X, T = torch.randn(5, 3), torch.rand(5, 2)
    parameters = net.hypernetwork(net.treatment_embedding(T))
    expected = torch.stack([MLPFunctional(X[i], parameters[i], in_size=3,
                           layers=[[4, 2], [1]],
                           activations=[[torch.relu, torch.relu], [None]],
                           dropout_rate=0)[0] for i in range(5)])
    torch.testing.assert_close(net(X, T), expected)
    net(X, T).sum().backward()
    assert net.treatment_embedding.weight.grad.abs().sum() > 0


def test_s_network_matches_original_fixed_weights():
    # Extract only the original class; importing its utils module would pull in
    # legacy plotting/logging packages that the new empirical core does not need.
    tree = ast.parse(Path('src/utils.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MLP_Model')
    namespace = {'torch': torch, 'nn': torch.nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'legacy_mlp', 'exec'), namespace)
    legacy = namespace['MLP_Model']([[4], [1], [1]], [[torch.nn.ReLU], [None], [None]], 3, 2)
    new = OutcomeNetwork(3, 2, 2, (4,))
    with torch.no_grad():
        new.layers[0].load_state_dict(legacy.phi[0].state_dict())
        new.layers[-1].weight.copy_(torch.cat([head[0].weight for head in legacy.tasks]))
        new.layers[-1].bias.copy_(torch.cat([head[0].bias for head in legacy.tasks]))
    X, T = torch.randn(5, 3), torch.rand(5, 2)
    torch.testing.assert_close(new(X, T), legacy(X, T))


def test_masked_loss_and_oracle_evaluation():
    scores = torch.zeros((3, 2), requires_grad=True)
    Y = torch.tensor([[1., float('nan')]] * 3)
    loss = outcome_loss(scores, Y, ('continuous', 'binary'))
    assert float(loss.detach()) == 1.0
    loss.backward()
    assert torch.isfinite(scores.grad).all()
    dgp = CompositeDGP()
    X, _, _, _ = dgp.sample(5)

    class Oracle:
        def predict(self, X, *, treatment):
            return dgp.mean(X, treatment)

    rows = evaluate(Oracle(), dgp, X, dgp.intervention_grid(), [0, 0.5, 0], [1, 1])
    assert all(row['mean_mse'] == row['cate_mse'] == 0 for row in rows)
    assert any(row['withheld_pattern'] for row in rows)


def test_policy_and_interaction_oracles():
    from catelib.metrics.effects import evaluate_policy, evaluate_interactions
    dgp = CompositeDGP()
    X, _, _, _ = dgp.sample(8)

    class Oracle:
        def predict(self, X, *, treatment):
            return dgp.mean(X, treatment)

    rows = evaluate_interactions(Oracle(), dgp, X, [0, .5, 0], [1, .75, 1])
    assert all(row['interaction_mse'] == 0 for row in rows)
    policy = evaluate_policy(Oracle(), dgp, X, dgp.intervention_grid(), [1, -1])
    assert policy['policy_regret'] == 0


def test_all_estimators_clone_and_preserve_parameters():
    from catelib.methods.independent_outcomes import IndependentOutcomes
    from catelib.methods.compositional_regression import CompositionalRegression
    from catelib.methods.t_learner import TLearner
    from catelib.methods.deconfounder import Deconfounder
    from catelib.methods.vsr import VSR
    from catelib.methods.scp import SCP
    from catelib.methods.cisi_net import CISINet
    from catelib.methods.cihsi_net import CIHSINet
    from catelib.methods.ncore import NCoRE
    from catelib.methods.tece_vae import TECEVAE
    from catelib.methods.dr_learner import DRLearner
    from catelib.methods.tarnet import TARNet
    for cls in (IndependentOutcomes, CompositionalRegression, TLearner, Deconfounder,
                VSR, SCP, CISINet, CIHSINet, NCoRE, TECEVAE, DRLearner, TARNet):
        estimator = cls()
        assert clone(estimator).get_params() == estimator.get_params()


def test_unsupported_arm_rows_remain_in_evaluation():
    dgp = CompositeDGP(treatment_types=('binary',), outcome_types=('continuous',))
    X, _, _, _ = dgp.sample(5)

    class PartialOracle:
        def predict(self, X, *, treatment):
            if treatment[0] == 1:
                raise ValueError('Unsupported treatment arm')
            return dgp.mean(X, treatment)

    rows = evaluate(PartialOracle(), dgp, X, [[0], [1]], [0], allow_unsupported=True)
    assert rows[0]['cate_mse'] == 0
    assert rows[1]['status'] == 'unsupported' and rows[1]['cate_mse'] is None
