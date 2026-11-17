"""Single-cause perturbation with explicit regression-stage estimators.

Upstream attribution and MIT notices: see THIRD_PARTY_NOTICES.md.

Paper/code: https://github.com/ZhaozhiQIAN/Single-Cause-Perturbation-NeurIPS-2021
Paper title: Estimating Multi-cause Treatment Effects via Single-cause
Perturbation (Qian, Curth and van der Schaar, NeurIPS 2021).
Local reference: baselines/scp.py.

SCP is a framework whose first/second-stage learners can vary. first_stage='s'
selects SCP-S; first_stage='dr-cfr' selects two outcome heads with separate
representations, a propensity head and RBF-MMD regularisation. The latter uses
MSE rather than legacy RMSE, standard BCE, and explicit MMD weight 0.1; it is
not a numerical port. For each cause it learns conditional outcomes, flips that cause,
optionally regenerates descendants in a supplied causal order, and trains a
final regressor on factual and augmented examples. The default assumes no
causal edges between treatments conditional on X. Supply treatment_order for
a sequential complete-DAG factorisation; that order is an additional causal
assumption, not inferred from correlations. Binary interventions only.

Pseudo-outcome probabilities use squared-error regression in stage two and
are clipped on prediction. Only training patients are augmented. Validation
uses factual observations only, unlike legacy synthetic-validation selection.
The S-first-stage variant must be labelled SCP-S in comparative results.
"""
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from catelib.training import arrays
from catelib.methods.s_learner import SLearner
import torch
from torch import nn
from catelib.training import outcome_loss


class SingleCauseNetwork(nn.Module):
    """Separate confounding/outcome representations and factual treatment heads.

    Architecture reference: local scp_utils.py::DR_CRN. Uses standard BCE for
    binary outcomes and MSE rather than legacy RMSE for continuous outcomes.
    The propensity head and RBF-MMD regularise the confounding representation.
    """
    def __init__(self, inputs, outcomes, width):
        super().__init__()
        self.confounder = nn.Sequential(nn.Linear(inputs, width), nn.ReLU(), nn.Linear(width, width))
        self.outcome_features = nn.Sequential(nn.Linear(inputs, width), nn.ReLU(), nn.Linear(width, width))
        self.propensity = nn.Linear(width, 2)
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(2*width, width), nn.ReLU(), nn.Linear(width, outcomes)) for _ in range(2)])

    def components(self, X, T):
        confounder = self.confounder(X)
        features = torch.cat((confounder, self.outcome_features(X)), dim=1)
        return self.heads[0](features)*(1-T) + self.heads[1](features)*T, confounder

    def forward(self, X, T):
        return self.components(X, T)[0]

    def training_loss(self, X, Y, T, kinds):
        scores, confounder = self.components(X, T)
        loss = outcome_loss(scores, Y, kinds)
        if loss is None:
            return None
        propensity = nn.functional.cross_entropy(self.propensity(confounder), T[:, 0].long())
        a, b = confounder[T[:, 0] == 0], confounder[T[:, 0] == 1]
        mmd = confounder.sum()*0
        if min(len(a), len(b)) > 1:
            aa, bb, ab = [torch.exp(-torch.cdist(u, v).square()) for u, v in ((a,a),(b,b),(a,b))]
            mmd = (aa.sum()-aa.diagonal().sum())/(len(a)*(len(a)-1))
            mmd = mmd + (bb.sum()-bb.diagonal().sum())/(len(b)*(len(b)-1))-2*ab.mean()
        return loss + propensity + 0.1*mmd


class SingleCauseDRCFR(SLearner):
    def _build_network(self):
        return SingleCauseNetwork(self.n_features_in_, self.n_outcomes_, self.hidden_sizes[0] if self.hidden_sizes else 16)


class SCP(BaseEstimator):
    def __init__(self, estimator=None, outcome_types=None, treatment_order=None, random_state=0, first_stage='s'):
        self.estimator = estimator
        self.outcome_types = outcome_types
        self.treatment_order = treatment_order
        self.random_state = random_state
        self.first_stage = first_stage

    def fit(self, X, Y, *, treatment, validation_data=None):
        X, Y, T = arrays(X, Y, treatment)
        if not np.isin(T, [0, 1]).all():
            raise ValueError('SCP-S requires binary treatment components.')
        if validation_data is None:
            a, b = train_test_split(np.arange(len(X)), test_size=.2, random_state=self.random_state)
            validation_data = X[b], Y[b], T[b]
            X, Y, T = X[a], Y[a], T[a]
        Xv, Yv, Tv = arrays(*validation_data)
        template = self.estimator if self.estimator is not None else SLearner()
        if self.first_stage not in ('s', 'dr-cfr'):
            raise ValueError('first_stage must be s or dr-cfr.')
        self.outcome_types_ = tuple(self.outcome_types or ('continuous',)*Y.shape[1])
        k = T.shape[1]
        order = list(range(k)) if self.treatment_order is None else list(self.treatment_order)
        if sorted(order) != list(range(k)):
            raise ValueError('treatment_order must be a permutation of component indices.')
        self.cause_models_ = {}
        if self.treatment_order is not None:
            for position, cause in enumerate(order):
                parents = order[:position]
                cx, cv = np.column_stack((X, T[:, parents])), np.column_stack((Xv, Tv[:, parents]))
                model = clone(template).set_params(outcome_types=('binary',))
                model.fit(cx, T[:, cause:cause+1], treatment=np.zeros((len(X), 1)),
                          validation_data=(cv, Tv[:, cause:cause+1], np.zeros((len(Xv), 1))))
                self.cause_models_[cause] = model
        augmented_x, augmented_t, augmented_y = [X], [T], [Y]
        self.first_stage_models_ = []
        rng = np.random.default_rng(self.random_state)
        for cause in range(k):
            others = [j for j in range(k) if j != cause]
            context, context_v = np.column_stack((X, T[:, others])), np.column_stack((Xv, Tv[:, others]))
            if self.first_stage == 'dr-cfr':
                options = {key: value for key, value in template.get_params().items() if key in SLearner().get_params()}
                model = SingleCauseDRCFR(**options).set_params(outcome_types=self.outcome_types_)
            else:
                model = clone(template).set_params(outcome_types=self.outcome_types_)
            model.fit(context, Y, treatment=T[:, cause:cause+1],
                      validation_data=(context_v, Yv, Tv[:, cause:cause+1]))
            self.first_stage_models_.append(model)
            changed = T.copy()
            changed[:, cause] = 1-changed[:, cause]
            if self.treatment_order is not None:
                for position in range(order.index(cause)+1, k):
                    child, parents = order[position], order[:position]
                    probability = self.cause_models_[child].predict(
                        np.column_stack((X, changed[:, parents])), treatment=[0])[:, 0]
                    changed[:, child] = rng.binomial(1, probability)
            pseudo = model.predict(np.column_stack((X, changed[:, others])),
                                   treatment=changed[:, cause:cause+1])
            augmented_x.append(X)
            augmented_t.append(changed)
            augmented_y.append(pseudo)
        self.model_ = clone(template).set_params(outcome_types=('continuous',)*Y.shape[1])
        self.model_.fit(np.concatenate(augmented_x), np.concatenate(augmented_y),
                        treatment=np.concatenate(augmented_t), validation_data=(Xv, Yv, Tv))
        self.history_ = dict(first_stage=[m.history_ for m in self.first_stage_models_],
                             outcome=self.model_.history_)
        return self

    def predict(self, X, *, treatment):
        check_is_fitted(self, 'model_')
        if not np.isin(treatment, [0, 1]).all():
            raise ValueError('SCP-S requires binary treatment components.')
        prediction = self.model_.predict(X, treatment=treatment)
        for j, kind in enumerate(self.outcome_types_):
            if kind == 'binary':
                prediction[:, j] = prediction[:, j].clip(0, 1)
        return prediction

    def effect(self, X, *, treatment, reference):
        return self.predict(X, treatment=treatment)-self.predict(X, treatment=reference)
