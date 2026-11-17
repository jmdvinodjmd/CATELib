"""Evaluate conditional means and effects on a declared intervention grid."""
import numpy as np
from itertools import combinations, product


def evaluate(model, dgp, X, interventions, reference, holdout_pattern=None, allow_unsupported=False):
    """Return per-intervention/outcome errors; test truth is used only here.

    Effects are relative to the same reference vector for every patient. These
    are squared CATE errors, not errors against sampled binary/noisy outcomes.
    """
    try:
        reference_prediction = model.predict(X, treatment=reference)
    except ValueError:
        if not allow_unsupported:
            raise
        reference_prediction = None
    reference_truth = dgp.mean(X, reference)
    rows = []
    binary = [j for j, kind in enumerate(dgp.treatment_types) if kind == 'binary']
    for intervention in np.asarray(interventions):
        error_message = ''
        try:
            prediction = model.predict(X, treatment=intervention)
        except ValueError as error:
            if not allow_unsupported:
                raise
            prediction = None
            error_message = str(error)
        truth = dgp.mean(X, intervention)
        error = prediction - truth if prediction is not None else None
        effect_error = ((prediction-reference_prediction)-(truth-reference_truth)
                        if prediction is not None and reference_prediction is not None else None)
        withheld = holdout_pattern is not None and np.array_equal(intervention[binary], holdout_pattern)
        for j in range(truth.shape[1]):
            rows.append(dict(intervention=intervention.tolist(), outcome=j,
                             withheld_pattern=bool(withheld),
                             is_reference=bool(np.array_equal(intervention, reference)),
                             status='ok' if effect_error is not None else 'unsupported',
                             reason=error_message or ('Unsupported reference arm' if reference_prediction is None else ''),
                             mean_mse=float(np.mean(error[:, j] ** 2)) if error is not None else None,
                             cate_mse=float(np.mean(effect_error[:, j] ** 2)) if effect_error is not None else None))
    return rows


def evaluate_interactions(model, dgp, X, reference, active_levels):
    """Second-order finite-difference effects; other components stay at reference.

    For binary treatments this is the pair interaction contrast. For continuous
    doses this is a finite dose contrast, NOT a cross-partial derivative.
    """
    rows = []
    for a, b in combinations(range(len(reference)), 2):
        estimated, truth = 0, 0
        for first, second in product((0, 1), repeat=2):
            intervention = np.array(reference, dtype=float).copy()
            if first:
                intervention[a] = active_levels[a]
            if second:
                intervention[b] = active_levels[b]
            sign = (-1)**(2-first-second)
            estimated = estimated + sign*model.predict(X, treatment=intervention)
            truth = truth + sign*dgp.mean(X, intervention)
        for j in range(truth.shape[1]):
            rows.append(dict(component_a=a, component_b=b, outcome=j,
                             interaction_mse=float(np.mean((estimated[:, j]-truth[:, j])**2))))
    return rows


def evaluate_policy(model, dgp, X, interventions, utility_weights, treatment_costs=None):
    """Oracle regret over the supplied finite feasible intervention set only."""
    weights = np.asarray(utility_weights, float)
    if weights.shape != (len(dgp.outcome_types),):
        raise ValueError('One utility weight per outcome is required.')
    costs = np.zeros(len(dgp.treatment_types)) if treatment_costs is None else np.asarray(treatment_costs, float)
    if costs.shape != (len(dgp.treatment_types),):
        raise ValueError('One treatment cost per component is required.')
    predicted = np.column_stack([model.predict(X, treatment=t) @ weights - t @ costs for t in interventions])
    truth = np.column_stack([dgp.mean(X, t) @ weights - t @ costs for t in interventions])
    choice = predicted.argmax(1)
    value = truth[np.arange(len(X)), choice]
    return dict(policy_value=float(value.mean()), oracle_value=float(truth.max(1).mean()),
                policy_regret=float((truth.max(1)-value).mean()), candidate_count=len(interventions))
