"""Example truth-based metric using the runner's explicit metric contract."""
import numpy as np
from catelib.run import METRICS, TRUTH_REQUIRED


def cate_mae(model, data, X, interventions, reference, options):
    rows=[]
    for intervention in interventions:
        if np.array_equal(intervention,reference):
            continue
        prediction=model.effect(X,treatment=intervention,reference=reference)
        truth=data.truth.mean(X,intervention)-data.truth.mean(X,reference)
        rows.append(dict(intervention=intervention.tolist(),cate_mae=float(np.abs(prediction-truth).mean())))
    return rows


METRICS['cate_mae'] = cate_mae
TRUTH_REQUIRED.add('cate_mae')
