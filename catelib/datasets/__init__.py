"""Dataset providers return factual data and optional simulation truth."""
import numpy as np
from .composite import CompositeDGP


def composite(*, n=2000, data_seed=0, **parameters):
    from catelib.data import Dataset
    dgp = CompositeDGP(mechanism_seed=data_seed, **parameters)
    X, T, Y, _ = dgp.sample(n, sample_seed=data_seed+1)
    return Dataset(X, Y, T, dgp.treatment_types, dgp.outcome_types, dgp)


def diabetes(*, n=400, data_seed=0, **parameters):
    """Real diabetes covariates, simulated assignment/outcomes; not clinical effects."""
    from sklearn.datasets import load_diabetes
    from catelib.data import Dataset
    population = load_diabetes().data
    if n > len(population):
        raise ValueError('Diabetes has 442 unique patients; reduce n.')
    k = len(parameters.get('treatment_types', ('binary','continuous','binary')))
    parameters.setdefault('feature_blocks', (population.shape[1]-k,) + (1,)*k)
    dgp = CompositeDGP(mechanism_seed=data_seed, **parameters)
    covariates = population[np.random.default_rng(data_seed+1).permutation(len(population))[:n]]*np.sqrt(len(population))
    X,T,Y,_ = dgp.sample(n, sample_seed=data_seed+1, covariates=covariates)
    return Dataset(X,Y,T,dgp.treatment_types,dgp.outcome_types,dgp)
