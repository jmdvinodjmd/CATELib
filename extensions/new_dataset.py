"""Register a simple provider; replace this with your own loader or simulator."""
from catelib.datasets import composite
from catelib.run import DATASETS


def binary_and_dose(*, n=2000, data_seed=0, **parameters):
    parameters.setdefault('treatment_types', ('binary','continuous'))
    return composite(n=n, data_seed=data_seed, **parameters)


DATASETS['binary_and_dose'] = binary_and_dose
