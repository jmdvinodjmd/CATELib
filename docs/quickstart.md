# Direct estimator use

Install from the repository with `uv sync --extra notebooks`. Architectures and
training parameters are estimator constructor arguments; the runner is optional.

```python
from catelib.datasets import composite
from catelib.data import split_data
from catelib.methods.s_learner import SLearner
from catelib.tuning import tune

data = composite(n=2000, treatment_types=('binary', 'continuous'),
                 outcome_types=('continuous',))
splits = split_data(data, fractions=(0.6, 0.2, 0.2), seed=2)
train = data.subset(splits['train'])
validation = data.subset(splits['validation'])
model = SLearner(hidden_sizes=(64, 64), max_epochs=200, batch_size=128)
model.fit(train[0], train[1], treatment=train[2], validation_data=validation)
effect = model.effect(data.X[splits['test']], treatment=[1, .75], reference=[0, .5])
```

For tuning, replace the direct fit with:

```python
search = tune(model, train=train, validation=validation,
              param_space={'learning_rate': [.0001, .001, .01],
                           'hidden_sizes': [(32,32), (64,64), (128,128)]},
              strategy='random', n_trials=9, save_dir='results/my_search')
model = search.best_estimator
print(search.best_params, search.best_score)
```

Grid search enumerates combinations; random search samples the supplied space.
Nested parameters such as `estimator__hidden_sizes` tune composite estimators.
Bayesian search is not implemented yet. All candidates use the same development
split and lower-is-better validation loss; test effects never select candidates.
Completed candidates are reused. Failed candidates are reported, and all-failed
searches raise an error. Set `retry_failed=True` to retry recorded failures.
Only load checkpoints you created locally.

For observational arrays use `Dataset(X, Y, T, treatment_types, outcome_types)`
from `catelib.data`; ensure Y has shape (patients, outcomes). `truth` defaults to
None. The same splitting and tuning functions work; simulation CATE metrics do
not apply without ground truth. Users can supply their own metric functions.
Encoded categories must be queried with valid category combinations.
