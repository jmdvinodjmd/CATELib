# Adding a component

The core estimator API is independent of experiment configuration and result files.
See the runnable examples in `extensions/`; these are not research analyses.

## Method

Implement an sklearn-compatible constructor with explicit parameters, `fit(X, Y,
treatment=T, validation_data=(Xv,Yv,Tv))`, `predict(X, treatment=T)` and
`effect(X, treatment=T1, reference=T0)`. `fit` returns self. Outcomes/predictions
have shape (patients, outcomes), including a single outcome. Inherit BaseEstimator
for get_params/set_params and clone support. Store learned attributes with a
trailing underscore. Report unsupported treatments explicitly.

For shared neural optimization, inherit `catelib.methods.base.NeuralEstimator`
and implement `_build_network`. The network belongs to your method and can be an
MLP, transformer or another differentiable architecture. Follow the training loop's
forward and optional loss-hook contract. S/H are concrete examples. Multi-stage
methods own their stage order and call reusable training helpers as needed;
forests and other non-neural models do not inherit the neural base.

Register the constructor in `catelib.run.METHODS` only if name-based CLI access is
wanted. See `extensions/new_method.py`. Method-specific search spaces live in each
analysis's tuning.json; the tuner does not need a branch for a new estimator.

## Dataset

Return `catelib.data.Dataset`. Arrays are X, Y and treatment with matching patient
rows. Declare treatment/outcome types. Optional truth supplies `mean(X,T)` and,
for the default simulated runner, `intervention_grid(doses)`. Observational data
need no truth; provide explicit evaluation interventions where appropriate.
Register a provider accepting n/data_seed and its own parameters in DATASETS.
Keep raw data in data/ or a user-provided path, not the package.

## Metric

A runner metric receives `(model, data, X_test, interventions, reference, options)`
and returns a dictionary or list of dictionaries. Register it in METRICS; add its
name to TRUTH_REQUIRED if it needs simulation truth. Direct calls need no registry.
New observational metrics and policy estimators are separate scientific additions,
not silently substituted for oracle simulation quantities.

## CLI and configuration

```bash
uv run python -m catelib.run --extension extensions.new_method --method additive --dataset composite
```

Config defaults are followed by scenario overrides, CLI flags, and finally `--set`
JSON overrides. For example `--set training.batch_size=128`,
`--set 'parameters.hidden_sizes=[64,64]'`, or
`--set 'split.fractions=[0.7,0.15,0.15]'`. Unsupported parameters raise errors.
Common training settings apply to matching nested estimators and are recorded in
saved effective parameters. Do not duplicate a parameter in training and parameters.
Do not tune model random_state: tuning initialization and evaluation seeds differ.
Grid/conditional spaces and nested parameter names are supported through sklearn.
Run only one writer per result directory. Progress is printed in the analysis itself.
