# CATELib: A Library for Personalised Causal Effect Estimation

**Conditional average treatment effect estimation for multiple simultaneous interventions.**

CATELib is a Python library for learning how treatment effects vary across people
when several interventions may be applied together. It provides readable,
scikit-learn-style estimators and an optional experiment harness for comparing
methods with shared data splits, tuning, evaluation and saved results.

Developed by **Vinod Kumar Chauhan**. Repository: [jmdvinodjmd/CATELib](https://github.com/jmdvinodjmd/CATELib).

> **Under development:** CATELib is research software in active development.
> APIs, defaults and implementations may change. The included examples demonstrate
> the software workflow; they are not evidence that any method is universally
> superior or that the implementations reproduce every published benchmark.
> Check each method's assumptions and implementation notes before interpreting
> results. The package is not validated for clinical or other high-stakes decisions.

## What CATELib supports

- **Simultaneous interventions:** represent a treatment as a vector of binary and/or
  continuous components. A single intervention is a special case. Individual
  estimators declare their supported treatment types.
- **Heterogeneous effects:** estimate conditional outcome means and treatment
  contrasts relative to a user-specified reference intervention.
- **Multiple outcomes:** continuous and binary outcomes, with method-dependent
  sharing across outcomes.
- **Direct estimator use:** construct a model, call `fit`, then `predict` or `effect`.
  No experiment configuration or runner is required.
- **Shared tuning:** grid and random search over estimator-specific parameters,
  including nested estimators, using training/validation data only.
- **Repeatable experiments:** explicit patient splits and seeds, saved predictions,
  fitted models, parameters, metrics and resumable completed searches/evaluations.
- **Two complete workflows:** shell scripts and notebooks independently run the
  analysis and produce the same saved-result tables and plots.
- **Extensibility:** add a method, dataset or metric without rebuilding the harness.
  Network architecture remains inside each estimator; compatible methods share
  numerical training helpers.

Treatments encoded from categorical variables must retain valid category
combinations. Predictions for interventions absent from training depend on the
method's structural assumptions; producing a prediction does not establish causal
identification without adequate assumptions and support.

## Installation

Python 3.11 or later is required. Clone the repository, then install with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/jmdvinodjmd/CATELib.git
cd CATELib
uv sync --extra notebooks
```

This installs the library and notebook-kernel dependencies in `.venv`, using the
committed `uv.lock`. It does not train models. Select that environment as the
notebook kernel. 

## Quickstart: use an estimator directly

```python
from catelib.datasets import composite
from catelib.data import split_data
from catelib.methods.h_learner import HLearner

# Three covariates, one binary intervention and one continuous dose.
data = composite(
    n=240,
    treatment_types=('binary', 'continuous'),
    outcome_types=('continuous',),
    feature_blocks=(1, 1, 1),
)
splits = split_data(data, seed=2)
X, Y, T = data.subset(splits['train'])
validation = data.subset(splits['validation'])

model = HLearner(
    hidden_sizes=(16,), hypernet_sizes=(16,),
    max_epochs=20, batch_size=64, random_state=0,
)
model.fit(X, Y, treatment=T, validation_data=validation)

X_test = data.X[splits['test']]
means = model.predict(X_test, treatment=[1, 0.75])
effects = model.effect(X_test, treatment=[1, 0.75], reference=[0, 0.5])
```

Here, `effects` compares the joint intervention `[1, 0.75]` with `[0, 0.5]` for each
patient. Arrays use `X=(patients, features)`, `T=(patients, interventions)` and
`Y=(patients, outcomes)`. Small settings above are for demonstration, not prescribed
architectures for final studies. Observational arrays can be supplied through
`catelib.data.Dataset` or directly to an estimator.

## Tune a method

```python
from catelib.tuning import tune

search = tune(
    model,
    train=(X, Y, T),
    validation=validation,
    param_space={
        'learning_rate': [0.0001, 0.001, 0.01],
        'hidden_sizes': [(8,), (16,), (32,)],
    },
    strategy='random',        # 'grid' is also supported
    n_trials=8,
    save_dir='results/my_search',
)
best_model = search.best_estimator
print(search.best_params, search.best_score)
```

Each estimator has its own search space. The common tuner fits candidates and
selects a lower-is-better factual validation loss; it never uses test CATE truth.
It returns the selected fitted estimator, parameters, score and trial history.
The example harness tunes once per estimator/dataset/setting and reuses selected
parameters across evaluation initialization seeds. Bayesian optimization is not
yet implemented. See [the tuning guide](docs/quickstart.md).

## Included methods

There are **18 named methods/variants**. They are documented implementations and
adaptations, rather than a claim of exact numerical reproduction of every paper.
References and limitations are included in their source modules.

| Supported interventions | Methods / CLI names |
| --- | --- |
| Binary and continuous components | H-learner (`h`), S-learner (`s`), independent-outcome variants (`xh`, `xs`), compositional ridge (`ridge`), deconfounder (`dec`) |
| Binary components | T-learner (`t`), VSR (`vsr`), SCP variants (`scp`, `scp-s`), CISI (`cisi`), CIHSI (`cihsi`), NCoRE (`ncore`), TECE variants (`tece`, `tece-marginal`), DR (`dr`), TARNet (`tarnet`), CFR (`cfr`) |

The names `xs` and `xh` refer to independent-outcome wrappers, not the imputed-effect
X-learner. Restrictions beyond treatment type still apply; inspect the relevant
[module](catelib/methods) before using a method on a new dataset.

## Run a complete example

### Small comparison of all methods

```bash
bash analyses/method_comparison/study.sh
```

This runs **one dataset scenario** with 240 patients, 3 features, two binary
interventions and two outcomes. All 18 methods, including H-learner, participate.
Each method gets a small tuning search, followed by two evaluation initialization
seeds. The script trains, evaluates and generates final CSV tables and PNG figures.

Alternatively, open [the complete comparison notebook](analyses/method_comparison/analysis.ipynb),
set `RUN_EXPERIMENTS = True`, and run its cells. The notebook constructs and fits
estimators directly and includes the final reporting step. It does not require
running the shell script first. With `RUN_EXPERIMENTS = False`, it loads saved results.

Results are saved under `results/method_comparison/`; reports are under
`results/method_comparison/reports/binary/`. Comparisons include per-outcome CATE
and conditional-mean errors, interaction errors, oracle policy value/regret,
intervention coverage, tuning costs and evaluation fit times.

To regenerate tables and figures without fitting:

```bash
uv run python -m catelib.run --config analyses/method_comparison/config.json --report-only
```

### Smaller mixed-intervention example

[Simultaneous interventions](analyses/simultaneous_interventions/README.md) compares
S-learner and ridge using a binary intervention and a continuous dose. It also has
an independent shell script and complete notebook:

```bash
bash analyses/simultaneous_interventions/study.sh
```

Examples intentionally use small architectures and limited searches for quick
inspection. They are not publication benchmarks. Initialization seeds share the
same patients and do not measure uncertainty across independent datasets. Policy
metrics use simulator truth over a finite intervention grid; they are not
observational off-policy evaluation estimates.

## Organisation and extension

```text
catelib/
    methods/        # estimator classes and method-owned architectures
    datasets/       # simulations and dataset providers
    metrics/        # evaluation functions
    training.py     # shared numerical training helpers
    tuning.py       # common parameter-search interface
    run.py          # single optional runner and CLI
    results.py      # saved artifacts and completion checks
    plotting.py     # shared saved-result reports
analyses/           # self-contained analyses: shell script + complete notebook
extensions/         # small method/dataset/metric extension examples
docs/               # direct-use and extension guides
data/               # local data; not required by the generated examples
results/            # generated outputs; not committed
```

Start with [direct use](docs/quickstart.md), [adding components](docs/extending.md)
and the [runnable extension examples](extensions). Estimators work without the
runner. Registration is needed only for name-based CLI access. Custom plotting can
use saved predictions and tables without refitting models.

## Development and contributions

Bug reports and contributions are welcome. Include the command, configuration,
package versions and traceback when reporting an issue. Keep examples small and
method logic readable. For local development checks:

```bash
uv sync --extra notebooks --extra dev
uv run pytest
```

## License

CATELib is available under the [MIT License](LICENSE), permitting use,
modification and redistribution, including commercial use, with the copyright
and licence notices preserved. See [third-party notices](THIRD_PARTY_NOTICES.md)
for upstream attribution. Dependencies retain their own licences.

## Citation

If you use CATELib, please cite the software and the original papers for the
methods you use. Method references are listed in their source modules, including
the [H-learner paper](https://arxiv.org/abs/2502.08282).

**Chauhan, Vinod Kumar (2026). CATELib: A Library for Personalised Causal Effect Estimation.**

```bibtex
@software{chauhan2026catelib,
  author  = {Chauhan, Vinod Kumar},
  title   = {CATELib: A Library for Personalised Causal Effect Estimation},
  year    = {2026},
  url     = {https://github.com/jmdvinodjmd/CATELib}
}
```

Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).
