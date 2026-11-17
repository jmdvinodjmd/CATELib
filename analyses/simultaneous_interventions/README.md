# Simultaneous-intervention example

A standalone public example: S-learner versus treatment-feature ridge with one
binary treatment and one continuous dose. No H-learner imports or study files are
needed. `study.sh` runs one command per estimator. `analysis.ipynb` visibly creates
data, splits, estimators, tuning searches, fits, metrics and plots, using the same
library components. Set RUN_EXPERIMENTS=True to train; otherwise it loads results.

The default is a small demonstration (240 patients), not publication evidence.
Edit config.json and tuning.json for larger runs. Results go to
results/simultaneous_interventions. The public example tunes each estimator;
validation loss selects settings and test CATE truth is used only for reporting.
Evaluation seeds share patients and do not estimate population uncertainty.

## Two complete workflows

`study.sh` now trains/tunes, saves evaluations, and automatically exports the final
CSV tables and PNG plots. It does not require opening a notebook. Reports are
saved under `results/simultaneous_interventions/reports/<scenario>/` (or your overridden output directory).

The notebook is an independent alternative: enable RUN_EXPERIMENTS to fit, then
run its reporting cell. Both paths use `catelib.plotting.report_results`; the
notebook displays the same tables and figures it saves. With training disabled,
the reporting cell reads saved results only.

To regenerate reports without any training:

```bash
uv run python -m catelib.run --config analyses/simultaneous_interventions/config.json --report-only
```

Pass the same `--output` and `--seeds` overrides used for the experiment when
applicable. In the all-method example, `--method ridge` restricts
report expectations to the requested methods. Missing runs remain visible.
