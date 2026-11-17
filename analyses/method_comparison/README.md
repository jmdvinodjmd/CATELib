# Small all-method comparison

One dataset only: **240 patients, 3 features, two binary interventions**, and two
outcomes (one continuous, one binary). The patient split is 144 training / 48
validation / 48 test. Binary interventions let all 18 implemented methods/variants
participate; this example does not test continuous doses or unseen combinations.

Methods: h, s, xh, xs, t, ridge, dec, vsr, scp, scp-s, cisi, cihsi, ncore, tece,
tece-marginal, dr, tarnet and cfr. These are the implemented adaptations documented
in their modules, not a claim of exact reproduction of every original paper.

```bash
bash analyses/method_comparison/study.sh
```

Each method gets its own small search (up to eight sampled configurations), small
8/16-unit networks and at most 20 epochs with early stopping. Two evaluation seeds
reuse the selected parameters and the same data. The modest search is for a fast
integration example, not evidence of best achievable performance. Architecture,
training, sample size and search settings remain editable in config.json/tuning.json.
The larger H-learner study is unchanged.

Open analysis.ipynb for the entire direct-component workflow and saved-result plots.
RUN_EXPERIMENTS=False loads results without fitting. Outputs are stored in
results/method_comparison, including completion and coverage tables, per-outcome
CATE and conditional-mean MSE, interaction errors, oracle policy regret/value,
validation scores, tuning costs and evaluation fit times. Scores across incomplete
or differing intervention coverage are not silently treated as comparable.
Repeated identical commands resume verified searches/evaluations.

Optional filters use the same shell script:

```bash
METHOD=s bash analyses/method_comparison/study.sh
```

H-learner and xH are included normally. This example is self-contained and does
not require the separate, detailed H-learner research analysis folder.

## Two complete workflows

`study.sh` now trains/tunes, saves evaluations, and automatically exports the final
CSV tables and PNG plots. It does not require opening a notebook. Reports are
saved under `results/method_comparison/reports/<scenario>/` (or your overridden output directory).

The notebook is an independent alternative: enable RUN_EXPERIMENTS to fit, then
run its reporting cell. Both paths use `catelib.plotting.report_results`; the
notebook displays the same tables and figures it saves. With training disabled,
the reporting cell reads saved results only.

To regenerate reports without any training:

```bash
uv run python -m catelib.run --config analyses/method_comparison/config.json --report-only
```

Pass the same `--output` and `--seeds` overrides used for the experiment when
applicable. In the all-method example, `--method ridge` restricts
report expectations to the requested methods. Missing runs remain visible.
