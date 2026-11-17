"""Plots consume saved tables; they never train estimators."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from catelib.results import completed_model


def load_results(folder):
    """Load only checksum-verified completed evaluations, excluding reference contrasts."""
    rows = []
    for path in sorted(Path(folder).rglob('evaluation_complete.json')):
        status = completed_model(path.parent, 'evaluation')
        if status is None or status.get('status') != 'ok':
            continue
        payload = json.loads((path.parent/'evaluation_metrics.json').read_text())
        for metric, records in payload['metrics'].items():
            for row in records if isinstance(records,list) else [records]:
                if not row.get('is_reference',False):
                    rows.append(dict(method=payload['method'], seed=payload['seed'],
                                     scenario=payload['scenario'], metric=metric, **row))
    return pd.DataFrame(rows)


def plot_comparison(table, *, metric='cate_mse', ax=None):
    """Mean over saved rows; no inferential confidence claims across fixed-data seeds."""
    if table.empty or metric not in table:
        raise ValueError(f'No saved {metric} results available.')
    ax = ax or plt.subplots()[1]
    table.groupby('method')[metric].mean().sort_values().plot.bar(ax=ax)
    ax.set_ylabel(metric)
    ax.set_title('Mean saved error (evaluation seeds share patients)')
    ax.figure.tight_layout()
    return ax


def report_results(folder, *, scenario, expected_methods, expected_seeds, show=False):
    """Export saved-result tables and figures for one scenario; never fit models.

    CLI and notebooks call this same reporting function. Comparisons use verified
    artifacts, reject duplicate configurations, and display missing coverage.
    Output is reports/<scenario>/ under the result directory. show=True displays
    tables/figures in a notebook as well as saving them. Returned tables support
    further analysis-specific plots without fitting again.
    """
    OUTPUT = Path(folder) / 'reports' / scenario
    if show:
        from IPython.display import display
    else:
        def display(value):
            pass
    import pandas as pd
    import matplotlib.pyplot as plt
    from catelib.results import completed_model

    metric_rows, timing_rows, search_rows = [], [], []
    seen, searched = set(), set()
    for path in sorted(Path(folder).rglob('evaluation_complete.json')):
        if completed_model(path.parent, 'evaluation') is None:
            continue
        payload = json.loads((path.parent/'evaluation_metrics.json').read_text())
        name, seed = payload['method'], payload['seed']
        if name not in expected_methods or seed not in expected_seeds or payload['scenario'] != scenario:
            continue
        key = (name, seed)
        if key in seen:
            raise ValueError(f'Multiple saved configurations for {key}. Choose a separate OUTPUT directory before comparing.')
        seen.add(key)
        timing_rows.append(dict(method=name, seed=seed, fit_seconds=payload['fit_seconds']))
        for metric, records in payload['metrics'].items():
            for row in records if isinstance(records, list) else [records]:
                if not row.get('is_reference', False):
                    metric_rows.append(dict(method=name, seed=seed, metric=metric, **row))
        search_folder = path.parent.parent / 'tuning'
        if search_folder not in searched:
            searched.add(search_folder)
            for marker in search_folder.glob('*/search-*_complete.json'):
                record = completed_model(marker.parent, marker.name.removesuffix('_complete.json'))
                if record is None:
                    continue
                trials = record['records']
                successful = [trial for trial in trials if trial['status']=='ok']
                search_rows.append(dict(method=name, candidates=len(trials),
                    failed_candidates=len(trials)-len(successful),
                    tuning_seconds=sum(trial['fit_seconds'] for trial in trials),
                    best_validation_loss=min(trial['score'] for trial in successful)))

    completion = pd.DataFrame([dict(method=name, seed=seed, completed=(name,seed) in seen)
                              for name in expected_methods for seed in expected_seeds])
    display(completion.pivot(index='method',columns='seed',values='completed'))
    print(f'{len(seen)}/{len(completion)} expected evaluations completed')
    results = pd.DataFrame(metric_rows)
    timings = pd.DataFrame(timing_rows)
    searches = pd.DataFrame(search_rows, columns=['method','candidates','failed_candidates','tuning_seconds','best_validation_loss'])
    if results.empty:
        print('No completed results yet. Run study.sh or enable RUN_EXPERIMENTS above.')
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        completion.to_csv(OUTPUT/'completion.csv', index=False)
        results.to_csv(OUTPUT/'metrics.csv', index=False)
        searches.to_csv(OUTPUT/'tuning_summary.csv', index=False)
        display(searches)

    if not results.empty:
        effects = results[results.metric=='effects'].copy()
        effects['contrast'] = effects['intervention'].map(json.dumps)
        effects['supported'] = effects['cate_mse'].notna() & effects['status'].eq('ok')
        coverage = effects.groupby(['method','seed','outcome']).agg(
            requested=('supported','size'), supported=('supported','sum'))
        coverage['coverage'] = coverage['supported']/coverage['requested']
        coverage.to_csv(OUTPUT/'coverage.csv')
        display(coverage)
        available = effects[effects.supported].groupby(['method','seed','outcome'])[['cate_mse','mean_mse']].mean()
        available.to_csv(OUTPUT/'available_contrast_errors.csv')
        common_parts = []
        for (seed, outcome), group in effects.groupby(['seed','outcome']):
            if set(group.method) != set(expected_methods):
                continue
            valid = group[group.supported]
            counts = valid.groupby('contrast')['method'].nunique()
            common = counts[counts==len(expected_methods)].index
            common_parts.append(valid[valid.contrast.isin(common)])
        common_effects = pd.concat(common_parts, ignore_index=True) if common_parts else pd.DataFrame()
        if common_effects.empty:
            print('No complete common-contrast comparison yet; inspect completion/coverage above.')
        else:
            per_seed = common_effects.groupby(['method','seed','outcome'])[['cate_mse','mean_mse']].mean().reset_index()
            per_seed.to_csv(OUTPUT/'common_contrast_errors.csv', index=False)
            display(per_seed.groupby(['method','outcome'])[['cate_mse','mean_mse']].agg(['mean','std']))
            for outcome, rows in per_seed.groupby('outcome'):
                for metric in ('cate_mse','mean_mse'):
                    ax = plot_comparison(rows, metric=metric)
                    ax.set_title(f'Outcome {int(outcome)}: common-contrast {metric}')
                    ax.figure.savefig(OUTPUT/f'outcome-{int(outcome)}-{metric}.png', dpi=150)
                    if not show: plt.close(ax.figure)

    if not results.empty:
        for metric_name in ('interactions','policy'):
            rows = results[results.metric==metric_name].copy()
            display(rows.dropna(axis=1, how='all'))
            rows.to_csv(OUTPUT/f'{metric_name}.csv', index=False)
            targets = ['interaction_mse'] if metric_name=='interactions' else ['policy_regret','policy_value']
            for metric in targets:
                if metric not in rows:
                    continue
                valid = rows[rows[metric].notna()]
                if valid.empty:
                    continue
                groups = valid.groupby('outcome') if metric_name=='interactions' else [('utility',valid)]
                for outcome, group in groups:
                    ax = plot_comparison(group, metric=metric)
                    ax.set_title(f'{metric}: {outcome} (see coverage/status above)')
                    ax.figure.savefig(OUTPUT/f'{metric}-{outcome}.png', dpi=150)
                    if not show: plt.close(ax.figure)

    if not timings.empty:
        fit_summary = timings.groupby('method').fit_seconds.agg(['mean','std']).rename(columns={'mean':'mean_fit_seconds','std':'fit_seconds_sd'})
        cost = fit_summary.join(searches.set_index('method')[['tuning_seconds','candidates','failed_candidates']])
        cost.to_csv(OUTPUT/'runtime.csv')
        display(cost)
        for column in ('mean_fit_seconds','tuning_seconds'):
            if cost[column].notna().sum() == 0:
                continue
            ax = cost[column].sort_values().plot.bar(figsize=(10,4), ylabel='seconds', title=column)
            ax.figure.tight_layout()
            ax.figure.savefig(OUTPUT/f'{column}.png', dpi=150)
            if show: plt.show()
            else: plt.close(ax.figure)

    return dict(results=results, completion=completion, timings=timings,
                searches=searches, output=OUTPUT)
