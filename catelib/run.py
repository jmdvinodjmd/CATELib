"""One optional experiment runner and CLI; estimators remain usable directly.

Register a method constructor, dataset provider or metric function in the explicit
catalogues below. CLI extensions can do the same in an importable Python module.
"""
import argparse
from catelib.progress import progress
import sys
if __name__ == "__main__":
    sys.modules["catelib.run"] = sys.modules[__name__]
from functools import partial
import importlib
import json
from pathlib import Path
import time
import numpy as np
from sklearn.base import clone

from catelib.methods.h_learner import HLearner
from catelib.methods.s_learner import SLearner
from catelib.methods.independent_outcomes import IndependentOutcomes
from catelib.methods.t_learner import TLearner
from catelib.methods.compositional_regression import CompositionalRegression
from catelib.methods.deconfounder import Deconfounder
from catelib.methods.vsr import VSR
from catelib.methods.scp import SCP
from catelib.methods.cisi_net import CISINet
from catelib.methods.cihsi_net import CIHSINet
from catelib.methods.ncore import NCoRE
from catelib.methods.tece_vae import TECEVAE
from catelib.methods.dr_learner import DRLearner
from catelib.methods.tarnet import TARNet
from catelib.datasets import composite, diabetes
from catelib.data import Dataset, split_data
from catelib.metrics.effects import evaluate, evaluate_interactions, evaluate_policy
from catelib.tuning import tune as tune_estimator
from catelib.results import experiment_folder, save_evaluation, completed_model

METHODS = {
    's': SLearner, 'ridge': CompositionalRegression,
    'xs': partial(IndependentOutcomes, estimator=SLearner()),
    't': partial(TLearner, estimator=SLearner()),
    'dec': partial(Deconfounder, estimator=SLearner()),
    'vsr': partial(VSR, estimator=SLearner()),
    'scp': partial(SCP, estimator=SLearner(), first_stage='dr-cfr'),
    'scp-s': partial(SCP, estimator=SLearner(), first_stage='s'),
    'cisi': CISINet, 'cihsi': CIHSINet, 'ncore': NCoRE,
    'tece': TECEVAE, 'tece-marginal': partial(TECEVAE, inference='marginal'),
    'dr': DRLearner, 'tarnet': partial(TARNet, balancing=0.0),
    'cfr': partial(TARNet, balancing=0.1),
}
METHODS.update(h=HLearner, xh=partial(IndependentOutcomes, estimator=HLearner()))

DATASETS = {'composite': composite, 'diabetes': diabetes}
BINARY_ONLY = {'t','vsr','scp','scp-s','cisi','cihsi','ncore','tece','tece-marginal','dr','tarnet','cfr'}


def effects(model, data, X, interventions, reference, options):
    return evaluate(model, data.truth, X, interventions, reference,
                    holdout_pattern=options.get('holdout_pattern'),
                    allow_unsupported=isinstance(model,(TLearner,DRLearner,TARNet)))


def interactions(model, data, X, interventions, reference, options):
    return evaluate_interactions(model, data.truth, X, reference, np.max(interventions,axis=0))


def policy(model, data, X, interventions, reference, options):
    return evaluate_policy(model, data.truth, X, interventions,
                           options.get('utility_weights', np.ones(data.Y.shape[1])),
                           options.get('treatment_costs'))

# Metric callable signature is shared; truth requirements are explicit.
METRICS = {'effects': effects, 'interactions': interactions, 'policy': policy}
TRUTH_REQUIRED = {'effects','interactions','policy'}


def make_estimator(method, *, outcome_types, parameters=None, training=None, seed=0):
    """Resolve explicit parameters, including nested sklearn estimator__ parameters."""
    model = clone(METHODS[method]())
    available = model.get_params(deep=True)
    values = {}
    for key in available:
        if key.split('__')[-1] == 'outcome_types':
            values[key] = tuple(outcome_types)
        elif key.split('__')[-1] == 'random_state':
            values[key] = seed
    for name,value in (training or {}).items():
        targets = [key for key in available if key.split('__')[-1] == name]
        if not targets:
            raise ValueError(f'{method} does not support training parameter {name}; use method-specific config.')
        for key in targets:
            values[key] = value
    overlap = set(parameters or {}) & {key for key in values if key.split('__')[-1] in (training or {})}
    if overlap:
        raise ValueError(f'Duplicate model/training parameters: {sorted(overlap)}')
    values.update(parameters or {})
    return model.set_params(**values)


def run(*, method, dataset='composite', dataset_params=None, split=None,
        parameters=None, training=None, tuning=None, seeds=(0,), metrics=('effects',),
        output='results/default', scenario='default', evaluation=None, data_seed=0):
    """Tune once, then evaluate fresh fits across seeds on the same patient splits.

    dataset may be a registered name or a Dataset holding observational arrays.
    No metric requiring counterfactual truth is valid without a truth provider.
    All fits use training patients; validation selects hyperparameters/stopping.
    Returns verified evaluation directories, not an in-memory fitted estimator.
    """
    import torch
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Supply at least one unique evaluation seed.')
    torch.set_num_threads(1)
    evaluation = dict(evaluation or {})
    data = DATASETS[dataset](data_seed=data_seed, **(dataset_params or {})) if isinstance(dataset,str) else dataset
    if not isinstance(data,Dataset):
        raise TypeError('Dataset providers must return Dataset.')
    if method in BINARY_ONLY and 'continuous' in data.treatment_types:
        raise ValueError(f'{method} supports binary interventions only.')
    unknown = set(metrics)-METRICS.keys()
    if unknown:
        raise ValueError(f'Unknown metrics: {unknown}')
    if data.truth is None and set(metrics)&TRUTH_REQUIRED:
        raise ValueError('Requested metrics require simulation truth; register an observational metric or use metrics=[].')
    splits = split_data(data, **dict({'seed':data_seed+2}, **(split or {})))
    train, validation, test = (data.subset(splits[k]) for k in ('train','validation','test'))
    missing = evaluation.get('missing_outcome_rate', 0)
    if not 0 <= missing < 1:
        raise ValueError('missing_outcome_rate must lie in [0,1).')
    if missing:
        X,Y,T = train
        mask=np.random.default_rng(data_seed+4).random(Y.shape)<missing
        train=(X,np.where(mask,np.nan,Y),T)
    base = make_estimator(method, outcome_types=data.outcome_types, parameters=parameters, training=training,
                          seed=(tuning or {}).get('model_seed',0))
    folder=experiment_folder(output,method=method,dataset=dataset,data_seed=data_seed,
                             train=train,validation=validation,test=test,estimator=base,
                             tuning=tuning,evaluation=evaluation,metrics=metrics,scenario=scenario)
    np.savez_compressed(folder/'data.npz',X=data.X,Y=data.Y,treatment=data.treatment,**splits)
    print(f'{scenario}/{method}: {len(train[0])} training, {len(validation[0])} validation, {len(test[0])} test patients',flush=True)
    best_params={}
    if tuning is not None:
        search_options={k:v for k,v in tuning.items() if k!='model_seed'}
        search=tune_estimator(base,train=train,validation=validation,save_dir=folder/'tuning',**search_options)
        best_params=search.best_params
        # Evaluation seeds represent initialization, not a tunable model choice.
        if any(k.split('__')[-1]=='random_state' for k in best_params):
            raise ValueError('Do not tune random_state; configure model_seed and evaluation seeds separately.')
    reference=np.asarray(evaluation.get('reference',[0 if t=='binary' else .5 for t in data.treatment_types]))
    interventions=np.asarray(evaluation['interventions']) if 'interventions' in evaluation else (
        data.truth.intervention_grid(evaluation.get('doses',[.25,.75])) if data.truth else np.array([reference]))
    outputs=[]
    for i,seed in enumerate(seeds):
        destination=folder/f'seed-{seed}'
        status=completed_model(destination,'evaluation')
        progress('evaluation', i, len(seeds), f'seed={seed} running')
        if status is not None:
            progress('evaluation', i+1, len(seeds), f'seed={seed} cached')
            outputs.append(destination)
            continue
        model=clone(base).set_params(**best_params)
        model.set_params(**{k:seed for k in model.get_params(deep=True) if k.split('__')[-1]=='random_state'})
        start=time.perf_counter()
        model.fit(*train[:2],treatment=train[2],validation_data=validation)
        elapsed=time.perf_counter()-start
        payload={}
        for name in metrics:
            try:
                payload[name]=METRICS[name](model,data,test[0],interventions,reference,evaluation)
            except ValueError as error:
                if method not in ('t','dr','tarnet','cfr'): raise
                payload[name]=dict(status='unsupported',reason=str(error))
        save_evaluation(destination,model=model,data=data,splits=splits,
                        interventions=interventions,reference=reference,metrics=payload,
                        method=method,scenario=scenario,seed=seed,fit_seconds=elapsed,best_params=best_params)
        progress('evaluation', i+1, len(seeds), f'seed={seed} {elapsed:.2f}s')
        outputs.append(destination)
    return outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--method', help='Estimator name; optional for --report-only')
    parser.add_argument('--report-only', action='store_true', help='Export final tables and plots from saved results, without fitting')
    parser.add_argument('--dataset')
    parser.add_argument('--scenario',default='default')
    parser.add_argument('--n',type=int)
    parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--tune',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--set',action='append',default=[],metavar='SECTION.KEY=JSON',help='Override any configuration leaf with a JSON value')
    parser.add_argument('--extension',action='append',default=[],help='Import module registering custom components')
    args=parser.parse_args()
    for module in args.extension:
        importlib.import_module(module)
    config=json.loads(args.config.read_text()) if args.config else {}
    if args.report_only:
        if not args.config:
            parser.error('--report-only requires --config')
        import matplotlib
        matplotlib.use('Agg')
        from catelib.plotting import report_results
        scenarios = list(config.get('methods', {}))
        if args.scenario != 'default':
            if args.scenario not in scenarios: parser.error('Unknown scenario')
            scenarios = [args.scenario]
        for scenario in scenarios:
            settings = dict(config.get('defaults', {}), **config.get('scenarios', {}).get(scenario, {}))
            folder = args.output or (args.config.parent/settings.get('output','../../results/default')).resolve()
            seeds = args.seeds if args.seeds is not None else settings.get('seeds',[0])
            # Honour output/seeds overrides exactly as the fitting command does.
            for override in args.set:
                key,value = override.split('=',1)
                if key == 'output': folder = Path(json.loads(value))
                elif key == 'seeds': seeds = json.loads(value)
            methods = config['methods'][scenario]
            if args.method:
                if args.method not in methods: parser.error('Method not configured for this scenario')
                methods = [args.method]
            if not methods: continue
            report = report_results(folder, scenario=scenario, expected_methods=methods, expected_seeds=seeds)
            print(f"Report: {report['output']}", flush=True)
        return
    if not args.method:
        parser.error('--method is required unless --report-only is used')
    options=dict(config.get('defaults',{}))
    options.update(config.get('scenarios',{}).get(args.scenario,{}))
    options.update(method=args.method,scenario=args.scenario)
    if args.scenario!='default' and args.scenario not in config.get('scenarios',{}):
        parser.error('Unknown scenario')
    options['parameters']=config.get('parameters',{}).get(args.method,{})
    options['training']=config.get('training',{}).get(args.method,{})
    options['parameters']=dict(options['parameters'],**config.get('scenario_parameters',{}).get(args.scenario,{}).get(args.method,{}))
    if args.dataset: options['dataset']=args.dataset
    if args.n is not None: options['dataset_params']=dict(options.get('dataset_params',{}),n=args.n)
    if args.seeds is not None: options['seeds']=args.seeds
    if args.output: options['output']=str(args.output)
    elif args.config:
        options['output']=str((args.config.parent/options.get('output','../../results/default')).resolve())
    if args.tune:
        if not args.config: parser.error('--tune requires a config with tuning_file')
        spaces=json.loads((args.config.parent/config.get('tuning_file','tuning.json')).read_text())
        options['tuning']=config.get('scenario_tuning',{}).get(args.scenario,{}).get(args.method,spaces[args.method])
    for override in args.set:
        key,value=override.split('=',1)
        target=options
        parts=key.split('.')
        for part in parts[:-1]:target=target.setdefault(part,{})
        target[parts[-1]]=json.loads(value)
    print('\nCATELib: '+args.method,flush=True)
    run(**options)


if __name__=='__main__':
    main()
