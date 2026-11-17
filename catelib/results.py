"""Durable, content-addressed experiment artifacts; no training logic.

Completed models are reused only when their manifest and file hashes match.
Files are replaced atomically, with completion written last. Interrupted fits
restart that model; interrupted evaluation can reuse a completed fitted model.
Do not run two writers against the same cache directory simultaneously.
"""
import hashlib
import json
import os
import time
from pathlib import Path
import joblib


def _json_value(value):
    """Preserve numpy search values as numbers, rather than strings on resume."""
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=_json_value).encode()).hexdigest()


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def source_identity():
    root = Path(__file__).resolve().parents[1]
    paths = [root/'uv.lock', root/'requirements.lock', root/'pyproject.toml'] + sorted((root/'catelib').rglob('*.py')) + sorted((root/'extensions').rglob('*.py'))
    return {str(path.relative_to(root)): file_hash(path) for path in paths if path.is_file()}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=_json_value, allow_nan=False))
    _replace(temporary, path)


def save_model(path, model):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    joblib.dump(model, temporary, compress=3)
    _replace(temporary, path)
    write_json(path.with_suffix('.json'), {'sha256': file_hash(path)})


def load_model(path):
    """Load only locally produced trusted checkpoints with matching checksums."""
    path = Path(path)
    manifest = path.with_suffix('.json')
    if not path.exists() or not manifest.exists():
        return None
    try:
        if json.loads(manifest.read_text())['sha256'] != file_hash(path):
            return None
    except (ValueError, KeyError):
        return None
    return joblib.load(path)


def finish_model(folder, name, status):
    folder = Path(folder)
    files = {p.name: file_hash(p) for p in folder.glob(name+'_*')
             if p.is_file() and not p.name.endswith(('.tmp', '_complete.json'))}
    write_json(folder/f'{name}_complete.json', {'status': status, 'files': files})


def completed_model(folder, name):
    path = Path(folder)/f'{name}_complete.json'
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text())
        for filename, checksum in manifest['files'].items():
            artifact = path.parent/filename
            if not artifact.is_file() or file_hash(artifact) != checksum:
                return None
        return manifest['status']
    except (ValueError, KeyError):
        return None


def _replace(temporary, path):
    """Bounded retries for transient cloud-sync locks; never silently lose writes."""
    for delay in (0.25, 0.5, 1, 2, 4, 8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            time.sleep(delay)
    os.replace(temporary, path)


def experiment_folder(output, *, method, dataset, data_seed, train, validation,
                      test, estimator, tuning, evaluation, metrics, scenario):
    """Shared identity for direct notebooks and CLI runs; no fitting occurs here."""
    import numpy as np
    import sklearn
    import torch
    arrays = []
    for part in (train, validation, test):
        for a in part:
            a = np.ascontiguousarray(a)
            arrays.append(dict(shape=a.shape,dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest()))
    import inspect
    model_class = type(estimator)
    implementation = inspect.getsourcefile(model_class)
    identity = dict(method=method, estimator_class=model_class.__module__+'.'+model_class.__qualname__,
                    estimator_source=file_hash(implementation) if implementation else None,
                    dataset=dataset if isinstance(dataset,str) else 'arrays',
                    data=arrays,data_seed=data_seed,parameters=estimator.get_params(deep=True),
                    tuning=tuning,evaluation=evaluation,metrics=list(metrics),scenario=scenario,
                    source=source_identity(),versions=dict(numpy=np.__version__,sklearn=sklearn.__version__,torch=torch.__version__))
    folder=Path(output)/scenario/method/digest(identity)[:20]
    write_json(folder/'identity.json',identity)
    return folder


def save_evaluation(folder, *, model, data, splits, interventions, reference,
                    metrics, method, scenario, seed, fit_seconds, best_params):
    """Persist model, factual/grid predictions, truth when available and completion.

    Only call after evaluation succeeds. Unsupported intervention predictions are
    explicitly NaN in arrays, with reasons saved separately. This helper does not fit.
    """
    import numpy as np
    from itertools import combinations
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    X,Y,T=data.subset(splits['test'])
    queries=[reference,*interventions]
    active=np.max(interventions,axis=0)
    for size in (1,2):
        for columns in combinations(range(len(reference)),size):
            q=np.array(reference,copy=True);q[list(columns)]=active[list(columns)];queries.append(q)
    for column,kind in enumerate(data.treatment_types):
        if kind == 'continuous':
            for dose in np.linspace(0,1,11):
                q=np.array(reference,copy=True);q[column]=dose;queries.append(q)
    queries=np.unique(np.asarray(queries),axis=0)
    predictions=np.full((len(queries),len(X),Y.shape[1]),np.nan)
    unsupported=[]
    for i,q in enumerate(queries):
        try:predictions[i]=model.predict(X,treatment=q)
        except ValueError as error:
            if method not in ('t','dr','tarnet','cfr'):raise
            unsupported.append(dict(intervention=q.tolist(),reason=str(error)))
    truth={} if data.truth is None else dict(truth=np.stack([data.truth.mean(X,q) for q in queries]))
    try:factual=model.predict(X,treatment=T)
    except ValueError:
        if method not in ('t','dr','tarnet','cfr'):raise
        factual=np.full_like(Y,np.nan)
    save_model(folder/'evaluation_model.joblib',model)
    np.savez_compressed(folder/'evaluation_predictions.npz',factual=factual,
                        prediction=predictions,interventions=queries,reference=reference,
                        policy_interventions=interventions,X_test=X,test_indices=splits['test'],**truth)
    write_json(folder/'evaluation_metrics.json',dict(method=method,scenario=scenario,seed=seed,
               metrics=metrics,fit_seconds=fit_seconds,best_params=best_params,unsupported=unsupported))
    write_json(folder/'evaluation_parameters.json',model.get_params(deep=True))
    write_json(folder/'evaluation_history.json',getattr(model,'history_',[]))
    finish_model(folder,'evaluation',dict(status='ok'))
