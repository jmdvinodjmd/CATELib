"""Explicit, resumable factual-validation search; never receives test truth.

Each method gets a declared list/grid of candidates. Candidate scores, failures
and fitted checkpoints are saved immediately, before trying the next candidate.
The selected fitted candidate is returned without a second fit. Changing inputs,
base settings or source changes the candidate cache key. Sequential writers only.
"""
from catelib.progress import progress
from dataclasses import dataclass
import hashlib
import time
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, ParameterSampler
from catelib.results import (digest, source_identity, write_json, save_model,
                               load_model, finish_model, completed_model)


def validation_score(prediction, Y, Yv, outcome_types):
    """Equal outcome weights; training-SD-scaled MSE or unscaled binary Brier loss.

    Only observed validation entries are scored. Nonfinite predictions on any
    observed entry fail the candidate rather than improving its score by omission.
    """
    prediction, Y, Yv = map(np.asarray, (prediction, Y, Yv))
    if prediction.shape != Yv.shape:
        raise ValueError('Validation prediction shape differs from the outcomes.')
    scores = []
    for j, kind in enumerate(outcome_types):
        train = np.isfinite(Y[:, j]); observed = np.isfinite(Yv[:, j])
        if not train.any() or not observed.any():
            raise ValueError('Each outcome needs observed training and validation labels.')
        if not np.isfinite(prediction[observed, j]).all():
            raise ValueError('Nonfinite predictions on observed validation entries.')
        scale = 1.0 if kind == 'binary' else max(float(Y[train, j].std()), 1e-6)
        scores.append(float(np.mean(((prediction[observed, j]-Yv[observed, j])/scale)**2)))
    score = float(np.mean(scores))
    if not np.isfinite(score):
        raise ValueError('Nonfinite validation score.')
    return score, scores


def _search(estimator, X, Y, T, validation_data, parameter_grid, *, cache_dir=None,
            max_candidates=None, retry_failed=False, scoring="factual_loss"):
    """Return (best already-fitted estimator, candidate records).

    Candidate grids use sklearn ParameterGrid syntax. Lists of singleton grids
    specify a small joint search budget without a large Cartesian product.
    Ties select the first candidate. The initialization seed is held fixed across
    candidates within a replication; independent study seeds assess variability.
    """
    Xv, Yv, Tv = validation_data
    Y, Yv = np.asarray(Y), np.asarray(Yv)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Yv.ndim == 1:
        Yv = Yv[:, None]
    kinds = estimator.get_params().get('outcome_types') or ('continuous',)*Y.shape[1]
    if isinstance(parameter_grid, dict) and 'space' in parameter_grid:
        count = max_candidates or parameter_grid.get('n_iter', 100)
        candidates = list(ParameterSampler(parameter_grid['space'], n_iter=count,
                                          random_state=parameter_grid.get('random_state', 0)))
    else:
        candidates = list(ParameterGrid(parameter_grid))
    if max_candidates is not None:
        if max_candidates < 1:
            raise ValueError('max_candidates must be positive.')
        candidates = candidates[:max_candidates]
    if not candidates:
        raise ValueError('The tuning grid has no candidates.')
    # Fail configuration mistakes before doing any expensive fits.
    for parameters in candidates:
        clone(estimator).set_params(**parameters)
    scope = None
    if cache_dir is not None:
        inputs = []
        for array in (X, Y, T, Xv, Yv, Tv):
            array = np.ascontiguousarray(array)
            inputs.append(dict(shape=array.shape, dtype=str(array.dtype),
                               sha256=hashlib.sha256(array.tobytes()).hexdigest()))
        identity = dict(inputs=inputs, estimator=type(estimator).__module__+"."+type(estimator).__qualname__,
                        base_parameters=estimator.get_params(deep=True), source=source_identity(),
                        scoring=scoring if isinstance(scoring,str) else scoring.__module__+'.'+scoring.__qualname__)
        scope = Path(cache_dir)/digest(identity)[:24]
        write_json(scope/'identity.json', identity)
        search_name = 'search-' + digest(candidates)[:16]
        complete = completed_model(scope, search_name)
        if complete is not None and not retry_failed:
            model = load_model(scope/f'{search_name}_model.joblib')
            if model is not None:
                progress('tuning', len(candidates), len(candidates), 'cached search')
                return model, [dict(r, reused=True) for r in complete['records']]
    best, best_score, records = None, float('inf'), []
    for index, parameters in enumerate(candidates):
        progress('tuning', index, len(candidates), f'candidate {index+1} running')
        trial = scope/digest(parameters)[:24] if scope is not None else None
        if trial is not None:
            trial.mkdir(parents=True, exist_ok=True)
        record = completed_model(trial, 'candidate') if trial is not None else None
        candidate = None
        reused = False
        if record is not None and not (retry_failed and record['status']=='failed'):
            if record['status']=='ok':
                candidate = load_model(trial/'candidate_model.joblib')
                reused = candidate is not None
            else:
                reused = True
        if not reused:
            candidate = clone(estimator).set_params(**parameters)
            start = time.perf_counter()
            try:
                candidate.fit(X, Y, treatment=T, validation_data=(Xv, Yv, Tv))
                prediction = candidate.predict(Xv, treatment=Tv)
                if scoring == 'factual_loss':
                    score, per_outcome = validation_score(prediction, Y, Yv, kinds)
                else:
                    score, per_outcome = float(scoring(prediction, Y, Yv, kinds)), []
                    if not np.isfinite(score):
                        raise ValueError('Scorer returned a nonfinite loss.')
                record = dict(parameters=parameters, score=score, per_outcome_scores=per_outcome,
                              status='ok', fit_seconds=time.perf_counter()-start)
                if trial is not None:
                    save_model(trial/'candidate_model.joblib', candidate)
            except (ValueError, FloatingPointError, RuntimeError) as error:
                record = dict(parameters=parameters, status='failed', error=str(error),
                              fit_seconds=time.perf_counter()-start)
            if trial is not None:
                write_json(trial/'candidate_record.json', record)
                finish_model(trial, 'candidate', record)
        record = dict(record, candidate_index=index, reused=reused)
        records.append(record)
        detail = record['status'] + (' (cached)' if reused else '')
        if record['status'] == 'failed':
            detail += ': ' + record.get('error', '').replace('\n', ' ')
        progress('tuning', index+1, len(candidates), detail,
                 keep=record['status']=='failed')
        if record['status']=='ok' and record['score'] < best_score:
            best, best_score = candidate, record['score']
        if scope is not None:
            write_json(scope/'search_progress.json', records)
    if best is None:
        raise ValueError(f'All tuning candidates failed: {records}')
    selected = min((r for r in records if r['status']=='ok'), key=lambda r:r['score'])
    if scope is not None:
        write_json(scope/'selection.json', selected)
        save_model(scope/f'{search_name}_model.joblib',best)
        finish_model(scope,search_name,dict(records=records))
    return best, records


@dataclass
class SearchResult:
    """Validation-selected fitted model and complete trial records (lower is better)."""
    best_estimator: object
    best_params: dict
    best_score: float
    history: list


def tune(estimator, *, param_space, train, validation, strategy="random",
         n_trials=100, random_state=0, scoring="factual_loss", save_dir=None,
         retry_failed=False):
    """Tune any cloneable CATELib estimator, without access to test data.

    train/validation are (X, Y, T) tuples. Grid searches enumerate all combinations;
    random searches sample n_trials, using sklearn distributions/list semantics.
    Lists of dictionaries express conditional spaces. Nested sklearn parameters
    (e.g. estimator__hidden_sizes) are passed directly to set_params. Scoring is
    factual_loss or a callable(prediction, training_Y, validation_Y, outcome_types)
    returning a scalar loss to minimise. Saved callable searches require a stable
    module-level callable (not a lambda), and record its implementation hash.
    Selection returns the fitted validation-selected candidate without silently
    refitting on validation patients. Changing evaluation seeds does not require
    repeating this search: freeze best_params and fit independent fresh clones.
    """
    if strategy not in ('grid', 'random'):
        raise ValueError('Supported search strategies are grid and random; Bayesian is not implemented yet.')
    if not callable(scoring) and scoring != 'factual_loss':
        raise ValueError('scoring must be factual_loss or a callable loss.')
    if n_trials < 1:
        raise ValueError('n_trials must be positive.')
    if callable(scoring) and save_dir is not None:
        import inspect
        try:
            code = inspect.getsource(scoring)
        except (OSError, TypeError):
            raise ValueError('Saved custom scoring requires a source-defined function.') from None
        # Separate scorer scopes; never reuse scores from another loss implementation.
        save_dir = Path(save_dir)/digest(dict(module=scoring.__module__, code=code))[:16]
    grid = param_space if strategy == 'grid' else dict(space=param_space, n_iter=n_trials, random_state=random_state)
    best, records = _search(estimator, *train, validation, grid, cache_dir=save_dir,
                            retry_failed=retry_failed, scoring=scoring)
    selected = min((r for r in records if r['status'] == 'ok'), key=lambda r:r['score'])
    result = SearchResult(best, selected['parameters'], selected['score'], records)
    if save_dir is not None:
        write_json(Path(save_dir)/'best.json', dict(parameters=result.best_params,
                   score=result.best_score, strategy=strategy, requested_trials=n_trials,
                   candidate_count=len(records)))
    return result
