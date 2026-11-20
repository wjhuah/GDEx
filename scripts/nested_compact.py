import sys, json, ast, itertools, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, f1_score
from lightgbm import LGBMClassifier, LGBMRegressor
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from coptic_examples.compact_features import COMPACT
from coptic_examples.config import Config
from coptic_examples.lexicon import load
cfg = Config()
out = cfg.reports / 'nested_20261120'
out.mkdir(parents=True, exist_ok=True)

def factory(name, classification, p):
    if name == 'RF':
        cls = RandomForestClassifier if classification else RandomForestRegressor
        m = cls(n_estimators=200 if classification else 100, random_state=42, n_jobs=4, **p)
    else:
        cls = LGBMClassifier if classification else LGBMRegressor
        m = cls(n_estimators=120, min_child_samples=20, random_state=42, n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True, **p)
    return make_pipeline(SimpleImputer(keep_empty_features=True), m)

def fit(m, x, y, w):
    m.fit(x, y, **{m.steps[-1][0] + '__sample_weight': w})
    return m

def main():
    ns = {}
    tree = ast.parse((cfg.work / 'scripts/compare_original_algorithms.py').read_text())
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))], type_ignores=[]), 'metrics', 'exec'), ns)
    rows, X, Y, folds = joblib.load(Path(sys.argv[1]))
    grids = {'RF': [dict(max_depth=d, min_samples_leaf=l, max_features=f) for d, l, f in itertools.product([8, 16], [5, 20], ['sqrt', 1.0])], 'LightGBM': [dict(num_leaves=n, learning_rate=lr, reg_lambda=l) for n, lr, l in itertools.product([7, 15], [0.03, 0.1], [2, 10])]}
    families = np.array([r['family'] for r in rows])
    texts = np.array([r['text_key'] for r in rows])
    results = []
    search = []
    importances = []
    for outer in range(5):
        vi = np.flatnonzero(folds == outer)
        held = set(texts[vi])
        ti = np.array([i for i in np.flatnonzero(folds != outer) if texts[i] not in held])
        inner = []
        for tr, va in GroupKFold(n_splits=2).split(X[ti], Y[ti], families[ti]):
            v = ti[va]
            h = set(texts[v])
            t = np.array([i for i in ti[tr] if texts[i] not in h])
            inner.append((t, v))
        for name, grid in grids.items():
            for task in ['classification', 'ranking']:
                done = out / f'{outer}_{name}_{task}.json'
                if done.exists():
                    saved = json.loads(done.read_text())
                    results.append(saved['result'])
                    search += saved['search']
                    importances += saved['importance']
                    continue
                options = []
                for params in grid:
                    scores = []
                    for tr, va in inner:
                        if task == 'classification':
                            cnt = np.bincount(Y[tr].astype(int))
                            w = np.array([len(tr) / (2 * cnt[int(y)]) for y in Y[tr]])
                            m = fit(factory(name, True, params), X[tr], Y[tr], w)
                            pred = m.predict_proba(X[va])[:, 1]
                            scores.append(float(roc_auc_score(Y[va], pred)))
                        else:
                            rank = [(i, v['grade']) for i in tr if rows[i]['y'] > 0 for v in rows[i]['ranks']]
                            ii = np.array([i for i, g in rank])
                            yy = np.array([g for i, g in rank])
                            m = fit(factory(name, False, params), X[ii], yy, np.ones(len(yy)))
                            met = ns['rank_metrics']([rows[i] for i in va], m.predict(X[va]))
                            parts = [met[s]['ndcg6'] for s in ['tab', 'crum'] if met[s]['ndcg6'] is not None]
                            scores.append(float(np.mean(parts)))
                    options.append(dict(outer=outer, algorithm=name, task=task, params=params, inner_scores=scores, objective=float(np.mean(scores))))
                best = max(options, key=lambda v: v['objective'])
                params = best['params']
                if task == 'classification':
                    cnt = np.bincount(Y[ti].astype(int))
                    w = np.array([len(ti) / (2 * cnt[int(y)]) for y in Y[ti]])
                    m = fit(factory(name, True, params), X[ti], Y[ti], w)
                    pred = m.predict_proba(X[vi])[:, 1]
                    metrics = dict(auc=float(roc_auc_score(Y[vi], pred)), macro_f1=float(f1_score(Y[vi], pred >= 0.5, average='macro')))
                else:
                    rank = [(i, v['grade']) for i in ti if rows[i]['y'] > 0 for v in rows[i]['ranks']]
                    ii = np.array([i for i, g in rank])
                    yy = np.array([g for i, g in rank])
                    m = fit(factory(name, False, params), X[ii], yy, np.ones(len(yy)))
                    metrics = ns['rank_metrics']([rows[i] for i in vi], m.predict(X[vi]))
                estimator = m.steps[-1][1]
                imp = estimator.feature_importances_ if name == 'RF' else estimator.booster_.feature_importance(importance_type='gain')
                imp = imp / max(1, imp.sum())
                im = dict(outer=outer, algorithm=name, task=task, values=dict(zip(COMPACT, imp.tolist())))
                r = dict(outer=outer, algorithm=name, task=task, params=params, metrics=metrics)
                results.append(r)
                search += options
                importances.append(im)
                done.write_text(json.dumps(dict(result=r, search=options, importance=[im]), indent=2))
                print(outer, name, task, metrics, flush=True)
    (out / 'results.json').write_text(json.dumps(dict(results=results, search=search, importance=importances, protocol=dict(outer=5, inner=2, group='spelling_family', threshold=0.5, grid=grids, ranking_objective='equal mean of available TAB and Crum NDCG@6', length_policy='unfiltered training/evaluation for comparison to original62; hard20 only at retrieval')), indent=2))
    for name in grids:
        for task in ['classification', 'ranking']:
            candidates = [v for v in search if v['algorithm'] == name and v['task'] == task]
            pp = defaultdict(list)
            for v in candidates:
                pp[json.dumps(v['params'], sort_keys=True)].append(v['objective'])
            params = json.loads(max(pp, key=lambda p: np.mean(pp[p])))
            folder = cfg.work / 'models/tuned15_20261120' / name
            folder.mkdir(parents=True, exist_ok=True)
            if task == 'classification':
                cnt = np.bincount(Y.astype(int))
                w = np.array([len(Y) / (2 * cnt[int(y)]) for y in Y])
                m = fit(factory(name, True, params), X, Y, w)
            else:
                rank = [(i, v['grade']) for i, r in enumerate(rows) if r['y'] > 0 for v in r['ranks']]
                ii = np.array([i for i, g in rank])
                yy = np.array([g for i, g in rank])
                m = fit(factory(name, False, params), X[ii], yy, np.ones(len(yy)))
            joblib.dump(m, folder / (task + '.joblib'))
            (folder / (task + '.json')).write_text(json.dumps(dict(columns=COMPACT, params=params, selection='highest mean inner-CV objective across outer searches', deployed=False), indent=2))
    print('Complete', flush=True)
if __name__ == '__main__':
    main()
