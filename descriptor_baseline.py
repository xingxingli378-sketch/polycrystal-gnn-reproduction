"""Additional orientation/volume descriptor baseline; existing Anaconda sklearn."""
import csv
import json
import time
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
root = Path(__file__).parent
descriptors = []
targets = []
groups = []
for sid in range(1, 493):
    p = root / 'data' / ('structure-' + str(sid))
    f = np.loadtxt(p / 'feature.txt')[1:, 1:]
    angles = f[:, :3]
    volume = f[:, 3]
    w = volume / volume.sum()
    periodic = np.concatenate([np.sin(angles), np.cos(angles)], axis=1)
    features = []
    features.extend(periodic.mean(0))
    features.extend(periodic.std(0))
    features.extend(w @ periodic)
    features.extend(np.quantile(periodic, [.1, .5, .9], axis=0).flatten())
    # Bunge ZXZ convention: direction cosines of laboratory x in crystal frame.
    p1, P, p2 = angles.T
    c1 = np.cos(p1)
    s1 = np.sin(p1)
    c2 = np.cos(p2)
    s2 = np.sin(p2)
    c = np.cos(P)
    alpha = np.column_stack(
        [c1 * c2 - s1 * s2 * c, -c1 * s2 - s1 * c2 * c, s1 * np.sin(P)])
    fourth = (alpha**4).sum(1)
    features.extend([w @ fourth, fourth.mean(), fourth.std(), len(f) /
                     300, volume.std() / volume.mean(), f[:, 4].mean(), f[:, 4].std()])
    props = np.atleast_2d(np.loadtxt(p / 'property.txt'))
    for h, y in props[props.min(1) >= 0]:
        descriptors.append(features + [h / 10000])
        targets.append(y)
        groups.append(sid)
X = np.array(descriptors)
y = np.array(targets)
groups = np.array(groups)
assert len(y) == 2287 and np.isfinite(X).all()
for split in ['official', 'grouped']:
    s = np.load(root /
                'results' /
                ('%s_baseline_seed124' %
                 split) /
                'split.npz')
    assert np.array_equal(s['groups'], groups)
    tr, va, te = s['train'], s['validation'], s['test']
    out = root / 'results' / ('%s_descriptor_seed124' % split)
    out.mkdir(parents=True, exist_ok=True)
    candidates = {
        'extra_trees': ExtraTreesRegressor(n_estimators=300, min_samples_leaf=1, max_features=1., random_state=124, n_jobs=2),
        'random_forest': RandomForestRegressor(n_estimators=300, min_samples_leaf=1, max_features=1., random_state=124, n_jobs=2),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=500, max_depth=3, learning_rate=.03, random_state=124, loss='huber')}
    scores = {}
    fitted = {}
    start = time.time()
    for name, model in candidates.items():
        model.fit(X[tr], y[tr])
        scores[name] = float(
            np.sqrt(np.mean((model.predict(X[va]) - y[va])**2)))
        fitted[name] = model
        print(split, name, 'validation RMSE', scores[name], flush=True)
    winner = min(scores, key=scores.get)
    model = fitted[winner]
    yp = model.predict(X[te])
    report = {'split': split,
              'model': winner,
              'seed': 124,
              'samples': len(y),
              'descriptor_dimensions': X.shape[1],
              'validation_candidates': scores,
              'test_rmse': float(np.sqrt(np.mean((yp - y[te])**2))),
              'test_mae': float(np.mean(np.abs(yp - y[te]))),
              'test_relative_absolute_error': float(np.abs(yp - y[te]).sum() / np.abs(y[te]).sum()),
              'test_r2': float(1 - ((yp - y[te])**2).sum() / ((y[te] - y[te].mean())**2).sum()),
              'train_test_graph_overlap': len(set(groups[tr]) & set(groups[te])),
              'seconds': time.time() - start}
    (out / 'metrics.json').write_text(json.dumps(report, indent=2))
    joblib.dump(model, out / 'model.joblib')
    np.savetxt(out / 'predictions.csv',
               np.column_stack([groups[te],
                                X[te,
                                  -1] * 10000,
                                y[te],
                                yp]),
               delimiter=',',
               header='structure,field,true,predicted',
               comments='')
    np.savetxt(
        out / 'feature_importances.csv',
        model.feature_importances_,
        delimiter=',',
        header='importance_in_descriptor_column_order',
        comments='')
    print(json.dumps(report, indent=2), flush=True)
