"""Подготовка данных, обучение и применение модели оттока клиентов."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
                            recall_score, f1_score, fbeta_score, accuracy_score,
                            confusion_matrix, brier_score_loss)
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
TARGET = 'Churn'
NUMERIC = ['Call Failure', 'Subscription Length', 'Charge Amount', 'Seconds of Use',
           'Frequency of use', 'Frequency of SMS', 'Distinct Called Numbers']
CATEGORICAL = ['Complains', 'Tariff Plan']
FEATURES = NUMERIC + CATEGORICAL
EXCLUDED = ['Status', 'Customer Value', 'Age', 'Age Group']


def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    """В исходном CSV есть двойные пробелы; значения не преобразуются."""
    result = df.copy()
    result.columns = [' '.join(str(c).split()) for c in result.columns]
    if result.columns.duplicated().any():
        raise ValueError('После нормализации названия столбцов повторяются')
    return result


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_names(df)
    absent = sorted(set(FEATURES) - set(df.columns))
    if absent:
        raise ValueError('Нет обязательных признаков: ' + ', '.join(absent))
    x = df.loc[:, FEATURES].copy()
    for c in FEATURES:
        x[c] = pd.to_numeric(x[c], errors='raise')
        if np.isinf(x[c].to_numpy(dtype=float)).any():
            raise ValueError('Бесконечное значение в столбце ' + c)
        if (x[c].dropna() < 0).any():
            raise ValueError('Отрицательное значение в столбце ' + c)
    return x


def load_data(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = clean_names(pd.read_csv(path))
    validate_features(raw)
    if TARGET not in raw or not set(raw[TARGET].unique()) <= {0, 1}:
        raise ValueError('Метка Churn должна содержать 0 или 1')
    data = raw.drop_duplicates().copy()
    data['source_row'] = data.index.to_numpy() + 2  # Строка CSV с учетом заголовка.
    data.reset_index(drop=True, inplace=True)
    # Группируем по используемым признакам, а не по целевой переменной.
    data['profile_group'] = pd.util.hash_pandas_object(data[FEATURES], index=False).astype(str)
    conflicting = int((data.groupby('profile_group')[TARGET].nunique() > 1).sum())
    audit = {'raw_rows': len(raw), 'clean_rows': len(data),
             'exact_duplicates_removed': int(raw.duplicated().sum()),
             'missing_values': int(raw.isna().sum().sum()),
             'raw_churn': int(raw[TARGET].sum()), 'clean_churn': int(data[TARGET].sum()),
             'unique_profiles': int(data.profile_group.nunique()),
             'conflicting_profiles': conflicting,
             'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
             'features': FEATURES, 'excluded': EXCLUDED}
    return data, audit


def split_data(data: pd.DataFrame) -> dict[str, np.ndarray]:
    """Фиксированное разделение около 60/20/20 без пересечения профилей."""
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rest, test = next(outer.split(data[FEATURES], data[TARGET], data.profile_group))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + 1)
    tr, val = next(inner.split(data.iloc[rest][FEATURES], data.iloc[rest][TARGET], data.iloc[rest].profile_group))
    return {'train': rest[tr], 'validation': rest[val], 'test': test}


def preprocessing() -> ColumnTransformer:
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')),
                        ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer([('numeric', numeric, NUMERIC), ('categorical', categorical, CATEGORICAL)])


def candidates():
    return {
        'dummy': (DummyClassifier(strategy='prior'), {}),
        'logistic': (LogisticRegression(max_iter=2000, random_state=SEED),
                     {'model__C': [0.1, 1.0, 10.0], 'model__class_weight': [None, 'balanced']}),
        'forest': (RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1),
                   {'model__max_depth': [8, None], 'model__min_samples_leaf': [2, 8],
                    'model__class_weight': [None, 'balanced']}),
        # early_stopping=False исключает скрытое случайное разделение внутри fit.
        'boosting': (HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08,
                    early_stopping=False, random_state=SEED),
                     {'model__max_leaf_nodes': [7, 15], 'model__l2_regularization': [1.0, 5.0]})
    }


def select_threshold(y, p) -> tuple[float, pd.DataFrame]:
    """Порог выбирается только на validation по F2, при равенстве выше порог."""
    records = []
    for t in np.arange(1, 100) / 100:
        pred = np.asarray(p) >= t
        records.append({'threshold': float(t), 'precision': precision_score(y, pred, zero_division=0),
                        'recall': recall_score(y, pred, zero_division=0),
                        'f1': f1_score(y, pred, zero_division=0),
                        'f2': fbeta_score(y, pred, beta=2, zero_division=0)})
    table = pd.DataFrame(records)
    best = table.sort_values(['f2', 'threshold'], ascending=[False, False]).iloc[0]
    return float(best.threshold), table


def metrics(y, p, threshold):
    prediction = np.asarray(p) >= threshold
    return {'ap': float(average_precision_score(y, p)), 'roc_auc': float(roc_auc_score(y, p)),
            'precision': float(precision_score(y, prediction, zero_division=0)),
            'recall': float(recall_score(y, prediction, zero_division=0)),
            'f1': float(f1_score(y, prediction, zero_division=0)),
            'f2': float(fbeta_score(y, prediction, beta=2, zero_division=0)),
            'accuracy': float(accuracy_score(y, prediction)),
            'brier': float(brier_score_loss(y, p)),
            'confusion': confusion_matrix(y, prediction, labels=[0, 1]).tolist()}


def bootstrap_intervals(y, p, threshold, groups, repeats=1000):
    """Bootstrap профилей сохраняет зависимость строк одной группы."""
    rng = np.random.default_rng(SEED)
    y, p = np.asarray(y), np.asarray(p)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    values = {k: [] for k in ('ap', 'roc_auc', 'recall', 'precision', 'f1')}
    for _ in range(repeats):
        idx = np.concatenate([members[g] for g in rng.choice(unique, len(unique), replace=True)])
        if len(np.unique(y[idx])) < 2: continue
        result = metrics(y[idx], p[idx], threshold)
        for key in values: values[key].append(result[key])
    return {key: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for key, v in values.items()}


def train(output=ROOT/'results'):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    data, audit = load_data(ROOT/'data/customer_churn.csv')
    splits = split_data(data)
    assignment = data[['source_row', 'profile_group', TARGET]].copy()
    assignment['split'] = ''
    for key, idx in splits.items(): assignment.loc[idx, 'split'] = key
    assignment.to_csv(output/'split_manifest.csv', index=False)
    split_info = {k: {'rows': len(ix), 'churn': int(data.iloc[ix][TARGET].sum()),
                      'profiles': int(data.iloc[ix].profile_group.nunique())} for k, ix in splits.items()}
    for a, b in [('train','validation'), ('train','test'), ('validation','test')]:
        assert set(data.iloc[splits[a]].profile_group).isdisjoint(data.iloc[splits[b]].profile_group)
    tr, va, te = (data.iloc[splits[k]] for k in ('train','validation','test'))
    cv = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED+2).split(
        tr[FEATURES], tr[TARGET], tr.profile_group))
    fitted, comparison, search_rows = {}, [], []
    for name, (model, grid) in candidates().items():
        print('Обучение:', name, flush=True)
        pipe = Pipeline([('prepare', preprocessing()), ('model', model)])
        search = GridSearchCV(pipe, grid, scoring='average_precision', cv=cv, n_jobs=1,
                              refit=True, error_score='raise', return_train_score=False)
        start = time.perf_counter(); search.fit(tr[FEATURES], tr[TARGET])
        elapsed = time.perf_counter() - start
        fitted[name] = search.best_estimator_
        p = fitted[name].predict_proba(va[FEATURES])[:, 1]
        m = metrics(va[TARGET], p, .5)
        comparison.append({'model': name, 'cv_ap': float(search.best_score_),
                           'cv_std': float(search.cv_results_['std_test_score'][search.best_index_]),
                           'validation_ap': m['ap'], 'validation_roc_auc': m['roc_auc'],
                           'fit_seconds': elapsed, 'best_params': search.best_params_})
        for params, mean, std in zip(search.cv_results_['params'], search.cv_results_['mean_test_score'], search.cv_results_['std_test_score']):
            search_rows.append({'model': name, 'params': json.dumps(params), 'cv_ap': mean, 'cv_std': std})
    winner = max(comparison, key=lambda x: x['cv_ap'])['model']
    selected = fitted[winner]
    threshold, curve = select_threshold(va[TARGET], selected.predict_proba(va[FEATURES])[:, 1])
    curve.to_csv(output/'threshold_validation.csv', index=False)
    # Все выборы закончены. Test используется один раз для итоговой оценки.
    p = selected.predict_proba(te[FEATURES])[:, 1]
    final = metrics(te[TARGET], p, threshold)
    predictions = te[['source_row', 'profile_group', TARGET]].copy()
    predictions['probability'] = p
    predictions['prediction'] = (p >= threshold).astype(int)
    predictions.to_csv(output/'test_predictions.csv', index=False)
    importance = permutation_importance(selected, va[FEATURES], va[TARGET],
                                       scoring='average_precision', n_repeats=15, random_state=SEED, n_jobs=1)
    pd.DataFrame({'feature': FEATURES, 'mean_ap_drop': importance.importances_mean,
                  'std_ap_drop': importance.importances_std}).sort_values('mean_ap_drop', ascending=False).to_csv(output/'permutation_validation.csv', index=False)
    intervals = bootstrap_intervals(te[TARGET], p, threshold, te.profile_group)
    baseline_p = fitted['dummy'].predict_proba(te[FEATURES])[:, 1]
    result = {'seed': SEED, 'audit': audit, 'splits': split_info,
              'model_comparison': comparison, 'winner': winner, 'threshold': threshold,
              'threshold_rule': 'max validation F2; tie: larger threshold',
              'test': final, 'test_at_05': metrics(te[TARGET], p, .5),
              'dummy_test': metrics(te[TARGET], baseline_p, .5), 'bootstrap_95': intervals,
              'bootstrap_repeats': 1000,
              'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                            'pandas': pd.__version__, 'sklearn': __import__('sklearn').__version__,
                            'pyarrow': __import__('pyarrow').__version__},
              'environment': {'system': platform.platform(), 'processor': platform.processor()}}
    (output/'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(comparison).to_csv(output/'model_comparison.csv', index=False)
    pd.DataFrame(search_rows).to_csv(output/'cv_search.csv', index=False)
    bundle = {'pipeline': selected, 'threshold': threshold, 'features': FEATURES,
              'dataset_sha256': audit['sha256'], 'model_name': winner,
              'versions': result['versions']}
    joblib.dump(bundle, output/'model.joblib', compress=3)
    te[FEATURES].head(10).to_csv(ROOT/'data/example_customers.csv', index=False)
    print(json.dumps({'winner': winner, 'threshold': threshold, 'test': final}, ensure_ascii=False, indent=2))
    return result


def predict_frame(bundle, frame):
    x = validate_features(frame)
    if x.empty:
        return pd.DataFrame({'probability': pd.Series(dtype=float),
                             'risk_flag': pd.Series(dtype=int)})
    p = bundle['pipeline'].predict_proba(x)[:, 1]
    return pd.DataFrame({'probability': p, 'risk_flag': (p >= bundle['threshold']).astype(int)})


def predict_file(input_path, output_path, model_path, batch_size=10000):
    """Пакетное чтение CSV или Parquet без загрузки всего файла в память."""
    if batch_size <= 0: raise ValueError('Размер пакета должен быть положительным')
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    if input_path == output_path: raise ValueError('Входной и выходной файлы совпадают')
    if output_path.exists(): raise FileExistsError('Выходной файл уже существует: ' + str(output_path))
    bundle = joblib.load(model_path)  # Загружать только собственный доверенный файл.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(output_path.name + '.partial')
    if temp.exists(): raise FileExistsError('Уже существует временный файл: ' + str(temp))
    reader = None
    if input_path.suffix.lower() == '.parquet':
        import pyarrow.parquet as pq
        reader = pq.ParquetFile(input_path)
        frames = (b.to_pandas() for b in reader.iter_batches(batch_size=batch_size, columns=FEATURES))
    else:
        reader = pd.read_csv(input_path, chunksize=batch_size)
        frames = reader
    count = 0
    try:
        for frame in frames:
            result = predict_frame(bundle, frame)
            result.insert(0, 'row_id', np.arange(count, count + len(result)))
            result.to_csv(temp, mode='a', header=(count == 0), index=False)
            count += len(frame)
        if count == 0:
            pd.DataFrame(columns=['row_id','probability','risk_flag']).to_csv(temp,index=False)
        temp.replace(output_path)
    finally:
        if reader is not None:
            reader.close()
        if temp.exists(): temp.unlink()
    return count


def main():
    parser = argparse.ArgumentParser(description='Прогнозирование оттока клиентов')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('train')
    p = commands.add_parser('predict')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', type=Path, default=ROOT/'results/model.joblib')
    p.add_argument('--batch-size', type=int, default=10000)
    args = parser.parse_args()
    if args.command == 'train': train()
    else: print('Обработано строк:', predict_file(args.input, args.output, args.model, args.batch_size))


if __name__ == '__main__': main()
