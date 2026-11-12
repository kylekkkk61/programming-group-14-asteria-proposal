#!/usr/bin/env python3
"""Identity-attribute and data-scope audit for the Asteria repayment-difficulty models.

Every variant reuses the locked fit/calibration/holdout partitions, the same five CV
folds and the CatBoost settings selected by pipeline.py, so differences come from the
features alone. Results are written to outputs/audit/; submitted outputs are untouched.

Tests
1. Feature-group removals (CV and holdout): sex, age, family, occupation proxies, all identity.
2. Data scope: Lean and Lean without any external-agency information.
3. Group outcomes: queue rate, recall, false positive rate, precision and calibration.
4. Paired bootstrap intervals for performance and gender-gap differences.
5. Proxy leakage: how well the remaining features predict gender.
6. Counterfactual gender flip for the model that uses gender.

Retention rule (data minimisation, decided before seeing results): an identity group is
used only if adding it raises mean CV AP and raises AP in at least four of five folds.
The first audit run removed each group from the sex-excluded model; this run adds each
group back to the final policy model to confirm the decision under its own settings.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from src.features import lean_columns, model_columns, EXCLUDED_FROM_MODELS, ID
from src.modelling import (SEED, prepare, tune, make_model, score, best_threshold,
                           fit_calibrator, calibrated)

SEX = ['CODE_GENDER']
AGE = ['DAYS_BIRTH', 'AGE_YEARS']
FAMILY = ['NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'INCOME_PER_PERSON', 'NAME_TYPE_SUITE']
OCCUPATION = ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE']
SOCIOECONOMIC = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_HOUSING_TYPE']
IDENTITY = SEX + AGE + FAMILY + OCCUPATION + SOCIOECONOMIC
EXTERNAL = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] + [
    f'AMT_REQ_CREDIT_BUREAU_{s}' for s in ['HOUR', 'DAY', 'WEEK', 'MON', 'QRT', 'YEAR']]
RETENTION_GROUPS = {'sex': SEX, 'age': AGE, 'family': FAMILY, 'occupation': OCCUPATION}


def top_mask(p, ids, capacity=0.2):
    n = int(np.ceil(len(p) * capacity))
    order = np.lexsort((ids, -p))
    mask = np.zeros(len(p), dtype=bool)
    mask[order[:n]] = True
    return mask


def age_band(age):
    return np.select([age < 30, age < 45, age < 60], ['<30', '30-44', '45-59'], '60+')


def group_rows(label, y, p, ids, frame, threshold):
    queue = top_mask(p, ids)
    columns = {'gender': frame.CODE_GENDER.to_numpy(), 'age_band': age_band(frame.AGE_YEARS.to_numpy()),
               'family_status': frame.NAME_FAMILY_STATUS.fillna('missing').to_numpy(),
               'income_type': frame.NAME_INCOME_TYPE.fillna('missing').to_numpy(),
               'education': frame.NAME_EDUCATION_TYPE.fillna('missing').to_numpy()}
    rows = []
    for column, values in columns.items():
        for v in sorted(set(values)):
            g = values == v; yy = y[g]
            if g.sum() < 100 or yy.sum() < 10:
                continue
            q = queue[g]
            rows.append(dict(model=label, group_column=column, group=v, n=int(g.sum()), positives=int(yy.sum()),
                base_rate=yy.mean(), mean_probability=p[g].mean(), calibration_ratio=p[g].mean() / yy.mean(),
                roc_auc=roc_auc_score(yy, p[g]), queue_rate=q.mean(), queue_recall=q[yy == 1].mean(),
                queue_false_positive_rate=q[yy == 0].mean(), queue_precision=yy[q].mean() if q.any() else np.nan,
                f2_flag_rate=(p[g] >= threshold).mean()))
    return rows


def gender_gaps(y, p, ids, gender):
    queue = top_mask(p, ids); m, f = gender == 'M', gender == 'F'
    return dict(queue_rate_gap=queue[m].mean() - queue[f].mean(),
                recall_gap=queue[m & (y == 1)].mean() - queue[f & (y == 1)].mean(),
                fpr_gap=queue[m & (y == 0)].mean() - queue[f & (y == 0)].mean())


def bootstrap(y, ids, gender, scores, baseline, reps, seed=2026):
    rng = np.random.default_rng(seed); draws = {k: [] for k in scores}
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        for k, p in scores.items():
            q = top_mask(p[i], np.arange(len(i)))  # resampled positions break ties
            g = gender[i]; yy = y[i]
            draws[k].append([average_precision_score(yy, p[i]), q[yy == 1].mean(),
                             q[g == 'M'].mean() - q[g == 'F'].mean(),
                             q[(g == 'M') & (yy == 1)].mean() - q[(g == 'F') & (yy == 1)].mean()])
    names = ['ap', 'top20_recall', 'gender_queue_gap', 'gender_recall_gap']
    rows = []
    for k, d in draws.items():
        d = np.array(d)
        for j, n in enumerate(names):
            lo, hi = np.percentile(d[:, j], [2.5, 97.5])
            rows.append(dict(model=k, metric=n, estimate_mean=d[:, j].mean(), ci95_low=lo, ci95_high=hi))
        if k != baseline:
            diff = np.array(draws[baseline]) - d
            for j, n in enumerate(names[:2]):
                lo, hi = np.percentile(diff[:, j], [2.5, 97.5])
                rows.append(dict(model=f'{baseline} minus {k}', metric=n, estimate_mean=diff[:, j].mean(),
                                 ci95_low=lo, ci95_high=hi))
    return pd.DataFrame(rows)


def main(args):
    src = Path(args.outputs); out = src / 'audit'; out.mkdir(exist_ok=True)
    x = pd.read_parquet(src / 'development_features.parquet'); y = x.pop('TARGET'); raw = x.copy()
    cats = list(x.select_dtypes(exclude='number').columns); x = prepare(x, cats)
    fit_cal, hold = train_test_split(np.arange(len(x)), test_size=.2, stratify=y, random_state=SEED)
    fit, cal = train_test_split(fit_cal, test_size=.25, stratify=y.iloc[fit_cal], random_state=SEED)
    saved = pd.read_csv(src / 'partitions.csv').set_index(ID).partition.reindex(x.index)
    for name, idx in [('fit', fit), ('calibration', cal), ('holdout', hold)]:
        if not saved.iloc[idx].eq(name).all():
            raise ValueError('Partitions differ from the reported run')
    config = json.loads((src / 'selected_parameters.json').read_text())['catboost_full']
    everything = list(x.columns); policy = model_columns(everything); lean = lean_columns(policy)
    drop = lambda cols, removed: [c for c in cols if c not in removed]
    add = lambda cols, extra: [c for c in everything if c in cols or c in extra]
    variants = {
        'all_attributes': everything,
        'policy': policy,
        'policy_plus_sex': add(policy, SEX),
        'policy_plus_age': add(policy, AGE),
        'policy_plus_family': add(policy, FAMILY),
        'policy_plus_occupation': add(policy, OCCUPATION),
        'identity_blind': drop(everything, IDENTITY),
        'lean': lean,
        'lean_no_external': drop(lean, EXTERNAL),
    }
    y_hold = y.iloc[hold].to_numpy(); ids = x.index[hold].to_numpy(); frame = raw.iloc[hold]
    gender = frame.CODE_GENDER.to_numpy()
    metrics, groups, folds, scores, models = [], [], {}, {}, {}
    saved_policy = pd.read_csv(src / 'holdout_catboost_full.csv').set_index(ID).reindex(frame.index)
    for label, cols in variants.items():
        print(f'{label}: {len(cols)} features', flush=True)
        _, cv = tune('catboost', x.iloc[fit][cols], y.iloc[fit], [config], args.threads, label, out)
        folds[label] = cv.sort_values('fold').pr_auc_ap.to_numpy()
        model = make_model('catboost', x[cols], config, args.threads).fit(x.iloc[fit][cols], y.iloc[fit])
        cal_raw = model.predict_proba(x.iloc[cal][cols])[:, 1]
        calibrator = fit_calibrator(y.iloc[cal], cal_raw)
        threshold = best_threshold(y.iloc[cal], calibrated(calibrator, cal_raw))
        raw_p = model.predict_proba(x.iloc[hold][cols])[:, 1]; p = calibrated(calibrator, raw_p)
        if label == 'policy':
            # The policy variant must reproduce the submitted model exactly.
            if list(cols) != joblib.load(src / 'catboost_full.joblib')['columns']:
                raise ValueError('Submitted model does not use the current feature policy; rerun pipeline.py')
            np.testing.assert_allclose(raw_p, saved_policy.raw_probability, rtol=1e-9, atol=1e-12)
        scores[label] = p; models[label] = (model, cols)
        pd.DataFrame({ID: ids, 'TARGET': y_hold, 'calibrated_probability': p}).to_csv(out / f'holdout_{label}.csv', index=False)
        q = top_mask(p, ids)
        row = dict(model=label, features=len(cols), cv_mean_ap=folds[label].mean(), cv_sd_ap=folds[label].std(ddof=1),
                   **score(y_hold, p, threshold), **{f'top{c}_cases': int(y_hold[top_mask(p, ids, c / 100)].sum()) for c in [10, 20, 30]},
                   **gender_gaps(y_hold, p, ids, gender))
        metrics.append(row); groups += group_rows(label, y_hold, p, ids, frame, threshold)

    # Retention decisions use paired CV folds only, never the holdout.
    decisions = {}
    for group, cols in RETENTION_GROUPS.items():
        diff = folds[f'policy_plus_{group}'] - folds['policy']
        decisions[group] = dict(columns=cols, fold_ap_gain_from_adding=diff.round(5).tolist(),
                                mean_gain=float(diff.mean()), folds_better=int((diff > 0).sum()),
                                would_pass_retention_rule=bool(diff.mean() > 0 and (diff > 0).sum() >= 4),
                                legal_bar=group == 'sex')

    # Proxy leakage: predict gender from the features each model is allowed to use.
    is_male = (raw.CODE_GENDER == 'M').astype(int)
    leakage = []
    for label in ['policy_plus_occupation', 'policy', 'identity_blind']:
        cols = variants[label]
        clf = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.08, random_seed=SEED, verbose=False,
                                 thread_count=args.threads, allow_writing_files=False,
                                 cat_features=[c for c in cols if c in cats])
        clf.fit(x.iloc[fit][cols], is_male.iloc[fit])
        imp = pd.Series(clf.get_feature_importance(), index=cols).sort_values(ascending=False).head(5)
        leakage.append(dict(feature_set=label, gender_prediction_auc=roc_auc_score(is_male.iloc[hold], clf.predict_proba(x.iloc[hold][cols])[:, 1]),
                            top_predictors='; '.join(f'{k} ({v:.1f})' for k, v in imp.items())))

    # Counterfactual flip: change only the applicant's gender in the model that uses it.
    model, cols = models['policy_plus_sex']
    flipped = x.iloc[hold][cols].copy(); flipped['CODE_GENDER'] = np.where(flipped.CODE_GENDER == 'M', 'F', 'M')
    base_raw = model.predict_proba(x.iloc[hold][cols])[:, 1]; flip_raw = model.predict_proba(flipped)[:, 1]
    # Calibration is monotone, so the raw-score queue is the same as the calibrated queue.
    raw_cut = np.sort(base_raw)[::-1][top_mask(base_raw, ids).sum() - 1]
    in_q, in_q_flip = base_raw >= raw_cut, flip_raw >= raw_cut
    counterfactual = dict(mean_abs_raw_change=float(np.abs(flip_raw - base_raw).mean()),
        men_mean_change_if_female=float((flip_raw - base_raw)[gender == 'M'].mean()),
        women_mean_change_if_male=float((flip_raw - base_raw)[gender == 'F'].mean()),
        men_leaving_queue_if_female=int((in_q & ~in_q_flip & (gender == 'M')).sum()),
        women_entering_queue_if_male=int((~in_q & in_q_flip & (gender == 'F')).sum()),
        men_in_queue=int((in_q & (gender == 'M')).sum()), women_in_queue=int((in_q & (gender == 'F')).sum()))

    boot = bootstrap(y_hold, ids, gender, scores, 'policy', args.reps)
    pd.DataFrame(metrics).to_csv(out / 'audit_metrics.csv', index=False)
    pd.DataFrame(groups).to_csv(out / 'audit_groups.csv', index=False)
    boot.to_csv(out / 'audit_bootstrap.csv', index=False)
    pd.DataFrame(leakage).to_csv(out / 'audit_proxy_leakage.csv', index=False)
    summary = dict(config=config, excluded_from_models=EXCLUDED_FROM_MODELS, variants={k: len(v) for k, v in variants.items()},
                   retention_rule='use a group only if adding it raises mean CV AP and AP in at least 4 of 5 folds; sex is excluded by law regardless',
                   retention_decisions=decisions, counterfactual_gender_flip=counterfactual, bootstrap_replicates=args.reps)
    (out / 'audit_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    cols_show = ['model', 'features', 'cv_mean_ap', 'cv_sd_ap', 'pr_auc_ap', 'roc_auc', 'brier', 'top10_cases',
                 'top20_cases', 'top30_cases', 'queue_rate_gap', 'recall_gap', 'fpr_gap']
    print(pd.DataFrame(metrics)[cols_show].round(4).to_string(index=False))
    print(json.dumps(decisions, indent=1)); print(pd.DataFrame(leakage).to_string(index=False))
    print(json.dumps(counterfactual, indent=1))
    print(boot[boot.model.str.contains(' minus ')].round(4).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', default='outputs')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--reps', type=int, default=2000)
    main(parser.parse_args())
