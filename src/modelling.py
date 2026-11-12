"""Training-only model selection, independent calibration and locked holdout evaluation."""
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
    recall_score, fbeta_score, brier_score_loss, log_loss, precision_recall_curve, confusion_matrix)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 2026


def prepare(x, categories):
    out = x.copy()
    for c in categories:
        out[c] = out[c].fillna('__MISSING__').astype(str)
    return out


def make_model(kind, x, config, threads):
    cats = list(x.select_dtypes(exclude='number').columns)
    if kind == 'catboost':
        return CatBoostClassifier(**config, loss_function='Logloss', random_seed=SEED,
            thread_count=threads, verbose=False, allow_writing_files=False, cat_features=cats)
    nums = [c for c in x if c not in cats]
    numeric = Pipeline([('impute',SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                        ('scale',StandardScaler())])
    categorical = OneHotEncoder(handle_unknown='ignore')
    pre = ColumnTransformer([('numeric',numeric,nums),('categorical',categorical,cats)])
    return Pipeline([('preprocess',pre), ('classifier',LogisticRegression(
        **config, solver='liblinear', max_iter=2000, random_state=SEED))])


def score(y, p, threshold=0.5):
    pred = np.asarray(p) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return dict(n=len(y), positives=int(np.sum(y)), pr_auc_ap=average_precision_score(y,p),
        roc_auc=roc_auc_score(y,p), precision=precision_score(y,pred,zero_division=0),
        recall=recall_score(y,pred,zero_division=0), f2=fbeta_score(y,pred,beta=2,zero_division=0),
        brier=brier_score_loss(y,p), log_loss=log_loss(y,p), threshold=float(threshold),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))


def best_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f2 = 5 * precision[:-1] * recall[:-1] / np.maximum(4*precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[np.argmax(f2)])


def top_reviews(ids, y, p, credit, model):
    ids, y, p, credit = map(np.asarray, (ids,y,p,credit))
    if not np.isfinite(credit).all() or (credit < 0).any():
        raise ValueError('Exposure ranking requires finite, nonnegative credit amounts')
    rows=[]
    for strategy, rank in [('probability',p),('exposure',p*credit)]:
        order=np.lexsort((ids,-rank))
        for capacity in [0.1,0.2,0.3]:
            n=int(np.ceil(len(y)*capacity)); picked=order[:n]
            caught=y[picked].sum(); exposure=(y[picked]*credit[picked]).sum()
            total_exposure=(y*credit).sum()
            rows.append(dict(model=model,strategy=strategy,capacity=capacity,reviewed=n,
                cases_captured=int(caught),precision=caught/n,recall=caught/y.sum(),
                lift=(caught/n)/y.mean(), difficulty_credit_captured=exposure,
                difficulty_credit_coverage=exposure/total_exposure))
    return rows


def tune(kind, x, y, candidates, threads, label, out):
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    rows=[]
    for i, config in enumerate(candidates):
        for fold,(tr,va) in enumerate(cv.split(x,y),1):
            start=time.monotonic()
            m=make_model(kind,x,config,threads)
            m.fit(x.iloc[tr],y.iloc[tr])
            p=m.predict_proba(x.iloc[va])[:,1]
            rows.append(dict(model=label,candidate=i,fold=fold,config=str(config),
                             elapsed_seconds=time.monotonic()-start,**score(y.iloc[va],p)))
            print(f'{label} candidate {i+1}/{len(candidates)} fold {fold}/5 AP={rows[-1]["pr_auc_ap"]:.4f}',flush=True)
            pd.DataFrame(rows).to_csv(out/f'cv_{label}.csv',index=False)
    df=pd.DataFrame(rows)
    selected=int(df.groupby('candidate').pr_auc_ap.mean().idxmax())
    return candidates[selected],df


def fit_calibrator(y, p):
    # Sigmoid calibration on log-odds, using a separate labelled calibration partition.
    m=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000,random_state=SEED)
    m.fit(logit(p),y)
    if m.coef_[0,0] <= 0:
        raise ValueError('Calibration reversed risk ordering; investigate before deployment')
    return m


def logit(p):
    p=np.clip(np.asarray(p),1e-7,1-1e-7)
    return np.log(p/(1-p)).reshape(-1,1)


def calibrated(calibrator,p):
    return calibrator.predict_proba(logit(p))[:,1]
