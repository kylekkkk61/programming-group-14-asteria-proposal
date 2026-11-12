#!/usr/bin/env python3
"""Reproduce Member A+B deliverables directly from the supplied seven-table extracts."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.features import (build_features, lean_columns, model_columns, require_unique,
                          feature_documentation, EXCLUDED_FROM_MODELS, ID)
from src.modelling import (SEED, prepare, tune, make_model, score, best_threshold,
                           fit_calibrator, calibrated, top_reviews)
from src.reporting import eda, evaluation_figures, explanations, review_figure, write_results


def json_write(path, obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')


def predict_saved(bundle_path, features, template):
    bundle=joblib.load(bundle_path)
    x=prepare(features[bundle['columns']],bundle['categories'])
    p=calibrated(bundle['calibrator'],bundle['model'].predict_proba(x)[:,1])
    result=pd.DataFrame({ID:features.index,'TARGET':(p>=bundle['threshold']).astype(int)})
    return align_predictions(result,template), p


def align_predictions(result, template):
    require_unique(result,[ID],'predictions');require_unique(template,[ID],'template')
    if set(result[ID]) != set(template[ID]):
        raise ValueError('Prediction IDs do not exactly match template IDs')
    result=template[[ID]].merge(result,on=ID,how='left',validate='one_to_one')
    if list(result.columns) != [ID,'TARGET'] or not result.TARGET.isin([0,1]).all():
        raise ValueError('Predictions must have exactly SK_ID_CURR and binary TARGET')
    return result


def run(args):
    start=time.monotonic();out=Path(args.output);root=Path(args.data)
    out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    print('Building development and evaluation features',flush=True)
    x,y,q,c=build_features(root,'development');ev,_,qe,ce=build_features(root,'evaluation')
    if set(x.index)&set(ev.index): raise ValueError('Development/evaluation applicants overlap')
    if not x.columns.equals(ev.columns): raise ValueError('Feature schemas differ')
    if not y.isin([0,1]).all(): raise ValueError('Invalid labels')
    x.assign(TARGET=y).to_parquet(out/'development_features.parquet')
    ev.to_parquet(out/'evaluation_features.parquet')
    eda(x,y,pd.concat([q,qe]),pd.concat([c,ce]),out)
    dictionary=pd.read_csv(root/'data_dictionary.csv')
    feature_documentation(x.columns,dictionary).to_csv(out/'feature_dictionary.csv',index=False)
    # Locked random partitions: 60% fit/tune, 20% calibration/threshold, 20% final holdout.
    fit_cal,hold=train_test_split(np.arange(len(x)),test_size=.2,stratify=y,random_state=SEED)
    fit,cal=train_test_split(fit_cal,test_size=.25,stratify=y.iloc[fit_cal],random_state=SEED)
    partitions=pd.Series(index=x.index,dtype='str',name='partition')
    for name,idx in [('fit',fit),('calibration',cal),('holdout',hold)]: partitions.iloc[idx]=name
    partitions.to_csv(out/'partitions.csv')
    cats=list(x.select_dtypes(exclude='number').columns)
    x=prepare(x,cats);ev=prepare(ev,cats)
    # Monitoring-only protected attributes stay in x for subgroup reporting but never enter a model.
    full_cols=model_columns(x.columns);lean=lean_columns(full_cols)
    candidates=[dict(depth=d,l2_leaf_reg=l2,iterations=args.iterations,learning_rate=.06,auto_class_weights=w)
                for d,l2 in [(4,5),(6,8)] for w in ['None','SqrtBalanced']]
    lr_candidates=[dict(C=C,class_weight=w) for C in [.01,.1] for w in [None,'balanced']]
    all_cv=[];metrics=[];review=[];predictions={};thresholds={};choices={}
    models=[('logistic_full','logistic',full_cols,lr_candidates),('catboost_full','catboost',full_cols,candidates)]
    for label,kind,cols,grid in models:
        config,cv=tune(kind,x.iloc[fit][cols],y.iloc[fit],grid,args.threads,label,out)
        all_cv.append(cv);choices[label]=config
    # Same training budget and parameters isolate the contribution of the extra tables.
    _,lean_cv=tune('catboost',x.iloc[fit][lean],y.iloc[fit],[choices['catboost_full']],args.threads,'catboost_lean',out)
    all_cv.append(lean_cv);choices['catboost_lean']=choices['catboost_full']
    models.append(('catboost_lean','catboost',lean,[]))
    pd.concat(all_cv).to_csv(out/'cross_validation.csv',index=False)
    json_write(out/'selected_parameters.json',choices)
    for label,kind,cols,_ in models:
        print(f'Fitting and calibrating {label}',flush=True)
        model=make_model(kind,x[cols],choices[label],args.threads)
        model.fit(x.iloc[fit][cols],y.iloc[fit])
        cal_raw=model.predict_proba(x.iloc[cal][cols])[:,1]
        calibrator=fit_calibrator(y.iloc[cal],cal_raw)
        cal_p=calibrated(calibrator,cal_raw)
        threshold=best_threshold(y.iloc[cal],cal_p);thresholds[label]=threshold
        raw=model.predict_proba(x.iloc[hold][cols])[:,1];p=calibrated(calibrator,raw)
        predictions[label]=(raw,p)
        for phase,pp,yy,thr in [('holdout_raw_0.5',raw,y.iloc[hold],.5),
            ('holdout_calibrated_0.5',p,y.iloc[hold],.5),('holdout_calibrated_f2',p,y.iloc[hold],threshold),
            ('calibration_threshold_selection',cal_p,y.iloc[cal],threshold)]:
            metrics.append(dict(model=label,phase=phase,**score(yy,pp,thr)))
        review.extend(top_reviews(x.index[hold],y.iloc[hold],p,x.iloc[hold].AMT_CREDIT,label))
        pd.DataFrame({ID:x.index[hold],'TARGET':y.iloc[hold].values,'raw_probability':raw,
            'calibrated_probability':p,'predicted_class':(p>=threshold).astype(int)}).to_csv(out/f'holdout_{label}.csv',index=False)
        joblib.dump(dict(model=model,calibrator=calibrator,threshold=threshold,columns=cols,
            categories=[c for c in cats if c in cols]),out/f'{label}.joblib')
        if label=='catboost_full':
            model.save_model(str(out/'catboost_full.cbm'))
            print('Computing exact Tree SHAP explanations',flush=True)
            explanations(model,x.iloc[hold][cols],out,args.threads)
        if kind=='logistic':
            pre=model.named_steps['preprocess'];coef=model.named_steps['classifier'].coef_[0]
            pd.DataFrame({'feature':pre.get_feature_names_out(),'coefficient':coef}).sort_values('coefficient').to_csv(out/'logistic_coefficients.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    reviews=pd.DataFrame(review);reviews.to_csv(out/'review_strategies.csv',index=False)
    evaluation_figures(y.iloc[hold],predictions,thresholds,out);review_figure(reviews,out)
    template=pd.read_csv(root/'prediction_template.csv')
    result,p=predict_saved(out/'catboost_full.joblib',ev,template)
    result.to_csv(out/'predictions.csv',index=False)
    # Independent re-load checks the submitted CSV contract, not just the in-memory frame.
    pd.testing.assert_frame_equal(pd.read_csv(out/'predictions.csv'),result)
    risk=pd.DataFrame({ID:ev.index,'repayment_difficulty_probability':p,'AMT_CREDIT':ev.AMT_CREDIT.to_numpy()})
    risk['exposure_score']=risk.repayment_difficulty_probability*risk.AMT_CREDIT
    for strategy,col in [('probability','repayment_difficulty_probability'),('exposure','exposure_score')]:
        order=risk.sort_values([col,ID],ascending=[False,True]).index
        rank=pd.Series(np.arange(1,len(risk)+1),index=order).reindex(risk.index)
        risk[f'{strategy}_rank']=rank
        for pct in [10,20,30]: risk[f'{strategy}_top{pct}']=rank.le(int(np.ceil(len(risk)*pct/100))).astype(int)
    template[[ID]].merge(risk,on=ID,validate='one_to_one').to_csv(out/'evaluation_risk_scores.csv',index=False)
    # Descriptive subgroup results assist Member D; this is not a fairness certification.
    full_p=predictions['catboost_full'][1];groups=[]
    for col in ['CODE_GENDER','NAME_INCOME_TYPE']:
        vals=x.iloc[hold][col].to_numpy()
        for v in sorted(set(vals)):
            ix=np.flatnonzero(vals==v);yy=y.iloc[hold].iloc[ix]
            if yy.nunique()==2:
                groups.append(dict(group_column=col,group=v,**score(yy,full_p[ix],thresholds['catboost_full'])))
    pd.DataFrame(groups).to_csv(out/'subgroup_diagnostics.csv',index=False)
    files=sorted(root.rglob('*.csv'))
    json_write(out/'run_manifest.json',dict(seed=SEED,python=platform.python_version(),
        elapsed_seconds=time.monotonic()-start,arguments=vars(args),
        rows={'development':len(x),'evaluation':len(ev),'fit':len(fit),'calibration':len(cal),'holdout':len(hold)},
        full_features=len(full_cols),lean_features=len(lean),positive_rate=float(y.mean()),
        excluded_from_models=EXCLUDED_FROM_MODELS,
        final_model='catboost_full',refit_on_holdout=False,thresholds=thresholds,
        input_sha256={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}))
    write_results(out)
    print(pd.DataFrame(metrics).query("phase == 'holdout_calibrated_f2'").to_string(index=False),flush=True)
    print(f'Completed: {out / "predictions.csv"}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='asteria_student_data')
    parser.add_argument('--output',default='outputs')
    parser.add_argument('--iterations',type=int,default=400,help='Fixed tree budget for each CV candidate (no holdout early stopping)')
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.iterations<1 or args.threads<1: parser.error('iterations and threads must be positive')
    run(args)
