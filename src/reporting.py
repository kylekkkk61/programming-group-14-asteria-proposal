"""Export reviewable evidence and figures for the business and reporting teams."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix
from sklearn.calibration import calibration_curve
from catboost import Pool


def save(fig,path):
    fig.tight_layout(); fig.savefig(path,dpi=160,bbox_inches='tight'); plt.close(fig)


def eda(x,y,quality,columns,out):
    quality.to_csv(out/'data_quality.csv',index=False)
    columns.to_csv(out/'raw_column_profile.csv',index=False)
    missing=x.isna().mean().sort_values(ascending=False)
    pd.DataFrame({'missing_fraction':missing,'distinct':x.nunique()}).to_csv(out/'feature_profile.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    y.value_counts().sort_index().plot.bar(ax=axes[0],color=['#315c78','#d89042'])
    axes[0].set(title=f'Development labels (difficulty rate {y.mean():.2%})',xlabel='TARGET',ylabel='Applicants')
    for value in [0,1]:
        axes[1].hist(x.loc[y==value,'AGE_YEARS'].dropna(),bins=25,density=True,alpha=.55,label=f'TARGET={value}')
    axes[1].set(xlabel='Age (years)',ylabel='Density',title='Age distribution by observed outcome'); axes[1].legend()
    save(fig,out/'figures'/'eda.png')
    fig,ax=plt.subplots(figsize=(10,6)); missing.head(15).sort_values().plot.barh(ax=ax,color='#315c78')
    ax.set(xlabel='Missing fraction',title='Features with the most missing values')
    save(fig,out/'figures'/'missingness.png')


def evaluation_figures(y,predictions,thresholds,out):
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for name,(raw,p) in predictions.items():
        pr,re,_=precision_recall_curve(y,p); axes[0].plot(re,pr,label=name)
        fpr,tpr,_=roc_curve(y,p); axes[1].plot(fpr,tpr,label=name)
        obs,est=calibration_curve(y,p,n_bins=10,strategy='quantile'); axes[2].plot(est,obs,'o-',label=name)
    axes[0].axhline(np.mean(y),ls='--',color='gray',label='Prevalence')
    axes[1].plot([0,1],[0,1],'--',color='gray');axes[2].plot([0,1],[0,1],'--',color='gray')
    for ax,title,xlabel,ylabel in zip(axes,['Holdout precision-recall','Holdout ROC','Holdout calibration'],
        ['Recall','False positive rate','Mean predicted probability'],['Precision','True positive rate','Observed frequency']):
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel);ax.legend(fontsize=8)
    save(fig,out/'figures'/'evaluation.png')
    raw,p=predictions['catboost_full']
    fig,ax=plt.subplots(figsize=(5,4)); cm=confusion_matrix(y,p>=thresholds['catboost_full'])
    ax.imshow(cm,cmap='Blues')
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha='center',va='center')
    ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted TARGET',ylabel='Actual TARGET',title='Full CatBoost: holdout confusion matrix')
    save(fig,out/'figures'/'confusion_matrix.png')
    fig,ax=plt.subplots(figsize=(5,4))
    for name,pp in [('Raw',raw),('Sigmoid calibrated',p)]:
        obs,est=calibration_curve(y,pp,n_bins=10,strategy='quantile');ax.plot(est,obs,'o-',label=name)
    ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='Predicted probability',ylabel='Observed frequency',title='Full CatBoost calibration');ax.legend()
    save(fig,out/'figures'/'calibration.png')


def explanations(model,x,out,threads):
    # Deterministic random sample controls the cost of exact Tree SHAP; no labels are used.
    sample=x.sample(n=min(500,len(x)),random_state=2026)
    shap=model.get_feature_importance(Pool(sample,cat_features=model.get_cat_feature_indices()),type='ShapValues',thread_count=threads)
    raw=model.predict(sample,prediction_type='RawFormulaVal')
    np.testing.assert_allclose(shap.sum(axis=1),raw,atol=1e-6)
    pd.DataFrame(shap[:,:-1],columns=x.columns,index=sample.index).to_parquet(out/'shap_values.parquet')
    importance=pd.Series(np.abs(shap[:,:-1]).mean(axis=0),index=x.columns).sort_values(ascending=False)
    importance.rename('mean_absolute_shap_log_odds').to_csv(out/'shap_importance.csv')
    fig,ax=plt.subplots(figsize=(11,7));importance.head(20).sort_values().plot.barh(ax=ax,color='#315c78')
    ax.set(xlabel='Mean absolute SHAP (raw model log-odds)',title=f'Full CatBoost: {len(sample)} holdout applicants')
    save(fig,out/'figures'/'shap_importance.png')
    top=importance.head(12).index
    fig,ax=plt.subplots(figsize=(11,6));rng=np.random.default_rng(2026)
    for i,c in enumerate(reversed(top)):
        v=shap[:,x.columns.get_loc(c)]; values=pd.to_numeric(sample[c],errors='coerce')
        color=values.rank(pct=True).fillna(.5)
        ax.scatter(v,i+rng.normal(0,.09,len(v)),c=color,cmap='coolwarm',s=8,alpha=.6,vmin=0,vmax=1)
    ax.set(yticks=range(len(top)),yticklabels=list(reversed(top)),xlabel='SHAP contribution to raw log-odds',title='SHAP distribution (numeric: blue = low, red = high; categorical/missing = grey)')
    ax.axvline(0,color='gray',lw=.8);save(fig,out/'figures'/'shap_distribution.png')
    sources=importance.groupby([c.split('__')[0].split('_')[0] if '__' in c else 'APPLICATION' for c in importance.index]).sum()
    sources.rename('sum_mean_absolute_shap').to_csv(out/'shap_by_source.csv')


def review_figure(reviews,out):
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for strategy,g in reviews.loc[reviews.model.eq('catboost_full')].groupby('strategy'):
        axes[0].plot(g.capacity,g.recall,'o-',label=strategy)
        axes[1].plot(g.capacity,g.difficulty_credit_coverage,'o-',label=strategy)
    axes[0].plot([.1,.2,.3],[.1,.2,.3],'--',label='Random expectation',color='gray')
    for ax in axes: ax.set(xlabel='Review capacity');ax.legend()
    axes[0].set(ylabel='Difficulty cases captured',title='Holdout case coverage')
    axes[1].set(ylabel='Difficulty-associated credit captured',title='Credit exposure coverage (not loss)')
    save(fig,out/'figures'/'review_capacity.png')


def write_results(out):
    """Produce an English handoff from measured outputs, never proposal estimates."""
    manifest=json.loads((out/'run_manifest.json').read_text())
    metrics=pd.read_csv(out/'metrics.csv')
    held=metrics.loc[metrics.phase.eq('holdout_calibrated_f2')].set_index('model')
    reviews=pd.read_csv(out/'review_strategies.csv')
    cv=pd.read_csv(out/'cross_validation.csv')
    shap=pd.read_csv(out/'shap_importance.csv',index_col=0).iloc[:,0]
    quality=pd.read_csv(out/'data_quality.csv')
    pred=pd.read_csv(out/'predictions.csv')
    def table(headers,rows):
        return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                         ['| '+' | '.join(map(str,row))+' |' for row in rows])
    scores=table(['Model','AP (PR-AUC)','ROC-AUC','Precision','Recall','F2','Brier','Threshold'],
        [[name]+[f'{r[c]:.4f}' for c in ['pr_auc_ap','roc_auc','precision','recall','f2','brier','threshold']] for name,r in held.iterrows()])
    cv_summary=cv.groupby(['model','candidate']).agg(ap_mean=('pr_auc_ap','mean'),ap_std=('pr_auc_ap','std'),folds=('fold','count'))
    cv_text=table(['Model','Candidate (0-based)','Mean AP','AP SD','Folds'],
        [[name,i,f'{r.ap_mean:.4f}',f'{r.ap_std:.4f}',int(r.folds)] for (name,i),r in cv_summary.iterrows()])
    full=held.loc['catboost_full']; lean=held.loc['catboost_lean']
    top=reviews.loc[reviews.model.eq('catboost_full') & reviews.strategy.eq('probability') & reviews.capacity.eq(.2)].iloc[0]
    strategies=table(['Strategy','Capacity','Reviewed','Cases found','Precision','Recall','Lift','Difficulty credit coverage'],
        [[r.strategy,f'{r.capacity:.0%}',int(r.reviewed),int(r.cases_captured),f'{r.precision:.2%}',f'{r.recall:.2%}',f'{r.lift:.2f}x',f'{r.difficulty_credit_coverage:.2%}']
         for r in reviews.loc[reviews.model.eq('catboost_full')].itertuples()])
    calibration=table(['Model','Raw Brier','Calibrated Brier'],[
        [name,f'{metrics.loc[(metrics.model==name)&(metrics.phase=="holdout_raw_0.5"),"brier"].iloc[0]:.5f}',f'{r.brier:.5f}']
        for name,r in held.iterrows()])
    missing=quality.dropna(subset=['missing_previous_parent_loans'])
    missing_text=table(['Split','Source','Distinct loans missing previous-application parent'],
        [[r.split,r.table,int(r.missing_previous_parent_loans)] for r in missing.itertuples()])
    text=f'''# Member A+B: measured technical handoff

This file reports the completed local run, not the exploratory scores quoted in the proposal. It is supporting material for Members C-G, not the group's final 1,000-word report.

## Business question and scope

Predict the dataset-defined repayment difficulty outcome and prioritise human review. The approved main model remains Full CatBoost, with L2 Logistic Regression as the benchmark and Lean CatBoost as a data-complexity comparison. The scores are decision support, not automatic loan rejection.

## Data engineering evidence

- Development applicants: {manifest['rows']['development']:,}; evaluation applicants: {manifest['rows']['evaluation']:,}; development difficulty rate: {manifest['positive_rate']:.2%}.
- Full feature count: {manifest['full_features']:,}; Lean feature count: {manifest['lean_features']:,}. Full uses all seven tables. Lean uses current applications, previous applications and instalments only; it still includes the application's external scores and bureau enquiry counts.
- All features preserve one row per applicant. Bureau history is aggregated to account level before applicant aggregation. No applicant IDs or TARGET are used as predictors.
- {len(manifest['excluded_from_models'])} personal attributes are kept for outcome monitoring only and are not used by any model: {', '.join(manifest['excluded_from_models'])}. Sex is excluded because it has no lawful basis as a credit-scoring input under the Equality Act 2010; age, family and occupation fields are excluded under data minimisation because they gave no consistent cross-validated gain. See `audit/AUDIT.md` for the tests behind this policy.
- Split instalment payments are consolidated using exact source-decimal arithmetic before calculating arrears; no currency rounding or arbitrary forgiveness threshold is applied. Undefined ratios remain missing; employment sentinel values become missing with an indicator. Histories without previous-application parent rows are retained via their own validated applicant ID.
- Three development bureau records with post-application update dates are excluded. Dates with uncertain realised-outcome availability are omitted or masked; future scheduled contractual dates are distinguished from realised future events.

{missing_text}

The raw dictionary and feature-level definitions are in `feature_dictionary.csv`; audit evidence is in `data_quality.csv`, `raw_column_profile.csv` and `feature_profile.csv`. The dictionary defines TARGET with unspecified X/Y timing parameters, so it must not be renamed as a specific 90-day default or a loss event.

## Validation and selection

A fixed stratified split keeps {manifest['rows']['fit']:,} applicants for training/tuning, {manifest['rows']['calibration']:,} for sigmoid calibration and F2 threshold selection, and {manifest['rows']['holdout']:,} for final testing. Five-fold stratified CV runs only inside the training partition. Median imputation, scaling and one-hot encoding are fitted inside each Logistic Regression training fold. CatBoost handles numeric missingness and categorical values directly.

Four Logistic Regression candidates vary C and class weighting. Four Full CatBoost candidates vary depth, leaf regularisation and class weighting, under a fixed 400-tree budget. Candidates are selected by mean CV Average Precision. Lean uses the selected Full settings to isolate the incremental value of the extra tables; it is not independently tuned. There is no holdout-based early stopping, threshold adjustment or model refitting after the final test. CV standard deviations below describe variation across five folds, not confidence intervals.

{cv_text}

## Independent holdout performance

{scores}

AP is the non-interpolated Average Precision implementation of the PR metric. Random-ranking expected precision is the positive prevalence. Thresholds maximise F2 on the calibration partition only. The same calibration data are used to fit the sigmoid and choose the threshold; metrics on that partition are therefore selection metrics, not independent evidence.

Full-minus-Lean holdout AP difference: **{full.pr_auc_ap-lean.pr_auc_ap:+.4f}**. This is a descriptive difference from one split, not a significance claim or proof that more data always helps. The final model is retained as specified by the approved proposal; report any observed trade-offs honestly.

## Probability calibration

{calibration}

Lower Brier is better. Calibration is assessed rather than assumed to help. See `figures/calibration.png` for reliability curves and `metrics.csv` for log loss and uncalibrated/0.5-threshold comparisons. Calibration uses an independent sigmoid on log-odds and is monotone for this run. The resulting SHAP explanations still refer to the underlying raw model, not the calibrated probability scale.

## Review capacity and exposure analysis

At 20% review capacity, probability-ranked Full CatBoost identifies **{int(top.cases_captured)} of {int(full.positives)} difficulty cases ({top.recall:.2%})** among {int(top.reviewed):,} reviewed holdout applicants. Review precision is **{top.precision:.2%}**, or **{top.lift:.2f} times** the holdout prevalence.

{strategies}

Exposure ranking uses probability times loan amount. Credit coverage counts loan amounts associated with observed difficulty cases. It is not realised financial loss, expected loss, LGD, savings, or ROI. The table exposes the trade-off between identifying more cases and covering more credit. No actual monetary benefit can be inferred without operating costs, intervention effects and loss data.

## Interpretation and evidence for Members C-F

The largest mean absolute SHAP contributors in the fixed 500-applicant holdout sample are:

'''
    text+='\n'.join(f'{i}. `{name}`: {value:.4f} mean absolute log-odds contribution.' for i,(name,value) in enumerate(shap.head(10).items(),1))
    text+=f'''

Use `figures/shap_importance.png` and `figures/shap_distribution.png` with `feature_dictionary.csv` to explain these terms. SHAP additivity is checked against the raw prediction. Contributions are associations conditional on this model, not causal evidence. Coefficients in `logistic_coefficients.csv` use standardised numeric and one-hot features; exponentiating a numeric coefficient describes a one-standard-deviation change, conditional on the other predictors.

`subgroup_diagnostics.csv` supplies sample sizes and outcome counts for gender and income-category diagnostics, and `audit/` adds age, family-status and education results, proxy-leakage tests and a counterfactual gender flip. Small groups are unstable, and these checks do not certify fairness. Proxies such as occupation remain under monitoring; any real deployment needs compliance review and ongoing outcome monitoring by group.

## Predictions and reproducibility

`predictions.csv` contains exactly {len(pred):,} template-ordered applicants and two columns, SK_ID_CURR and binary TARGET. Predicted class 1 count: {int(pred.TARGET.sum()):,}. These binary labels use the calibration-selected F2 threshold ({full.threshold:.6f}), not a forced 20% positive rate. `evaluation_risk_scores.csv` separately supplies calibrated probabilities, risk/exposure ranks and exact 10/20/30% review indicators.

The saved production prototype is the same Full CatBoost evaluated above; it is trained on the fit partition, not all development labels. Reproduction instructions are in `../README.md`. `run_manifest.json` records input hashes, seed, partitions and selected settings; `verification.json` records the independent reconstruction check when completed.

## Limitations and remaining group responsibilities

The evaluation set has no labels, so no evaluation-set accuracy is claimed. The holdout is one random stratified split rather than a future time period. Model, hyperparameter and feature choices were made by cross-validation or legal requirement; the holdout was used only to report results. Unique customer identity across applications is unavailable. Source snapshot timing is assumed from the supplied dictionary and relative dates; real deployment needs point-in-time auditing and temporal validation. Missingness may indicate incomplete coverage rather than no debt. Scheduled-payment versions are retained separately because the supplied data do not identify which overlapping contractual schedule supersedes another.

Members C/D should develop the business interpretation and responsible-use analysis from these measured results. Members E/F/G still own the final report, slide deck, recording, group review and submission. The official assessment requires disclosure of AI use; see `../AI_USAGE.md` and confirm intellectual ownership only after reviewing and understanding the work.
'''
    (out/'RESULTS.md').write_text(text)
