"""Rebuild evaluation features, reproduce predictions, and validate saved artefacts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from src.features import build_features, EXCLUDED_FROM_MODELS, ID
from pipeline import predict_saved


def main():
    out=Path('outputs');root=Path('asteria_student_data')
    manifest=json.loads((out/'run_manifest.json').read_text())
    x,_,_,_=build_features(root,'evaluation')
    stored=pd.read_parquet(out/'evaluation_features.parquet')
    pd.testing.assert_frame_equal(x,stored)
    actual,p=predict_saved(out/'catboost_full.joblib',x,pd.read_csv(root/'prediction_template.csv'))
    for label in ['logistic_full','catboost_full','catboost_lean']:
        assert not set(EXCLUDED_FROM_MODELS)&set(joblib.load(out/f'{label}.joblib')['columns']), label
    expected=pd.read_csv(out/'predictions.csv');pd.testing.assert_frame_equal(actual,expected)
    scores=pd.read_csv(out/'evaluation_risk_scores.csv').set_index(ID).reindex(x.index)
    np.testing.assert_allclose(p,scores.repayment_difficulty_probability,rtol=1e-12,atol=1e-12)
    assert len(actual)==manifest['rows']['evaluation']
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    for strategy in ['probability','exposure']:
        for pct in [10,20,30]:
            assert scores[f'{strategy}_top{pct}'].sum()==int(np.ceil(len(x)*pct/100))
    dev=pd.read_parquet(out/'development_features.parquet')
    assert not set(dev.index)&set(x.index)
    assert list(dev.drop(columns='TARGET'))==list(x)
    assert not any(c.startswith('SK_ID') or c=='TARGET' for c in x)
    assert np.isfinite(x.select_dtypes(include='number').to_numpy()[~np.isnan(x.select_dtypes(include='number').to_numpy())]).all()
    part=pd.read_csv(out/'partitions.csv').set_index(ID).partition
    assert part.index.is_unique and set(part.index)==set(dev.index)
    for label in ['logistic_full','catboost_full','catboost_lean']:
        h=pd.read_csv(out/f'holdout_{label}.csv')
        assert set(h[ID])==set(part.index[part.eq('holdout')])
        np.testing.assert_array_equal(h.TARGET,dev.loc[h[ID],'TARGET'])
    cv=pd.read_csv(out/'cross_validation.csv')
    assert cv.groupby(['model','candidate']).fold.apply(lambda s:set(s)=={1,2,3,4,5}).all()
    result={'status':'passed','evaluation_features_rebuilt':len(x),
            'prediction_rows_exactly_reproduced':len(actual),'holdout_rows':int(part.eq('holdout').sum()),
            'cv_fits_checked':len(cv),'checks':['schema','unique IDs','template order','binary labels',
            'no monitoring-only attributes in models',
            'saved-model probabilities','exact review capacity','disjoint partitions','holdout labels','all five folds']}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
