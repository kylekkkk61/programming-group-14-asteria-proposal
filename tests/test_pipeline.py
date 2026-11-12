"""Regression checks for the consequential data and prediction contracts."""
import unittest
import numpy as np
import pandas as pd
from src.features import instalments, ratio, windows, require_unique, lean_columns, model_columns
from src.modelling import best_threshold, top_reviews, make_model
from pipeline import align_predictions
from sklearn.metrics import fbeta_score


class FeatureTests(unittest.TestCase):
    def payments(self, rows):
        return pd.DataFrame(rows,columns=['SK_ID_CURR','SK_ID_PREV','NUM_INSTALMENT_VERSION',
            'NUM_INSTALMENT_NUMBER','DAYS_INSTALMENT','AMT_INSTALMENT','DAYS_ENTRY_PAYMENT','AMT_PAYMENT'])

    def test_split_payments_are_one_instalment(self):
        d=self.payments([[1,10,1,1,-10,100,-12,40],[1,10,1,1,-10,100,-8,60]])
        x=instalments(d)
        self.assertEqual(len(x),1);self.assertEqual(x.late.iloc[0],1)
        self.assertEqual(x.delay_days.iloc[0],2);self.assertEqual(x.shortfall.iloc[0],0)
        self.assertEqual(x.payment_ratio.iloc[0],1)

    def test_future_payment_is_not_available_and_delay_is_censored(self):
        d=self.payments([[1,10,1,1,-10,100,-12,40],[1,10,1,1,-10,100,8,60],
                         [1,10,1,2,10,100,15,100]])
        x=instalments(d)
        self.assertEqual(len(x),1);self.assertEqual(x.shortfall.iloc[0],60)
        self.assertEqual(x.delay_days.iloc[0],10)
        d.loc[1,'AMT_PAYMENT']=600000
        pd.testing.assert_frame_equal(x,instalments(d))

    def test_paid_on_time_not_marked_late(self):
        x=instalments(self.payments([[1,10,1,1,-10,100,-12,40],[1,10,1,1,-10,100,-10,60]]))
        self.assertEqual(x.late.iloc[0],0);self.assertEqual(x.delay_days.iloc[0],0)

    def test_decimal_split_payment_is_paid_on_due_date(self):
        d=self.payments([[104276,2566955,1,5,-1024,6418.305,-1035,89.73],
                         [104276,2566955,1,5,-1024,6418.305,-1024,6328.575]])
        x=instalments(d)
        self.assertEqual(x.shortfall.iloc[0],0)
        self.assertEqual(x.late.iloc[0],0)
        self.assertEqual(x.delay_days.iloc[0],0)
        self.assertEqual(x.payment_ratio.iloc[0],1)

    def test_real_small_shortfall_is_not_forgiven(self):
        d=self.payments([[1,10,1,1,-10,'1.000000000001',-10,'1.0']])
        x=instalments(d)
        self.assertAlmostEqual(x.shortfall.iloc[0],1e-12,places=24)
        self.assertEqual(x.late.iloc[0],1)
        self.assertEqual(x.delay_days.iloc[0],10)

    def test_decimal_late_completion_uses_actual_day(self):
        d=self.payments([[1,10,1,1,-10,6418.305,-12,89.73],
                         [1,10,1,1,-10,6418.305,-8,6328.575]])
        x=instalments(d)
        self.assertEqual(x.shortfall.iloc[0],0)
        self.assertEqual(x.late.iloc[0],1)
        self.assertEqual(x.delay_days.iloc[0],2)

    def test_conflicting_schedule_fails(self):
        d=self.payments([[1,10,1,1,-10,100,-12,40],[1,10,1,1,-10,110,-8,60]])
        with self.assertRaisesRegex(ValueError,'Conflicting'): instalments(d)

    def test_unknown_scheduled_amount_does_not_become_good_payment(self):
        x=instalments(self.payments([[1,10,1,1,-10,np.nan,-12,40]]))
        self.assertTrue(np.isnan(x.late.iloc[0]));self.assertTrue(np.isnan(x.payment_ratio.iloc[0]))

    def test_unknown_payment_is_not_assumed_unpaid(self):
        x=instalments(self.payments([[1,10,1,1,-10,100,np.nan,np.nan]]))
        self.assertEqual(x.payment_unknown.iloc[0],1)
        self.assertTrue(x[['shortfall','payment_ratio','late','delay_days']].isna().all().all())

    def test_time_windows_and_nonoverlapping_change(self):
        d=pd.DataFrame({'SK_ID_CURR':[1]*4,'MONTHS_BALANCE':[-12,-7,-6,-1], 'late':[0.,0.,1.,1.]})
        x=windows(d,'POS','MONTHS_BALANCE',[('6m',6),('12m',12)])
        self.assertEqual(x.POS_6m__row_count.iloc[0],2)
        self.assertEqual(x.POS_12m__row_count.iloc[0],4)
        self.assertEqual(x.POS__late__recent_minus_prior6m.iloc[0],1)

    def test_ratio_does_not_create_infinity(self):
        x=ratio(pd.Series([10.,10.,10.]),pd.Series([0.,-1.,2.]))
        self.assertTrue(x.iloc[:2].isna().all());self.assertEqual(x.iloc[2],5)

    def test_duplicate_key_rejected(self):
        with self.assertRaises(ValueError): require_unique(pd.DataFrame({'id':[1,1]}),['id'],'test')

    def test_lean_does_not_include_external_information(self):
        self.assertEqual(lean_columns(['AMT_CREDIT','PREV__x','INST__x','EXTERNAL_DEBT_TO_INCOME','BUREAU__x','CARD__x','POS__x']),['AMT_CREDIT','PREV__x','INST__x'])

    def test_monitoring_only_attributes_never_reach_a_model(self):
        cols=['AMT_CREDIT','CODE_GENDER','DAYS_BIRTH','OCCUPATION_TYPE','DAYS_EMPLOYED','PREV__x','BUREAU__x']
        self.assertEqual(model_columns(cols),['AMT_CREDIT','DAYS_EMPLOYED','PREV__x','BUREAU__x'])
        self.assertEqual(lean_columns(model_columns(cols)),['AMT_CREDIT','DAYS_EMPLOYED','PREV__x'])


class ModelTests(unittest.TestCase):
    def test_threshold_really_maximises_f2(self):
        y=np.array([0,1,0,1,0]);p=np.array([.05,.1,.2,.3,.9])
        t=best_threshold(y,p)
        self.assertEqual(fbeta_score(y,p>=t,beta=2),max(fbeta_score(y,p>=v,beta=2) for v in p))

    def test_review_ties_have_exact_capacity_and_stable_ids(self):
        r=top_reviews(np.array([3,1,2,4,5,6,7,8,9,10]),np.array([0,1,1,0,0,0,0,0,0,0]),np.ones(10),np.ones(10),'test')
        self.assertEqual(r[0]['reviewed'],1);self.assertEqual(r[0]['cases_captured'],1)
        self.assertEqual(r[1]['reviewed'],2);self.assertEqual(r[1]['recall'],1)

    def test_imputer_fits_training_only_and_unknown_categories_work(self):
        x=pd.DataFrame({'number':[1.,np.nan,3.,5.],'cat':['a','a','b','b']})
        model=make_model('logistic',x,{'C':1},1).fit(x,[0,1,0,1])
        numeric=model.named_steps['preprocess'].named_transformers_['numeric']
        self.assertEqual(numeric.named_steps['impute'].statistics_[0],3)
        p=model.predict_proba(pd.DataFrame({'number':[1000000.,np.nan],'cat':['unseen','unseen']}))
        self.assertTrue(np.isfinite(p).all())
        self.assertEqual(numeric.named_steps['impute'].statistics_[0],3)

    def test_predictions_follow_template_order_and_reject_missing_ids(self):
        p=pd.DataFrame({'SK_ID_CURR':[2,1],'TARGET':[0,1]})
        t=pd.DataFrame({'SK_ID_CURR':[1,2],'TARGET':[np.nan,np.nan]})
        r=align_predictions(p,t)
        self.assertEqual(r.SK_ID_CURR.tolist(),[1,2]);self.assertEqual(r.TARGET.tolist(),[1,0])
        with self.assertRaises(ValueError): align_predictions(p.iloc[:1],t)
        p['TARGET']=p.TARGET.astype(float)
        p.loc[0,'TARGET']=.2
        with self.assertRaises(ValueError): align_predictions(p,t)


if __name__=='__main__': unittest.main()
