"""Independent exact-rational oracle for every supplied due instalment.

Run from the project root with: python -m tests.check_instalment_precision
This check reads original decimal strings and never changes source data.
"""
from fractions import Fraction
from itertools import groupby
import json
import math
from pathlib import Path
import pandas as pd
from src.features import instalments


def check(split):
    path=Path('asteria_student_data') / f'{split} dataset' / 'instalment_payments.csv'
    d=pd.read_csv(path,dtype={'AMT_INSTALMENT':'str','AMT_PAYMENT':'str'}).drop_duplicates()
    actual=instalments(d)
    keys=['SK_ID_CURR','SK_ID_PREV','NUM_INSTALMENT_VERSION','NUM_INSTALMENT_NUMBER','DAYS_INSTALMENT']
    ordered=d.loc[d.DAYS_INSTALMENT.le(0)].sort_values(keys+['DAYS_ENTRY_PAYMENT'])
    cols=keys+['AMT_INSTALMENT','DAYS_ENTRY_PAYMENT','AMT_PAYMENT']
    groups=groupby(ordered[cols].itertuples(index=False,name=None),key=lambda row:row[:5])
    checked=exactly_settled=0
    for (key,rows),result in zip(groups,actual.itertuples(index=False),strict=True):
        assert key == tuple(getattr(result,k) for k in keys)
        amount=None;paid=Fraction(0);on_time=Fraction(0);completion=None;unknown=False
        for row in rows:
            if pd.notna(row[5]): amount=Fraction(row[5])
            day,payment=row[6:]
            if pd.isna(day) or (day<=0 and pd.isna(payment)):
                unknown=True
            if pd.notna(day) and day<=0 and pd.notna(payment):
                paid+=Fraction(payment)
                if day<=key[4]: on_time+=Fraction(payment)
                if amount is not None and paid>=amount and completion is None: completion=day
        if amount is not None and amount>0 and not unknown:
            expected_shortfall=float(max(amount-paid,Fraction(0)))
            expected_late=float(on_time<amount)
            expected_delay=max((completion if completion is not None else 0)-key[4],0)
            assert result.shortfall==expected_shortfall,(split,key,'shortfall',result.shortfall,expected_shortfall)
            assert result.late==expected_late,(split,key,'late',result.late,expected_late)
            assert result.delay_days==expected_delay,(split,key,'delay',result.delay_days,expected_delay)
            exactly_settled+=paid==amount
        else:
            assert all(math.isnan(getattr(result,c)) for c in ['shortfall','late','delay_days'])
        checked+=1
    assert checked==len(actual)
    return dict(split=split,checked_instalments=checked,exactly_settled_instalments=exactly_settled,
                mismatches=0,oracle='fractions.Fraction from original CSV strings')


if __name__=='__main__':
    print(json.dumps({'status':'passed','results':[check(s) for s in ['development','evaluation']]},indent=2))
