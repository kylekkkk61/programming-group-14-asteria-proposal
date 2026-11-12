"""Applicant-level, as-of-application features; no fitted population statistics."""
from pathlib import Path
from decimal import Decimal
import numpy as np
import pandas as pd

ID = 'SK_ID_CURR'
# Personal attributes kept in the feature tables for outcome monitoring, never used by a model.
# Sex: no lawful basis as a credit-scoring input (Equality Act 2010 s.13 and s.29).
# Age, family and occupation: data minimisation (UK GDPR art. 5(1)(c)); audit.py found no
# consistent cross-validated gain from any of these groups, while occupation is a strong sex proxy.
EXCLUDED_FROM_MODELS = {
    'CODE_GENDER': 'sex (protected characteristic)',
    'DAYS_BIRTH': 'age: no consistent CV gain', 'AGE_YEARS': 'age: no consistent CV gain',
    'NAME_FAMILY_STATUS': 'family: no consistent CV gain', 'CNT_CHILDREN': 'family: no consistent CV gain',
    'CNT_FAM_MEMBERS': 'family: no consistent CV gain', 'INCOME_PER_PERSON': 'family: no consistent CV gain',
    'NAME_TYPE_SUITE': 'family: no consistent CV gain',
    'OCCUPATION_TYPE': 'occupation: sex proxy, no consistent CV gain',
    'ORGANIZATION_TYPE': 'occupation: sex proxy, no consistent CV gain',
}
# Pregnancy and maternity is protected in services (s.17); no score may depend on it.
PROTECTED_INCOME_TYPES = ['Maternity leave']
TABLES = ['previous_loan_applications', 'instalment_payments', 'external_credit_accounts',
          'external_account_history', 'pos_cash_balance', 'credit_card_balance']


def ratio(a, b):
    return a.div(b.where(b > 0)).replace([np.inf, -np.inf], np.nan)


def require_unique(df, keys, name):
    if df[keys].isna().any().any() or df.duplicated(keys).any():
        raise ValueError(f'{name}: missing or duplicate keys {keys}')


def aggregate(df, prefix, key=ID):
    """Numerical summaries only; identifiers never become model features."""
    numeric = [c for c in df.select_dtypes(include='number') if not c.startswith('SK_ID')]
    g = df.groupby(key, sort=True)
    out = g[numeric].agg(['mean', 'max', 'min'])
    out.columns = [f'{prefix}__{c}__{stat}' for c, stat in out.columns]
    out[f'{prefix}__row_count'] = g.size()
    if 'SK_ID_PREV' in df:
        out[f'{prefix}__loan_count'] = g.SK_ID_PREV.nunique()
    return out


def windows(df, prefix, time, spans, key=ID):
    result = aggregate(df, prefix, key)
    for label, length in spans:
        recent = df.loc[df[time].between(-length, 0, inclusive='both')]
        result = result.join(aggregate(recent, f'{prefix}_{label}', key), how='left', validate='one_to_one')
    # Compare the latest six months with the preceding six months, not overlapping averages.
    if len(spans) == 2:
        recent = df.loc[df[time].between(-spans[0][1], 0, inclusive='both')]
        older = df.loc[(df[time] >= -spans[1][1]) & (df[time] < -spans[0][1])]
        cols = [c for c in ['late', 'shortfall', 'utilisation', 'SK_DPD', 'delinquent'] if c in df]
        for c in cols:
            result[f'{prefix}__{c}__recent_minus_prior6m'] = recent.groupby(key)[c].mean() - older.groupby(key)[c].mean()
    return result


def instalments(raw):
    """Consolidate split payments before measuring arrears at the observation date."""
    keys = [ID, 'SK_ID_PREV', 'NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_NUMBER', 'DAYS_INSTALMENT']
    d = raw.loc[raw.DAYS_INSTALMENT.le(0)].copy()
    # Preserve source decimal amounts until all payment decisions have been made.
    # Do not round to cents: the supplied extract contains fractional-cent amounts.
    for col in ['AMT_INSTALMENT', 'AMT_PAYMENT']:
        d[col] = d[col].map(lambda value: Decimal(str(value)) if pd.notna(value) else np.nan).astype(object)
    amounts = d.groupby(keys, dropna=False).AMT_INSTALMENT.nunique()
    if (amounts > 1).any():
        raise ValueError('Conflicting scheduled amounts for the same instalment')
    d['payment_unknown'] = d.DAYS_ENTRY_PAYMENT.isna() | (d.DAYS_ENTRY_PAYMENT.le(0) & d.AMT_PAYMENT.isna())
    d['observed'] = d.DAYS_ENTRY_PAYMENT.le(0) & d.AMT_PAYMENT.notna()
    d['paid'] = d.AMT_PAYMENT.where(d.observed, Decimal(0))
    d['on_time_paid'] = d.paid.where(d.DAYS_ENTRY_PAYMENT.le(d.DAYS_INSTALMENT), Decimal(0))
    d = d.sort_values(keys + ['DAYS_ENTRY_PAYMENT'], na_position='last')
    # Pandas grouped cumsum does not support Decimal; scan already-sorted groups.
    cumulative = []
    previous_group = None
    total = Decimal(0)
    for group, paid in zip(d.groupby(keys, dropna=False).ngroup(), d.paid):
        if group != previous_group:
            total = Decimal(0)
            previous_group = group
        total += paid
        cumulative.append(total)
    d['cumulative'] = pd.Series(cumulative, index=d.index, dtype=object)
    completed = d.AMT_INSTALMENT.notna() & d.cumulative.ge(d.AMT_INSTALMENT.fillna(Decimal(0)))
    d['completion_day'] = d.DAYS_ENTRY_PAYMENT.where(d.observed & completed)
    out = d.groupby(keys, dropna=False).agg(
        amount=('AMT_INSTALMENT', 'first'), paid=('paid', 'sum'),
        on_time_paid=('on_time_paid', 'sum'), completion_day=('completion_day', 'min'),
        observed_payments=('observed', 'sum'), payment_unknown=('payment_unknown', 'max')).reset_index()
    out['payment_unknown'] = out.payment_unknown.astype(int)
    out.loc[out.payment_unknown.eq(1), ['paid', 'on_time_paid']] = np.nan
    valid = out.amount.gt(0) & out.payment_unknown.eq(0)
    out['shortfall'] = (out.amount - out.paid).map(
        lambda value: max(value, Decimal(0)) if pd.notna(value) else np.nan).where(valid)
    late = out.on_time_paid.fillna(Decimal(0)).lt(
        out.amount.fillna(Decimal(0))).astype(float).where(valid)
    # Numerical model features may be floats after exact settlement classification.
    for col in ['amount', 'paid', 'on_time_paid', 'shortfall']:
        out[col] = out[col].astype(float)
    out['payment_ratio'] = ratio(out.paid, out.amount)
    out['late'] = late
    # An unfinished instalment's delay is censored at the current application date (day 0).
    out['delay_days'] = (out.completion_day.fillna(0) - out.DAYS_INSTALMENT).clip(lower=0).where(valid)
    return out.drop(columns=['completion_day'])


def build_features(root, split):
    folder = Path(root) / f'{split} dataset'
    quality, columns = [], []
    def read(name):
        money_types = {'AMT_INSTALMENT': 'str', 'AMT_PAYMENT': 'str'} if name == 'instalment_payments' else None
        df = pd.read_csv(folder / f'{name}.csv', dtype=money_types)
        quality.append({'split': split, 'table': name, 'raw_rows': len(df),
                        'exact_duplicate_rows': int(df.duplicated().sum())})
        for c in df:
            columns.append({'split': split, 'table': name, 'column': c,
                            'dtype': str(df[c].dtype), 'missing_fraction': df[c].isna().mean(),
                            'distinct': df[c].nunique()})
        # Exact repeated exports are removed; legitimate split payments with different values remain.
        return df.drop_duplicates().copy()
    app = read('loan_applications' if split == 'development' else 'new_loan_applications')
    require_unique(app, [ID], 'applications')
    y = app.pop('TARGET') if 'TARGET' in app else None
    app = app.set_index(ID)
    tables = {name: read(name) for name in TABLES}
    for name, d in tables.items():
        if ID in d:
            if d[ID].isna().any() or not d[ID].isin(app.index).all():
                raise ValueError(f'{name}: orphan applicant IDs')
    prev, bureau = tables['previous_loan_applications'], tables['external_credit_accounts']
    require_unique(prev, ['SK_ID_PREV'], 'previous applications')
    require_unique(bureau, ['SK_ID_BUREAU'], 'external accounts')
    # Cross-check ownership where a previous application is present. Some extracts omit parent rows.
    for name in ['instalment_payments', 'pos_cash_balance', 'credit_card_balance']:
        d = tables[name]
        owner = d[['SK_ID_PREV', ID]].drop_duplicates().merge(
            prev[['SK_ID_PREV', ID]], on='SK_ID_PREV', how='left', suffixes=('', '_parent'), validate='many_to_one')
        if (owner[ID+'_parent'].notna() & owner[ID].ne(owner[ID+'_parent'])).any():
            raise ValueError(f'{name}: inconsistent previous-loan ownership')
        require_unique(d[['SK_ID_PREV', ID]].drop_duplicates(), ['SK_ID_PREV'], name+' ownership')
        quality.append({'split': split, 'table': name, 'missing_previous_parent_loans': int(owner[ID+'_parent'].isna().sum())})
    app['EMPLOYMENT_SENTINEL'] = app.DAYS_EMPLOYED.eq(365243).astype(int)
    for c in app.select_dtypes(include='number'):
        if c.startswith('DAYS_'):
            app[c] = app[c].mask(app[c].eq(365243))
    app['AGE_YEARS'] = -app.DAYS_BIRTH / 365.25
    app['EMPLOYMENT_YEARS'] = -app.DAYS_EMPLOYED.where(app.DAYS_EMPLOYED.le(0)) / 365.25
    for name, a, b in [('CREDIT_TO_INCOME','AMT_CREDIT','AMT_INCOME_TOTAL'),
                       ('ANNUITY_TO_INCOME','AMT_ANNUITY','AMT_INCOME_TOTAL'),
                       ('CREDIT_TO_GOODS','AMT_CREDIT','AMT_GOODS_PRICE'),
                       ('INCOME_PER_PERSON','AMT_INCOME_TOTAL','CNT_FAM_MEMBERS')]:
        app[name] = ratio(app[a], app[b])
    app = app.replace({'XNA': np.nan, 'XAP': np.nan})
    maternity = app.NAME_INCOME_TYPE.isin(PROTECTED_INCOME_TYPES)
    quality.append({'split': split, 'table': 'loan_applications', 'protected_income_type_masked': int(maternity.sum())})
    app.loc[maternity, 'NAME_INCOME_TYPE'] = np.nan
    parts = []
    def keep_asof(d, col, name):
        valid = d[col].le(0)
        quality.append({'split': split, 'table': name, 'unavailable_time_rows_excluded': int((~valid).sum())})
        return d.loc[valid].copy()
    prev = keep_asof(prev, 'DAYS_DECISION', 'previous_loan_applications')
    for c in prev:
        if c.startswith('DAYS_'):
            prev[c] = prev[c].mask(prev[c].eq(365243))
    # Ambiguous realised lifecycle dates are omitted; future outcomes must not be predictors.
    prev = prev.drop(columns=['DAYS_LAST_DUE', 'DAYS_TERMINATION'])
    prev['DAYS_FIRST_DRAWING'] = prev.DAYS_FIRST_DRAWING.where(prev.DAYS_FIRST_DRAWING.le(0))
    for status in ['Approved', 'Refused', 'Canceled', 'Unused offer']:
        prev[f'status_{status.replace(" ", "_")}'] = prev.NAME_CONTRACT_STATUS.eq(status).astype(float)
    prev['credit_to_requested'] = ratio(prev.AMT_CREDIT, prev.AMT_APPLICATION)
    parts.append(windows(prev, 'PREV', 'DAYS_DECISION', [('6m',183), ('12m',365)]))
    ins = instalments(tables['instalment_payments'])
    quality.append({'split':split, 'table':'instalment_payments', 'consolidated_due_instalments':len(ins),
                    'future_payment_rows_ignored':int(tables['instalment_payments'].DAYS_ENTRY_PAYMENT.gt(0).sum()),
                    'future_due_rows_excluded':int(tables['instalment_payments'].DAYS_INSTALMENT.gt(0).sum())})
    parts.append(windows(ins, 'INST', 'DAYS_INSTALMENT', [('6m',183), ('12m',365)]))
    history = tables['external_account_history']
    require_unique(history, ['SK_ID_BUREAU','MONTHS_BALANCE'], 'bureau monthly history')
    if not history.SK_ID_BUREAU.isin(bureau.SK_ID_BUREAU).all():
        raise ValueError('Orphan bureau history')
    history = keep_asof(history, 'MONTHS_BALANCE', 'external_account_history')
    history['delinquent'] = pd.to_numeric(history.STATUS, errors='coerce').gt(0).astype(float).where(history.STATUS.isin(['0','1','2','3','4','5']))
    history['unknown'] = history.STATUS.eq('X').astype(float)
    history['closed'] = history.STATUS.eq('C').astype(float)
    bh = windows(history, 'BH', 'MONTHS_BALANCE', [('6m',6),('12m',12)], 'SK_ID_BUREAU')
    bureau = keep_asof(bureau, 'DAYS_CREDIT', 'external_credit_accounts')
    bureau = keep_asof(bureau, 'DAYS_CREDIT_UPDATE', 'external_credit_accounts_update')
    bureau['active'] = bureau.CREDIT_ACTIVE.eq('Active').astype(float)
    bureau['debt_to_credit'] = ratio(bureau.AMT_CREDIT_SUM_DEBT, bureau.AMT_CREDIT_SUM)
    bureau['DAYS_ENDDATE_FACT'] = bureau.DAYS_ENDDATE_FACT.where(bureau.DAYS_ENDDATE_FACT.le(0))
    # Monthly histories become one row per bureau account before the applicant-level join.
    bureau = bureau.join(bh, on='SK_ID_BUREAU', validate='one_to_one')
    parts.append(windows(bureau, 'BUREAU', 'DAYS_CREDIT', [('6m',183),('12m',365)]))
    debt = bureau.groupby(ID).AMT_CREDIT_SUM_DEBT.sum(min_count=1)
    app['EXTERNAL_DEBT_TO_INCOME'] = ratio(debt.reindex(app.index), app.AMT_INCOME_TOTAL)
    for name, prefix in [('pos_cash_balance','POS'), ('credit_card_balance','CARD')]:
        d = tables[name]
        require_unique(d, ['SK_ID_PREV','MONTHS_BALANCE'], name)
        d = keep_asof(d, 'MONTHS_BALANCE', name)
        d['late'] = d.SK_DPD.gt(0).astype(float).where(d.SK_DPD.notna())
        if prefix == 'CARD':
            d['utilisation'] = ratio(d.AMT_BALANCE, d.AMT_CREDIT_LIMIT_ACTUAL)
            d['payment_to_minimum'] = ratio(d.AMT_PAYMENT_TOTAL_CURRENT, d.AMT_INST_MIN_REGULARITY)
        parts.append(windows(d, prefix, 'MONTHS_BALANCE', [('6m',6),('12m',12)]))
    for part in parts:
        app = app.join(part, how='left', validate='one_to_one')
        for c in part.columns:
            if c.endswith(('__row_count','__loan_count')):
                app[c] = app[c].fillna(0)
        prefix = part.columns[0].split('__')[0]
        app = app.copy().assign(**{f'{prefix}__has_history': app.index.isin(part.index).astype(int)})
    app = app.replace([np.inf,-np.inf], np.nan)
    require_unique(app.reset_index(), [ID], 'feature table')
    if y is not None:
        y.index = app.index
    return app, y, pd.DataFrame(quality), pd.DataFrame(columns)


def lean_columns(columns):
    return [c for c in columns if not c.startswith(('BUREAU','POS','CARD','EXTERNAL_DEBT'))]


def model_columns(columns):
    """Columns any model may use; monitoring-only personal attributes are excluded."""
    return [c for c in columns if c not in EXCLUDED_FROM_MODELS]


def feature_documentation(columns, dictionary):
    """Give every exported feature a source and mechanically traceable definition."""
    import re
    raw = dictionary.drop_duplicates('Row').set_index('Row').Description.to_dict()
    derived = {
        'AGE_YEARS':'-DAYS_BIRTH / 365.25',
        'EMPLOYMENT_YEARS':'-valid DAYS_EMPLOYED / 365.25',
        'EMPLOYMENT_SENTINEL':'1 if raw DAYS_EMPLOYED equals 365243; otherwise 0',
        'CREDIT_TO_INCOME':'AMT_CREDIT / positive AMT_INCOME_TOTAL',
        'ANNUITY_TO_INCOME':'AMT_ANNUITY / positive AMT_INCOME_TOTAL',
        'CREDIT_TO_GOODS':'AMT_CREDIT / positive AMT_GOODS_PRICE',
        'INCOME_PER_PERSON':'AMT_INCOME_TOTAL / positive CNT_FAM_MEMBERS',
        'EXTERNAL_DEBT_TO_INCOME':'sum of observed external debt / positive applicant income',
        'credit_to_requested':'previous AMT_CREDIT / positive AMT_APPLICATION',
        'amount':'one scheduled instalment amount, after consolidating split payments',
        'paid':'sum of instalment payments observed by application day 0',
        'on_time_paid':'sum of payments on or before the instalment due date',
        'completion_day':'first observed date cumulative payments reach the scheduled amount',
        'payment_unknown':'1 if any payment row has unknown payment date or unavailable observed payment amount',
        'observed_payments':'number of payment rows observed by application day 0',
        'shortfall':'max(scheduled amount - total observed payments, 0)',
        'payment_ratio':'observed instalment payments / positive scheduled amount',
        'delay_days':'max(completion day or day 0 if unfinished, minus due date, 0)',
        'late':'INST: on-time payments < positive scheduled amount; POS/CARD: SK_DPD > 0',
        'active':'1 if external CREDIT_ACTIVE is Active; otherwise 0',
        'debt_to_credit':'external AMT_CREDIT_SUM_DEBT / positive AMT_CREDIT_SUM',
        'delinquent':'bureau status 1-5 = 1, status 0 = 0; C/X = missing',
        'unknown':'1 if bureau monthly STATUS is X; otherwise 0',
        'closed':'1 if bureau monthly STATUS is C; otherwise 0',
        'utilisation':'AMT_BALANCE / positive AMT_CREDIT_LIMIT_ACTUAL',
        'payment_to_minimum':'AMT_PAYMENT_TOTAL_CURRENT / positive AMT_INST_MIN_REGULARITY'}
    sources={'PREV':'previous_loan_applications','INST':'instalment_payments',
             'BUREAU':'external_credit_accounts + external_account_history',
             'POS':'pos_cash_balance','CARD':'credit_card_balance','BH':'external_account_history'}
    def describe(name):
        match=re.match(r'^(PREV|INST|BUREAU|POS|CARD|BH)(_6m|_12m)?__(.*)$',name)
        if match:
            prefix,window,rest=match.groups()
            return f'{prefix} {window[1:] if window else "all available history"}: '+describe(rest)
        if name=='row_count': return 'number of available source observations (0 if absent)'
        if name=='loan_count': return 'number of distinct SK_ID_PREV loans (0 if absent)'
        if name=='has_history': return '1 if any as-of source history exists; otherwise 0'
        if name.endswith('__recent_minus_prior6m'):
            return 'mean over recent 6m minus mean over preceding nonoverlapping 6m of '+describe(name.removesuffix('__recent_minus_prior6m'))
        if '__' in name:
            field,stat=name.rsplit('__',1)
            if stat in ['mean','max','min']:return f'{stat}({describe(field)})'
        if name.startswith('status_'): return 'indicator of previous application status '+name.removeprefix('status_').replace('_',' ')
        return derived.get(name,raw.get(name,name))
    rows=[]
    lean=set(lean_columns(model_columns(columns)))
    for c in columns:
        prefix=c.split('__')[0].split('_')[0]
        source=sources.get(prefix,'loan_applications')
        if c=='EXTERNAL_DEBT_TO_INCOME':source='loan_applications + external_credit_accounts'
        rows.append(dict(feature=c,source=source,description=describe(c),
                         used_in_models=c not in EXCLUDED_FROM_MODELS,
                         exclusion_reason=EXCLUDED_FROM_MODELS.get(c,''),lean=c in lean))
    return pd.DataFrame(rows)
