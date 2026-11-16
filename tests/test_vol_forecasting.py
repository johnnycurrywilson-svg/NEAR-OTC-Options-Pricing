"""Deterministic smoke tests for walk-forward, look-ahead safeguards and model choices."""
import numpy as np
import pandas as pd
import pytest
from src.vol_forecasting import (compare_forecasts, make_history, prepare_closes,
                                  METHODS, LAMBDAS)


def sample(seed=24, n=720):
    gen=np.random.default_rng(seed)
    daily=gen.normal(0,1,n-1)*np.array([.03 if t<200 else .055 if t<400 else .035 for t in range(n-1)])
    closes=5*np.exp(np.r_[0,np.cumsum(daily)])
    return pd.DataFrame({'date':pd.date_range('2024-10-01',periods=n,tz='UTC'), 'close':closes})


def test_comparison_and_holdout_separation():
    rows, scores, summary=compare_forecasts(sample())
    assert set(scores.method)==set(METHODS)
    assert set(scores.segment)=={'selection','holdout','all'}
    assert summary['holdout_dates']>=6
    assert summary['selected_method'] in METHODS
    assert summary['deployed_method'] in METHODS
    if not summary['deployment_gate_passed']:
        assert summary['deployed_method']=='Fixed blend'
    early=scores[scores.segment.eq('selection')].sort_values(['mae_vol_points','method'])
    assert summary['selected_method']==early.iloc[0].method
    assert 0.15<summary['forecast_rv']<3.0
    assert len(rows)==sum(scores.loc[scores.segment.eq('all'),'n'])


def test_historical_origin_features_not_affected_by_future_prices():
    raw=sample()
    hist=make_history(raw)
    earliest=hist.iloc[0]
    changed=raw.copy()
    changed.loc[changed.index>int(earliest.origin_index),'close']*=2
    other=make_history(changed)
    for col in ['rv7','rv30','rv90']+[f'ewma_{v:g}' for v in LAMBDAS]:
        assert earliest[col]==pytest.approx(other.iloc[0][col])


def test_bad_dates_raise_instead_of_invented_history():
    x=sample()
    x=x.drop(index=250)
    with pytest.raises(ValueError,match='Missing daily'):
        prepare_closes(x)


def test_all_training_labels_mature_before_prediction():
    raw=sample()
    pred,_,_=compare_forecasts(raw)
    first=pred.iloc[0].forecast_date_utc
    rows=make_history(raw)
    t=int(rows.loc[rows.forecast_date_utc.eq(first),'origin_index'].iloc[0])
    eligible=rows[rows.outcome_end_index<=t]
    assert int(pred.iloc[0].matured_training_samples)==len(eligible)
    assert (eligible.outcome_end_index<=t).all()
