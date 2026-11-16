from datetime import datetime, timezone
import math
import pandas as pd
import pytest
import numpy as np
from src.historical_deribit import parse_trade, fetch_trade_window, build_atm_observations, backfill_history
from src.desk_calibration import historical_iv_vs_future_rv


def tr(name,day='2026-07-15',hr=23,iv=.78,price=100,seq=1):
    t=pd.Timestamp(day+'T%02d:00:00Z'%hr)
    return {'instrument_name':name,'timestamp':int(t.timestamp()*1000),
            'index_price':price,'iv':iv,'amount':2.0,'trade_id':f'{seq}',
            'price':3.1}


def test_history_trade_filter_and_scaling():
    a=tr('SOL_USDC-14AUG26-104-C',price=100)
    assert parse_trade(a)['asset']=='SOL'
    a['iv']=78
    assert parse_trade(a)['iv']==pytest.approx(.78)
    assert parse_trade(tr('BTC-14AUG26-104-C')) is None
    assert parse_trade(tr('SOL_USDC-14AUG26-200-C',price=100)) is None
    assert parse_trade(tr('SOL_USDC-14AUG26-104-C',price=100,iv=-1)) is None


def test_history_trade_build_requires_qualifying_instruments():
    trades=[tr('SOL_USDC-14AUG26-99-P',seq=1),
            tr('SOL_USDC-14AUG26-102-C',seq=2),
            tr('SOL_USDC-14AUG26-100-C',seq=3)]
    obs,detail=build_atm_observations(trades,'2026-07-15')
    assert len(obs)==1 and obs.iloc[0].asset=='SOL'
    assert obs.iloc[0].atm30_iv==pytest.approx(.78)
    assert obs.iloc[0].n_unique_instruments==3
    assert obs.iloc[0].quote_quality=='near-ATM actual trades, NOT mid marks'
    empty,_=build_atm_observations(trades[:1],'2026-07-15')
    assert empty.empty


def test_history_pages_fully_consumed():
    t1=tr('SOL_USDC-14AUG26-100-C',seq=1)
    t2=tr('SOL_USDC-14AUG26-100-C',seq=2)
    t2['timestamp']+=1000
    calls=[]
    def resp(session,params):
        calls.append(params.copy())
        if len(calls)==1:return {'trades':[t1,t2],'has_more':True}
        return {'trades':[],'has_more':False}
    got,info=fetch_trade_window('2026-07-15T20:00:00Z','2026-07-16T00:00:00Z',
                               page_size=2,max_pages=4,request_function=resp,pause_seconds=0)
    assert len(got)==2 and info['status']=='complete'
    assert calls[0]['currency']=='USDC' and calls[0]['include_old']=='true'
    assert calls[1]['start_timestamp']==t2['timestamp']+1


def test_incomplete_history_window_rejected():
    t1=tr('SOL_USDC-14AUG26-100-C',seq=1)
    t2=tr('SOL_USDC-14AUG26-101-C',seq=2);t2['timestamp']+=1000
    def resp(session,params):return {'trades':[t1,t2],'has_more':True}
    got,info=fetch_trade_window('2026-07-15T20:00:00Z','2026-07-16T00:00:00Z',
                        page_size=2,max_pages=1,request_function=resp,pause_seconds=0)
    assert got==[] and info['status']=='page_cap_incomplete'


def test_future_rv_exactly_after_snapshot_and_missing_excluded():
    dates=pd.date_range('2026-07-01',periods=70,freq='D',tz='UTC')
    closes=pd.DataFrame({'asset':'SOL','date':dates,'close':100*np.exp(.01*np.arange(70))})
    obs=pd.DataFrame([
        {'asset':'SOL','asof_utc':'2026-07-15T22:00:00Z','atm30_iv':.65},
        {'asset':'SOL','asof_utc':'2026-09-01T22:00:00Z','atm30_iv':.65},
    ])
    scores=historical_iv_vs_future_rv(obs,closes)
    assert len(scores)==1
    assert scores.iloc[0].window_start_completed_close=='2026-07-15'
    assert scores.iloc[0].window_end_close=='2026-08-14'
    assert scores.iloc[0].iv_squared_minus_future_rv_squared==pytest.approx(.65**2,abs=1e-8)
