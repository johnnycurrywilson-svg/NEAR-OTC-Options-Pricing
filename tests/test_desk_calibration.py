import math
import numpy as np
import pandas as pd
import pytest
from src.desk_calibration import (
    forecast_near_rv, rolling_forward_rv_backtest, robust_peer_premium,
    historical_iv_vs_future_rv, cost_aware_two_way_quote
)


def peers():
    return pd.DataFrame([
        {'asset':'AVAX','atm30_iv':.6852,'trailing30_rv':.6709,'quality_score':.705},
        {'asset':'HYPE','atm30_iv':.6295,'trailing30_rv':.6212,'quality_score':.759},
        {'asset':'SOL','atm30_iv':.5880,'trailing30_rv':.4242,'quality_score':.648},
        {'asset':'ZEC','atm30_iv':1.0968,'trailing30_rv':.9526,'quality_score':.612},
    ])

def test_robust_and_loo():
    r=robust_peer_premium(peers())
    assert 0.07<r['premium_ratio']<0.13
    assert len(r['leave_one_out'])==4
    assert r['leave_one_out_min']<=r['premium_ratio']<=r['leave_one_out_max']
    assert r['simple_median_ratio'] != r['weighted_mean_ratio']

def test_negative_allowed():
    p=peers();p['atm30_iv']=p['trailing30_rv']*0.9
    assert robust_peer_premium(p)['premium_ratio'] == pytest.approx(-.1,abs=1e-6)

def test_rough_near_forward_blend():
    f=forecast_near_rv({7:1.23164,30:1.4187,90:1.23309})
    assert f['forecast_rv']==pytest.approx(1.31958,abs=0.001)

def test_rolling_uses_only_prior_info_and_future_window():
    dates=pd.date_range('2025-01-01',periods=180,freq='D',tz='UTC')
    close=100*np.exp(np.cumsum(np.r_[0,.01+.008*np.sin(np.arange(179)*.37)]))
    raw=pd.DataFrame({'date':dates,'close':close})
    out=rolling_forward_rv_backtest(raw)
    assert not out.empty
    assert (out.subsequent_rv30>=0).all()
    # Changing any future prices must not change the first historical forecast.
    new=raw.copy();new.loc[new.index>90,'close']*=1.2
    assert out.iloc[0].forecast_rv30==pytest.approx(rolling_forward_rv_backtest(new).iloc[0].forecast_rv30)

def test_historical_iv_requires_completed_future_window():
    dates=pd.date_range('2026-01-01',periods=80,tz='UTC')
    closes=pd.DataFrame({'asset':'SOL','date':dates,'close':100*np.exp(np.cumsum(np.r_[0,np.full(79,0.003)]))})
    iv=pd.DataFrame([{'asset':'SOL','asof_utc':'2026-02-01T15:00:00Z','atm30_iv':.7},
                     {'asset':'SOL','asof_utc':'2026-03-20T15:00:00Z','atm30_iv':.7}])
    out=historical_iv_vs_future_rv(iv,closes)
    assert len(out)==1
    assert out.iloc[0].window_start_completed_close=='2026-02-01'

def test_costs_are_symmetric_and_change_with_costs():
    class Surface:
        def iv(self,forward,strike,days):return 1.3
        def at(self,days):return {'atm':1.3}
    a=cost_aware_two_way_quote(Surface(),4.4,4.4,30,'C',fee_bps=1,half_spread_bps=1,slippage_bps=1)
    b=cost_aware_two_way_quote(Surface(),4.4,4.4,30,'C',fee_bps=10,half_spread_bps=10,slippage_bps=10)
    assert a['bid_iv']<a['iv']<a['ask_iv']
    assert (a['ask_iv']-a['iv'])==pytest.approx(a['iv']-a['bid_iv'])
    assert b['spread_vol_pts']>a['spread_vol_pts']
    assert b['bid']<b['mid']<b['ask']
