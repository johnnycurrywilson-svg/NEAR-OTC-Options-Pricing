import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.iv_rv_premium import (weighted_median, realised_from_prices,
    make_peer_premium_table,estimate_neutral_premium,append_snapshots,
    evaluate_matured_snapshots,normalize_custom_closes)


def fake_closes(asset, first='2025-01-01', n=101, seed=1):
    rng=np.random.default_rng(seed)
    arr=np.exp(np.log(100)+np.cumsum(rng.normal(0,.035,n)))
    return pd.DataFrame({'asset':asset,'date':pd.date_range(first,periods=n,tz='UTC'), 'close':arr})


def test_weighted_median_not_forced_positive():
    assert weighted_median([-0.30,0.1,0.5],[3,1,1]) == -.30
    assert weighted_median([0.1,0.5],[1,3]) == 0.5


def test_negative_peer_ratio_is_allowed():
    tbl=pd.DataFrame({'asset':['HYPE','SOL','AVAX'],
                      'log_iv_rv_ratio':[math.log(.8),math.log(.9),math.log(1.2)],
                      'quality_score':[3,2,1]})
    est=estimate_neutral_premium(tbl)
    assert est['premium_ratio'] < 0
    assert est['n_assets']==3


def test_asset_not_venue_is_unit_of_peer_premium():
    fits=pd.DataFrame([
        {'asset':a,'venue':v,'dte':30.0,'atm':iv,'slope':0.0,
         'curvature':0.07,'quality_score':.9}
        for a,v,iv in [('HYPE','derive',.8),('HYPE','deribit',.9),('SOL','deribit',.6)]
    ])
    rv=pd.DataFrame({'asset':['HYPE','SOL'],'rv30':[.7,.5],
                     'last_close_day':['2026-10-29']*2})
    tbl=make_peer_premium_table(fits,rv,'2026-10-30T14:00:00+00:00')
    assert len(tbl)==2
    assert set(tbl.asset)=={'HYPE','SOL'}
    assert tbl.loc[tbl.asset.eq('HYPE'),'atm30_iv'].iloc[0] == pytest.approx(.85)
    assert estimate_neutral_premium(tbl)['n_assets']==2


def test_contiguous_price_check_and_30day_rv():
    prices=pd.concat([fake_closes('SOL'),fake_closes('HYPE',seed=2)])
    rv=realised_from_prices(prices)
    assert len(rv)==2
    assert np.isfinite(rv.rv30).all()
    broken=prices[~((prices.asset=='SOL')&(prices.date==pd.Timestamp('2025-02-01',tz='UTC')))]
    assert set(realised_from_prices(broken).asset)=={'HYPE'}


def test_forward_rv_only_after_30_complete_days(tmp_path:Path):
    prices=fake_closes('SOL',first='2025-01-01',n=101)
    old=pd.DataFrame([{'asof_utc':'2025-01-10T10:00:00+00:00','asset':'SOL',
        'atm30_iv':.7,'trailing30_rv':.6,'quality_score':1.0},
        {'asof_utc':'2025-04-10T10:00:00+00:00','asset':'SOL',
         'atm30_iv':.7,'trailing30_rv':.6,'quality_score':1.0}])
    path=tmp_path/'archive.csv'
    append_snapshots(path,old)
    append_snapshots(path,old)
    assert len(pd.read_csv(path))==2
    score=evaluate_matured_snapshots(path,prices)
    assert len(score)==1
    assert score.iloc[0].future30_rv>0
    assert score.iloc[0].evaluation_status.startswith('complete')


def test_custom_csv_validation():
    with pytest.raises(ValueError):normalize_custom_closes(pd.DataFrame({'close':[1]}))
    out=normalize_custom_closes(pd.DataFrame({'asset':['sol','SOL'],
        'date':['2026-10-28','2026-10-28'],'close':[2.0,3.0]}))
    assert len(out)==1 and out.iloc[0].close==3.0
