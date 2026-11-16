import json
from pathlib import Path
import pytest
from desk_update import estimate_two_way,build_review
from src.research_inputs import hourly_diagnostics
import pandas as pd


def test_symmetric_two_sided_pricing():
    r=estimate_two_way(3.51,3.51,'C',1.189)
    assert r['bid_usd']<r['fair_usd']<r['offer_usd']
    assert abs(r['fair_iv_pct']-((r['bid_iv_pct']+r['offer_iv_pct'])/2))<1e-9
    assert r['total_spread_vol_points']>0


def test_more_execution_cost_widens_quote():
    a=estimate_two_way(3.51,3.51,'C',1.189,execution_bps=8,hedges=12)
    b=estimate_two_way(3.51,3.51,'C',1.189,execution_bps=30,hedges=60)
    assert b['total_spread_vol_points']>a['total_spread_vol_points']
    assert b['bid_usd']<a['bid_usd'] and b['offer_usd']>a['offer_usd']


def test_frozen_snapshot_does_not_auto_price_har():
    m=json.loads((Path(__file__).parents[1]/'data'/'snapshot_2026-11-12.json').read_text())
    report=build_review()
    assert report['fair_atm_iv']>1.1
    assert abs(report['experimental_har']['har_rv_pct']-100*m['near_forward_rv_30'])<1e-6
    assert report['fixed_forward_rv']>m['near_forward_rv_30']


def test_hourly_requires_no_gaps():
    x=pd.DataFrame({'date':pd.to_datetime(['2026-11-01T00:00:00Z','2026-11-01T02:00:00Z']), 'close':[1,2]})
    with pytest.raises(ValueError,match='Missing hourly'):hourly_diagnostics(x)


def test_wrong_asof_is_rejected(tmp_path):
    m=json.loads((Path(__file__).parents[1]/'data'/'snapshot_2026-11-12.json').read_text())
    m['asof']='2026-11-13T00:00:00+00:00'
    p=tmp_path/'altered.json';p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='locked'): build_review(p)
