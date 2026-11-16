import math
import numpy as np
import pandas as pd
import pytest
from src.peer_svi_surface import svi_w,fit_slice,prepare_nodes,near_smile_from_peers,static_checks,TGT_T

def artificial_valid_slice(T=.082, asset='SOL',venue='deribit'):
    p=np.array([.015,.10,-.2,0.,.19])
    xs=np.linspace(-.24,.24,17)
    iv=np.sqrt(svi_w(xs,p)/T)
    return pd.DataFrame(dict(k=xs,strike=np.exp(xs)*100,iv=iv,bid_iv=iv-.01,ask_iv=iv+.01,dte=np.full(17,T*365)))

def test_svi_fits_known_synthetic_slice():
    frame=artificial_valid_slice()
    fitted=fit_slice(frame,'SOL','deribit','synthetic_test_only')
    assert fitted.rmse_vol_pts<1.1
    assert abs(fitted.iv(0)-frame.iloc[8].iv)<.03

def test_reject_one_wing():
    with pytest.raises(ValueError,match='both OTM wings'):
        fit_slice(artificial_valid_slice().query('k>=0'), 'SOL','deribit','test')

def test_forbid_synthetic_filled_market():
    d=pd.DataFrame({'asset':['SOL'],'strike':[100],'forward':[100],'expiry':['2026-12-12'],'dte':[30],
        'iv':[.5],'bid_iv':[.49],'ask_iv':[.51],'right':['C'],'quality':['two_sided'],
        'asof':['2026-11-12T07:41:41Z'],'source':['SYNTHETIC_DEMO_NOT_REAL_QUOTE']})
    with pytest.raises(ValueError,match='Synthetic'):
        prepare_nodes(d,'2026-11-12T07:41:41Z')

def test_no_near_curve_from_one_asset():
    rows=pd.DataFrame({'asset':['SOL'],'venue':['deribit'],'z':[0.],'iv_ratio':[1.]})
    with pytest.raises(ValueError,match='Only 1 peer assets'):
        near_smile_from_peers(rows,1.189,3.51,[1.0])

def test_constant_peer_iv_ratio_has_flat_surface():
    m=[.85,.90,.95,1.,1.05,1.10,1.15,1.20]
    z=np.log(m)/(1.189*np.sqrt(TGT_T))
    rows=[dict(asset=a,venue='derive',z=round(float(q),7),iv_ratio=1.) for a in ('SOL','HYPE','AVAX') for q in z]
    df=near_smile_from_peers(pd.DataFrame(rows),1.189,3.51,m)
    assert np.max(abs(df.model_iv_pct-118.9))<1e-9
    assert static_checks(df)==[]


def test_prepare_nodes_reads_asof_csv_column_not_pandas_method():
    # Regression for timestamp on actual dated 12-Nov source CSV.
    timestamp='2026-11-12T07:41:32.281927+00:00'
    raw=pd.DataFrame({
        'asset':['SOL'], 'strike':[100.0], 'forward':[100.0],
        'expiry':['2026-12-12T08:00:00+00:00'], 'dte':[30.0],
        'iv':[.65], 'bid_iv':[.64], 'ask_iv':[.66],
        'right':['C'], 'quality':['two_sided'], 'asof':[timestamp],
        'source':['Deribit LIVE']})
    rows=prepare_nodes(raw,'2026-11-12T07:41:41.643402+00:00')
    assert len(rows)==1
    assert str(rows.iloc[0]['stamp'].tzinfo)=='UTC'
    assert abs(rows.iloc[0]['k'])<1e-12

def test_prepare_nodes_rejects_stale_source_timestamps():
    timestamp='2026-11-11T07:41:32.281927+00:00'
    raw=pd.DataFrame({
        'asset':['SOL'], 'strike':[100.0], 'forward':[100.0],
        'expiry':['2026-12-12T08:00:00+00:00'], 'dte':[30.0],
        'iv':[.65], 'bid_iv':[.64], 'ask_iv':[.66],
        'right':['C'], 'quality':['two_sided'], 'asof':[timestamp],
        'source':['Deribit LIVE']})
    rows=prepare_nodes(raw,'2026-11-12T07:41:41.643402+00:00')
    assert rows.empty
