import math
import numpy as np
import pytest
from src.model import black76, NearSurface, fit_peer_smiles, simple_arbitrage_checks
from src.market_data import sample_synthetic_peers,parse_tickers
from datetime import datetime,timezone


def test_put_call_parity():
    c=black76(100,105,30/365,.8,'C',discount_rate=.03)['price']
    p=black76(100,105,30/365,.8,'P',discount_rate=.03)['price']
    assert c-p == pytest.approx(math.exp(-.03*30/365)*(100-105),abs=1e-9)


def test_delta_and_vega():
    q=black76(100,100,30/365,1.1,'C')
    assert 0.5<q['forward_delta']<.75
    assert q['forward_gamma']>0
    assert q['vega_per_1volpt']>0


def test_surface_anchor_and_odd_wing():
    fit=fit_peer_smiles(sample_synthetic_peers())
    sf=NearSurface(fit,1.10,.08)
    assert sf.at(30)['atm']==pytest.approx(1.188)
    assert 0.7<sum(sf.weights.values())<1.1
    assert sf.iv(4.4,4.4,30)==pytest.approx(1.188)
    assert sf.iv(4.4,4.4*.85,30)>0
    assert sf.quote(4.4,4.4,30)['bid']<sf.quote(4.4,4.4,30)['ask']


def test_stale_rejected():
    a=datetime(2026,10,30,10,tzinfo=timezone.utc)
    payload={'tickers':{'HYPE-20261127-25-C':{'option_pricing':{'i':'0.8','f':'25'},'t':1}}}
    assert parse_tickers(payload,'HYPE',a)==[]


def test_demo_butterfly_grid():
    fit=fit_peer_smiles(sample_synthetic_peers())
    sf=NearSurface(fit,1.10,.08)
    assert simple_arbitrage_checks(sf,4.4)==[]


def test_two_sided_ticker_and_mark_only_quality():
    asof=datetime(2026,10,30,10,tzinfo=timezone.utc)
    stamp=int(asof.timestamp()*1000)
    payload={'tickers':{
        'HYPE-20261127-30-C':{
            'instrument_name':'HYPE-20261127-30-C','t':stamp,
            'best_bid_price':'0.9','best_ask_price':'1.1',
            'best_bid_amount':'2','best_ask_amount':'3',
            'option_pricing':{'iv':'0.75','bid_iv':'0.73','ask_iv':'0.77','forward_price':'25','delta':'0.25'}},
        'HYPE-20261127-21-P':{
            'instrument_name':'HYPE-20261127-21-P','t':stamp,
            'option_pricing':{'iv':'0.82','forward_price':'25','delta':'-0.25'}}}}
    rows=parse_tickers(payload,'HYPE',asof)
    assert len(rows)==2
    assert rows[0]['quality']=='two_sided' and rows[0]['iv']==pytest.approx(.75)
    assert rows[1]['quality']=='mark_only' and rows[1]['quality_weight']<rows[0]['quality_weight']
