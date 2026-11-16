from datetime import datetime, timezone, timedelta
import pandas as pd
import pytest
from src.deribit_data import parse_deribit, _option_price, _price_to_iv, sample_synthetic_deribit
from src.market_data import sample_synthetic_peers
from src.model import fit_peer_smiles, peer_at, NearSurface


def test_bid_ask_implied_volatility_recovers_original():
    f=20.;k=22.;t=30/365
    for side in ('C','P'):
        p=_option_price(f,k,t,.78,side)
        assert _price_to_iv(p,f,k,t,side)==pytest.approx(.78,abs=1e-7)


def test_deribit_linear_usdc_quote_parsing_and_percent_scaling():
    asof=datetime(2026,10,30,tzinfo=timezone.utc)
    exp=asof+timedelta(days=30)
    name='HYPE_USDC-29NOV26-22-C'
    inst=[{'instrument_name':name,'expiration_timestamp':int(exp.timestamp()*1000),
           'strike':22.,'is_active':True,'state':'open'}]
    t=30/365
    summary=[{'instrument_name':name,'underlying_price':20.,'mark_iv':80.,
              'bid_price':_option_price(20,22,t,.75,'C'),
              'ask_price':_option_price(20,22,t,.83,'C'),'open_interest':45}]
    rows=parse_deribit(inst,summary,asof)
    assert len(rows)==1
    node=rows.iloc[0]
    assert node.venue=='deribit' and node.asset=='HYPE'
    assert node.mark_iv==pytest.approx(.80)
    assert node.iv==pytest.approx(.79,abs=1e-5)
    assert node.quality=='two_sided'
    assert not node.quote_time_verified


def test_two_venues_do_not_duplicate_peer_weight():
    nodes=pd.concat([sample_synthetic_peers(),sample_synthetic_deribit()],ignore_index=True)
    fits=fit_peer_smiles(nodes)
    assert set(fits.venue)=={'derive','deribit'}
    assert peer_at(fits,'SOL',30)['venue_count']==2
    surf=NearSurface(fits,1.10)
    assert len(surf.peers)==5
    assert surf.weights['HYPE']==pytest.approx(.30)
    assert surf.weights['SOL']==pytest.approx(.25)
    assert sum(surf.weights.values())==pytest.approx(1.0)
