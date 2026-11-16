from datetime import datetime, timezone, timedelta
import pandas as pd
import pytest
from src.near_data import historical_volatility, fetch_near_daily_prices


def test_historical_vol_windows():
    dates = pd.date_range('2026-01-01', periods=110, freq='D', tz='UTC')
    import numpy as np
    closes = np.exp(np.cumsum(np.linspace(-.04,.05,110)))
    rows=historical_volatility(pd.DataFrame({'date':dates,'close':closes}))
    assert rows['lookback_days'].tolist()==[7,14,30,60,90]
    assert (rows['realized_vol_decimal'] >= 0).all()


def test_auto_fetch_excludes_incomplete_today(monkeypatch):
    import src.near_data as md
    now = datetime.now(timezone.utc)
    midnight = now.replace(hour=0,minute=0,second=0,microsecond=0)
    candles=[]
    for i in range(111):
        t=midnight-timedelta(days=110-i)
        candles.append([int(t.timestamp()),'1','1','1',str(1+.001*i),'1','1',10])
    class Response:
        def __init__(self, data):self.data=data
        def raise_for_status(self):pass
        def json(self):return {'error':[], 'result':self.data}
    class Session:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def get(self,url,params,timeout):
            if 'OHLC' in url:return Response({'NEARUSD':candles,'last':1})
            return Response({'NEARUSD':{'c':['2.5','0']}})
    monkeypatch.setattr(md.requests,'Session',Session)
    prices,spot,meta=fetch_near_daily_prices()
    assert len(prices)==110
    assert float(prices.iloc[-1]['close'])==pytest.approx(1.109)
    assert spot==2.5
    assert meta['candles_count']==110
