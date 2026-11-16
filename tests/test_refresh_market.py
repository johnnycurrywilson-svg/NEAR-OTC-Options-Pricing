import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import pytest
from refresh_market import enrich

BASE={'asof':'2026-10-30T15:49:00+00:00','near_spot':4.28,
      'near_atm_iv_30':1.45,'near_forward_rv_30':1.32,
      'premium_peer_rows':[{'asset':'SOL','atm30_iv':.60}]}

def test_market_enrichment_uses_actual_timestamp_and_reference():
    now=pd.Timestamp('2026-11-11T10:00:00Z')
    days=pd.date_range('2026-08-01','2026-11-10',tz='UTC')
    prices=pd.DataFrame({'date':days,'close':[3.5+0.002*i for i in range(len(days))]})
    current={'mode':'live','asof':now.isoformat(), 'near_spot':3.56,
             'near_atm_iv_30':1.38,'near_forward_rv_30':1.22,
             'premium_peer_rows':[{'asset':'SOL','atm30_iv':.58}],
             'kraken_history':{'spot_asof_utc':now.isoformat()}}
    result=enrich(current,prices,BASE)
    m=result['market_update']
    assert m['type']=='full_live_refresh'
    assert m['spot_change_from_reference_pct']==pytest.approx(100*(3.56/4.28-1))
    assert m['peer_iv_comparison'][0]['change_iv_vol_pts']==pytest.approx(-2)
    assert 'return_30d_pct' in m['returns']

def test_market_enrichment_rejects_stale_history():
    days=pd.date_range('2026-10-01','2026-10-30',tz='UTC')
    prices=pd.DataFrame({'date':days,'close':[4.0]*len(days)})
    with pytest.raises(RuntimeError,match='stale'):
        enrich({'mode':'live','asof':'2026-11-11T10:00:00+00:00','near_spot':4},prices,BASE)
