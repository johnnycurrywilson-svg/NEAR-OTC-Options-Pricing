"""Read-only Deribit USDC-linear altcoin options adapter.

Deribit JSON-RPC v2 (no API key). One instrument-list request and one
book-summary request for the entire USDC options universe. Underlying forwards
and quote marks are supplied by Deribit; this module DOES NOT claim that a
public book summary has verified quote freshness or displayed executable size.
"""
from __future__ import annotations
import math
import re
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq
from scipy.stats import norm

BASE = 'https://www.deribit.com/api/v2/public'
ASSETS = ('AVAX', 'HYPE', 'SOL', 'TRX')
NAME = re.compile(r'^(AVAX|HYPE|SOL|TRX)_USDC-(\d{1,2}[A-Z]{3}\d{2})-([0-9.]+)-([CP])$')


def _n(value, default=float('nan')):
    try:
        f=float(value)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def public_get(session: requests.Session, method: str, params:dict, attempts=3):
    last=None
    for i in range(attempts):
        try:
            response=session.get(f'{BASE}/{method}',params=params,timeout=22)
            response.raise_for_status()
            data=response.json()
            if data.get('error'):
                raise RuntimeError(f'Deribit {method}: {data["error"]}')
            if 'result' not in data:
                raise ValueError(f'Deribit {method}: missing result')
            return data['result']
        except (requests.RequestException, ValueError, RuntimeError) as ex:
            last=ex
            if i+1 < attempts:
                time.sleep(.7*(i+1))
    raise RuntimeError(f'Deribit public/{method} failed: {last}')


def _option_price(f,k,t,iv,side):
    if not (f>0 and k>0 and t>0 and iv>0):return float('nan')
    d1=(math.log(f/k)+iv*iv*t/2)/(iv*math.sqrt(t));d2=d1-iv*math.sqrt(t)
    return f*norm.cdf(d1)-k*norm.cdf(d2) if side=='C' else k*norm.cdf(-d2)-f*norm.cdf(-d1)


def _price_to_iv(price,f,k,t,side):
    """USDC-linear quoted premium is USD/underlying; no discounting assumption."""
    if not (math.isfinite(price) and price>0):return float('nan')
    intrinsic=max(f-k,0.0) if side=='C' else max(k-f,0.0)
    if price<=intrinsic+1e-10:return float('nan')
    try:
        if price>=_option_price(f,k,t,6.,side):return float('nan')
        return float(brentq(lambda v:_option_price(f,k,t,v,side)-price,.005,6.,xtol=1e-9))
    except (ValueError, RuntimeError, OverflowError):return float('nan')


def parse_deribit(instruments:list, summaries:list, asof:datetime, max_days=180):
    """Join contract metadata to summary marks, preserve source and quote quality.

    WARNING: creation_timestamp is NOT the book update time. Treat the asof
    timestamp as a collection time, and allow a model mark even when the book
    is not two-sided, but downweight it sharply.
    """
    names={}
    for inst in instruments:
        if not isinstance(inst,dict):continue
        name=str(inst.get('instrument_name',''))
        m=NAME.fullmatch(name)
        if not m or not inst.get('is_active',True):continue
        if str(inst.get('state','open')).lower() not in ('open',''):continue
        expiry_ms=_n(inst.get('expiration_timestamp'))
        if not math.isfinite(expiry_ms):continue
        exp=datetime.fromtimestamp(expiry_ms/1000 if expiry_ms>1e11 else expiry_ms,tz=timezone.utc)
        days=(exp-asof).total_seconds()/86400.
        if not (2 <= days <= max_days):continue
        strike=_n(inst.get('strike'),float(m.group(3)))
        if strike<=0:continue
        names[name] = (m.group(1),exp,days,strike,m.group(4))
    rows=[]
    for item in summaries:
        if not isinstance(item,dict):continue
        name=str(item.get('instrument_name',''))
        if name not in names:continue
        asset,exp,days,k,side=names[name]
        f=_n(item.get('underlying_price'))
        if not (f>0):continue
        if (side=='C' and k <f*.999) or (side=='P' and k>f*1.001):continue
        mark_pct=_n(item.get('mark_iv'))
        if not (5. < mark_pct < 500.):continue  # Deribit reports IV in percentage points.
        mark_iv=mark_pct/100.
        bid_price=_n(item.get('bid_price'))
        ask_price=_n(item.get('ask_price'))
        t=days/365.
        bid_iv=_price_to_iv(bid_price,f,k,t,side)
        ask_iv=_price_to_iv(ask_price,f,k,t,side)
        valid_prices=(math.isfinite(bid_price) and math.isfinite(ask_price)
                      and ask_price > bid_price > 0)
        two_sided=(valid_prices and math.isfinite(bid_iv) and math.isfinite(ask_iv)
                   and 0<ask_iv-bid_iv<.50)
        if two_sided:
            iv=(bid_iv+ask_iv)/2
            quality='two_sided'
            quality_weight=.75   # Depth and per-quote freshness not provided in summary
        elif ((math.isfinite(bid_price) and bid_price>0) or
              (math.isfinite(ask_price) and ask_price>0)):
            iv=mark_iv;quality='one_sided';quality_weight=.25
        else:
            iv=mark_iv;quality='mark_only';quality_weight=.12
        rows.append({'asset':asset,'venue':'deribit','expiry':exp.isoformat(),
                     'dte':round(days,4),'strike':k,'right':side,'forward':f,
                     'iv':iv,'mark_iv':mark_iv,'bid_iv':bid_iv,'ask_iv':ask_iv,
                     'quality':quality,'quality_weight':quality_weight,
                     'source':'deribit_public_api_USDC_linear','asof':asof.isoformat(),
                     'instrument':name,'delta':float(norm.cdf((math.log(f/k)+.5*iv*iv*t)/(iv*math.sqrt(t))))*(1 if side=='C' else 1) - (1 if side=='P' else 0),
                     'bid_price':bid_price,'ask_price':ask_price,
                     'open_interest':_n(item.get('open_interest')),
                     'volume_24h':_n(item.get('volume')),
                     'quote_time_verified':False})
    return pd.DataFrame(rows)


def download_deribit(asof=None):
    asof=asof or datetime.now(timezone.utc)
    with requests.Session() as s:
        instruments=public_get(s,'get_instruments',{'currency':'USDC','kind':'option','expired':'false'})
        summaries=public_get(s,'get_book_summary_by_currency',{'currency':'USDC','kind':'option'})
    if not isinstance(instruments,list) or not isinstance(summaries,list):
        raise RuntimeError('Unexpected Deribit API response shape')
    nodes=parse_deribit(instruments,summaries,asof)
    if nodes.empty:raise RuntimeError('No suitable Deribit altcoin option nodes; inspect currency pair and API payload')
    return nodes


def sample_synthetic_deribit(asof='2026-10-30'):
    """Fake numbers ONLY for pipeline testing; do NOT cite as a market observation."""
    anchors={'AVAX':{'f':12.,'atm':(.85,.77,.72),'a':(-.055,-.045,-.025),'b':(.06,.055,.065)},
             'HYPE':{'f':27.,'atm':(.92,.83,.79),'a':(.065,.043,.035),'b':(.06,.05,.055)},
             'SOL':{'f':165.,'atm':(.56,.49,.47),'a':(-.01,-.012,-.02),'b':(.05,.045,.05)},
             'TRX':{'f':.34,'atm':(.62,.57,.54),'a':(.005,.008,-.012),'b':(.04,.05,.055)}}
    asof_ts=pd.Timestamp(asof,tz='UTC');rows=[]
    for asset,p in anchors.items():
        for j,days in enumerate((7.,30.,90.)):
            exp=asof_ts+pd.Timedelta(days=days)
            atm=p['atm'][j]
            for z in (-1.8,-1.4,-1,-.65,-.3,0,.3,.65,1,1.4,1.8):
                for side in (('P','C') if z==0 else ('P',) if z<0 else ('C',)):
                    k=p['f']*np.exp(z*atm*np.sqrt(days/365))
                    iv=atm*(1+p['a'][j]*z+p['b'][j]*z*z)
                    rows.append({'asset':asset,'venue':'deribit','expiry':exp.isoformat(),
                        'dte':days,'strike':float(k),'right':side,'forward':p['f'],
                        'iv':iv,'mark_iv':iv,'bid_iv':np.nan,'ask_iv':np.nan,
                        'quality':'synthetic_test','quality_weight':1.,
                        'source':'SYNTHETIC_DEMO_NOT_REAL_QUOTE','asof':asof,
                        'instrument':f'DEMO-DERIBIT-{asset}-{int(days)}-{side}',
                        'delta':np.nan,'quote_time_verified':False})
    return pd.DataFrame(rows)
