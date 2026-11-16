"""Extra market observations for NEAR research. Read-only public endpoints.

None of the snapshots is an executable promise or a historic 12 Nov order book.
An unavailable feed is explicitly recorded, never replaced with fake observations.
"""
from __future__ import annotations
import datetime as dt
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

COINBASE = 'https://api.exchange.coinbase.com'
HYPERLIQUID = 'https://api.hyperliquid.xyz/info'


def _request_get(s, url, params, attempts=2):
    last = None
    for n in range(attempts):
        try:
            resp=s.get(url, params=params, timeout=18, headers={'User-Agent':'NEAR-research/1.0'})
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last=exc
            if n+1<attempts: time.sleep(0.8)
    raise RuntimeError(f'Price feed failed: {str(last)[:180]}')


def coinbase_candles(start, end, granularity=86400, session=None):
    """Paginated Coinbase Exchange historical candles. No fills for missing bars.

    End is exclusive. Coinbase permits maximum 300 candles per request.
    Responses are [unix_time, low, high, open, close, volume].
    """
    if granularity not in (3600,86400):raise ValueError('Use only hourly or daily granularity')
    start=pd.Timestamp(start);end=pd.Timestamp(end)
    start=start.tz_localize('UTC') if start.tz is None else start.tz_convert('UTC')
    end=end.tz_localize('UTC') if end.tz is None else end.tz_convert('UTC')
    if end<=start:raise ValueError('end must follow start')
    own=session is None;s=session or requests.Session();rows=[]
    try:
        cursor=start
        while cursor<end:
            chunk_end=min(end,cursor+pd.Timedelta(seconds=granularity*299))
            raw=_request_get(s,COINBASE+'/products/NEAR-USD/candles',
                             {'start':cursor.isoformat(), 'end':chunk_end.isoformat(),'granularity':granularity})
            if not isinstance(raw,list):raise RuntimeError('Unexpected Coinbase candles payload')
            for entry in raw:
                if not isinstance(entry,list) or len(entry)<6:continue
                stamp=pd.Timestamp(int(entry[0]),unit='s',tz='UTC')
                if cursor<=stamp<chunk_end:
                    rows.append({'date':stamp,'close':float(entry[4]),'volume':float(entry[5])})
            cursor=chunk_end
            time.sleep(.04)
    finally:
        if own:s.close()
    if not rows:raise RuntimeError('Coinbase returned no NEAR/USD candles; market may be unsupported')
    x=pd.DataFrame(rows).drop_duplicates('date').sort_values('date').reset_index(drop=True)
    if (~np.isfinite(x.close)).any() or (x.close<=0).any():raise RuntimeError('Invalid historical prices')
    expected=pd.Timedelta(seconds=granularity)
    if (x.date.diff().dropna()!=expected).any():
        gaps=int((x.date.diff().dropna()!=expected).sum())
        raise RuntimeError(f'Incomplete {granularity}s Coinbase series: {gaps} missing intervals; no interpolation')
    return x


def hourly_diagnostics(prices):
    x=prices.copy().sort_values('date'); x['date']=pd.to_datetime(x['date'],utc=True)
    if (x.date.diff().dropna()!=pd.Timedelta(hours=1)).any():raise ValueError('Missing hourly candles')
    rets=np.diff(np.log(x.close.to_numpy(dtype=float)))
    answer={'n_hourly_candles':int(len(x)),'source':'Coinbase Exchange NEAR-USD hourly candles',
            'data_first_utc':x.date.iloc[0].isoformat(),'data_last_utc':x.date.iloc[-1].isoformat()}
    for days in (7,30,90):
        n=days*24
        if len(rets)>=n:
            answer[f'close_to_close_hourly_rv_{days}d_pct']=float(100*np.std(rets[-n:],ddof=1)*np.sqrt(24*365))
    return answer


def _hl_post(s,body):
    r=s.post(HYPERLIQUID,json=body,timeout=18);r.raise_for_status();return r.json()


def _vwap(levels,units):
    left=units;spent=0.0
    for entry in levels:
        px=float(entry['px']);sz=float(entry['sz'])
        take=min(left,sz);spent+=take*px;left-=take
        if left<=1e-9:break
    if left>1e-9:return None
    return spent/units


def hyperliquid_market_snapshot(notional_usd=20000, asof=None):
    """Live NEAR perp order-book/funding context, not historical or options-market liquidity.

    Only call for contemporary snapshot dates. A single snapshot is NOT average
    hedge cost during the life of a 30-day option.
    """
    asof=pd.Timestamp(asof);asof=asof.tz_localize('UTC') if asof.tz is None else asof.tz_convert('UTC')
    if abs((pd.Timestamp.now(tz='UTC')-asof).total_seconds())>36*3600:
        raise RuntimeError('The pinned valuation date is not current; live order-book/funding would be anachronistic')
    with requests.Session() as s:
        meta=_hl_post(s,{'type':'metaAndAssetCtxs'})
        if not isinstance(meta,list) or len(meta)!=2:raise RuntimeError('Unexpected Hyperliquid meta')
        ids=[a.get('name') for a in meta[0].get('universe',[])]
        if 'NEAR' not in ids:raise RuntimeError('NEAR not listed on the native Hyperliquid perp universe')
        context=meta[1][ids.index('NEAR')]
        book=_hl_post(s,{'type':'l2Book','coin':'NEAR'})
        if len(book.get('levels',[]))!=2:raise RuntimeError('Unexpected Hyperliquid L2 book')
        bids,asks=book['levels']
        bid=float(bids[0]['px']);ask=float(asks[0]['px']);mid=(bid+ask)/2
        units=notional_usd/mid
        buy_vwap=_vwap(asks,units);sell_vwap=_vwap(bids,units)
        start_ms=int((asof-pd.Timedelta(days=30)).timestamp()*1000)
        end_ms=int(asof.timestamp()*1000)
        funding=_hl_post(s,{'type':'fundingHistory','coin':'NEAR','startTime':start_ms,'endTime':end_ms})
    out={'source':'Hyperliquid native NEAR perpetual; snapshot only','timestamp_observed_utc':pd.Timestamp.now(tz='UTC').isoformat(),
         'book_time_utc':pd.Timestamp(int(book['time']),unit='ms',tz='UTC').isoformat(),
         'best_bid':bid,'best_ask':ask,'best_mid':mid,
         'quoted_spread_bps':(ask-bid)/mid*10000,'rfq_hedge_notional_usd':notional_usd,
         'buy_vwap':buy_vwap,'sell_vwap':sell_vwap,
         'buy_impact_bps':(buy_vwap/mid-1)*10000 if buy_vwap else None,
         'sell_impact_bps':(1-sell_vwap/mid)*10000 if sell_vwap else None,
         'mark_px':context.get('markPx'),'oracle_px':context.get('oraclePx'),
         'funding_rate_now':context.get('funding'),
         'open_interest_contracts':context.get('openInterest'),
         'funding_history_count':len(funding) if isinstance(funding,list) else 0}
    if isinstance(funding,list) and funding:
        f=pd.DataFrame(funding)
        f['fundingRate']=pd.to_numeric(f['fundingRate'],errors='coerce')
        out['past_30d_funding_sum_pct']=float(100*f.fundingRate.dropna().sum())
        out['funding_is_forward_prediction']=False
    try:
        out['perp_basis_vs_oracle_bps']=10000*(float(context['markPx'])/float(context['oraclePx'])-1)
    except (ValueError,TypeError,KeyError,ZeroDivisionError): pass
    return out
