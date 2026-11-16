"""Historical Deribit USDC options trades -> conservative near-ATM ~30d IV observations.

PUBLIC trade-tape method, NOT historical mark-IV chain. Does not use trades from
unrelated coins, does not fabricate days with no qualified option trades, and
never infer an executable mid from transaction prices. A deliberate optional
one-off backfill because paging a busy public USDC tape can be expensive.

See docs: https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import math
import re
import time
import pandas as pd
import numpy as np
import requests

BASE='https://history.deribit.com/api/v2/public'
TARGET_ASSETS=('AVAX','HYPE','SOL','TRX')
NAME_RE=re.compile(r'^(AVAX|HYPE|SOL|TRX)_USDC-(\d{1,2}[A-Z]{3}\d{2})-([\d.]+)-([CP])$')


def _date(ms):
    return pd.Timestamp(int(ms),unit='ms',tz='UTC')


def _iv_decimal(value):
    """Handle Deribit trade IV decimal and, defensively, percent-valued feeds."""
    try: x=float(value)
    except (ValueError,TypeError):return None
    if not math.isfinite(x) or x<=0:return None
    if 5 < x <= 600:x/=100
    return x if 0.02 <= x <= 5 else None


def parse_trade(trade, asof_time=None, min_days=18, max_days=45,
                max_abs_log_moneyness=.13):
    """Filter to OTM-ish instruments near ATM, expiry near 30d, with historical underlying ref."""
    if not isinstance(trade,dict):return None
    m=NAME_RE.fullmatch(str(trade.get('instrument_name','')))
    if not m:return None
    try:
        trade_at=_date(trade['timestamp'])
        expiration=pd.Timestamp(datetime.strptime(m.group(2),'%d%b%y').replace(tzinfo=timezone.utc))+pd.Timedelta(hours=8)
        dte=(expiration-trade_at).total_seconds()/86400.
        k=float(m.group(3));underlying=float(trade.get('index_price') or trade.get('underlying_price'))
        iv=_iv_decimal(trade.get('iv'))
        qty=float(trade.get('amount',1))
        if iv is None or not all(map(math.isfinite,[k,underlying,dte,qty])) or min(k,underlying,qty)<=0:return None
        if not min_days<=dte<=max_days:return None
        z=math.log(k/underlying)
        if abs(z)>max_abs_log_moneyness:return None
        # Avoid mixing deeply in-the-money calls/puts with near ATM vols.
        if m.group(4)=='C' and z < -.025:return None
        if m.group(4)=='P' and z > .025:return None
        return {'asset':m.group(1),'asof_utc':trade_at.isoformat(),
                'snapshot_date':trade_at.date().isoformat(),
                'instrument':m.group(0),'right':m.group(4), 'dte_days':dte,
                'strike':k,'index_price_at_trade':underlying,
                'iv':iv,'size':qty,'log_moneyness':z,
                'venue':'deribit', 'source_type':'historical_trade_iv_not_mid',
                'trade_id':str(trade.get('trade_id',''))}
    except (ValueError,TypeError,OverflowError,KeyError):return None


def _call_history(session,params,attempts=3):
    last=None
    for i in range(attempts):
        try:
            r=session.get(BASE+'/get_last_trades_by_currency_and_time',params=params,timeout=28)
            r.raise_for_status()
            payload=r.json()
            if payload.get('error'):raise RuntimeError('Deribit history API: '+str(payload['error']))
            result=payload.get('result')
            if not isinstance(result,dict) or not isinstance(result.get('trades'),list):
                raise ValueError('Unexpected history API response; expected result.trades list')
            return result
        except (requests.RequestException, ValueError, RuntimeError) as ex:
            last=ex
            if i<attempts-1:time.sleep(1.5*(i+1))
    raise RuntimeError(str(last))


def fetch_trade_window(start_utc,end_utc,session=None,max_pages=7,page_size=1000,
                       pause_seconds=.12,request_function=None):
    """Fetch an ENTIRE window; reject if limit hit so no partial-history bias.

    Deribit documented limit is 10,000/page; 1000 is conservative. We page by
    timestamp using ascending results. A capped/ambiguous page is marked
    incomplete and contributes no observations.
    """
    start=pd.Timestamp(start_utc).tz_convert('UTC') if pd.Timestamp(start_utc).tzinfo else pd.Timestamp(start_utc).tz_localize('UTC')
    end=pd.Timestamp(end_utc).tz_convert('UTC') if pd.Timestamp(end_utc).tzinfo else pd.Timestamp(end_utc).tz_localize('UTC')
    if not end>start:raise ValueError('end must follow start')
    if max_pages<1 or not 1<=page_size<=10000:raise ValueError('Invalid pagination settings')
    begin_ms=int(start.timestamp()*1000)
    end_ms=int(end.timestamp()*1000)
    fn=request_function or _call_history
    own=session is None
    session=session or requests.Session()
    all_trades=[];seen=set();pages=0;status='complete'
    try:
        for page in range(max_pages):
            params={'currency':'USDC','kind':'option','start_timestamp':begin_ms,
                    'end_timestamp':end_ms,'count':page_size,'include_old':'true','sorting':'asc'}
            result=fn(session,params)
            trades=result['trades'];pages+=1
            if not trades:break
            timestamps=[]
            for tr in trades:
                stamp=int(tr.get('timestamp',-1));timestamps.append(stamp)
                if not begin_ms<=stamp<=end_ms:
                    status='invalid_timestamp';break
                identifier=str(tr.get('trade_id') or (stamp,tr.get('instrument_name'),tr.get('price'),tr.get('amount')))
                if identifier not in seen:seen.add(identifier);all_trades.append(tr)
            if status!='complete':break
            if timestamps!=sorted(timestamps):
                status='unsorted_api_response';break
            # Deribit has_more + page length can signal more records. Never
            # assume a max-size final page is complete unless has_more false.
            has_more=result.get('has_more')
            if has_more is False or (len(trades)<page_size and has_more is not True):break
            last_ms=timestamps[-1]
            if last_ms>=end_ms:break
            if last_ms<begin_ms:
                status='pagination_stalled';break
            # timestamp cursor can discard trades tied at the same ms. Reject
            # windows that end on an ambiguous all-full timestamp boundary.
            if sum(x==last_ms for x in timestamps)>1 and len(trades)>=page_size:
                status='timestamp_cursor_ambiguous';break
            begin_ms=last_ms+1
            if page==max_pages-1:status='page_cap_incomplete';break
            if pause_seconds:time.sleep(pause_seconds)
        else:status='page_cap_incomplete'
    finally:
        if own:session.close()
    if status!='complete':return [],{'status':status,'pages':pages,'count':len(all_trades)}
    return all_trades,{'status':status,'pages':pages,'count':len(all_trades)}


def build_atm_observations(trades,reference_day,min_trades=3,min_instruments=2,
                           nearest_maturity_days=8):
    """Trade-weighted robust daily ATM-IV proxy from near-ATM traded contracts.

    Use *one observation per asset per sampled UTC date*. A 30d estimate can
    be interpolated in total variance when two expiries bracket day 30;
    otherwise only accept maturity within nearest_maturity_days.
    """
    parsed=[parse_trade(t) for t in trades]
    parsed=[t for t in parsed if t and t['snapshot_date']==str(reference_day)]
    if not parsed:return pd.DataFrame(),pd.DataFrame()
    detail=pd.DataFrame(parsed).drop_duplicates(['instrument','trade_id','asof_utc'])
    rows=[]
    for asset,group in detail.groupby('asset'):
        if len(group)<min_trades or group.instrument.nunique()<min_instruments:continue
        # Weighted median of near-ATM IV across trades at each expiry.
        # Weight decreasing with |log moneyness|, and capped size to avoid
        # one huge block overwhelming the price-discovery estimate.
        g=group.copy();g['expiration_days']=g.instrument.str.extract(r'_USDC-(\d{1,2}[A-Z]{3}\d{2})-')[0]
        grouped=[]
        for exp,part in g.groupby('expiration_days'):
            if len(part)<2 or part.instrument.nunique()<2:continue
            w=1/(.03+abs(part.log_moneyness.to_numpy(dtype=float)))
            # Winsorise trade sizes; do not privilege large block prints.
            w*=np.sqrt(np.clip(part['size'].to_numpy(dtype=float),.01,100.))
            values=part.iv.to_numpy(dtype=float)
            ix=np.argsort(values)
            iv=float(values[ix][np.searchsorted(np.cumsum(w[ix]),w.sum()/2,side='left')])
            dte=float(np.average(part.dte_days.to_numpy(dtype=float),weights=w))
            if math.isfinite(iv):grouped.append((dte,iv,len(part),part.instrument.nunique()))
        if not grouped:continue
        grouped.sort(key=lambda x:x[0]);lower=[p for p in grouped if p[0]<=30];upper=[p for p in grouped if p[0]>=30]
        chosen=[];est=None;method=None
        if lower and upper and lower[-1] != upper[0] and 30-lower[-1][0]<=12 and upper[0][0]-30<=12:
            lo,hi=lower[-1],upper[0]
            # Total variance w(T) = IV² T interpolation
            v0=lo[1]**2*lo[0]/365;v1=hi[1]**2*hi[0]/365
            wt=v0+(v1-v0)*(30-lo[0])/(hi[0]-lo[0]);est=math.sqrt(max(0,wt)*365/30)
            chosen=[lo,hi];method='total_variance_interpolated_trade_iv'
        else:
            closest=min(grouped,key=lambda p:abs(p[0]-30))
            if abs(closest[0]-30)<=nearest_maturity_days:
                est=closest[1];chosen=[closest];method='nearest_tenor_trade_iv_proxy'
        if est is None or not .03<est<5:continue
        rows.append({'asset':asset,'asof_utc':g['asof_utc'].max(),
                     'atm30_iv':est,'source':'Deribit historical option trades',
                     'proxy_method':method, 'n_trades':sum(p[2] for p in chosen),
                     'n_unique_instruments':sum(p[3] for p in chosen),
                     'tenor_range_days':','.join(f'{p[0]:.1f}' for p in chosen),
                     'quote_quality':'near-ATM actual trades, NOT mid marks',
                     'source_window_day':str(reference_day)})
    return pd.DataFrame(rows),detail


def backfill_history(asof_utc,lookback_days=150,step_days=7,window_hours=6,
                     max_pages=7,page_size=1000,request_function=None,
                     min_trades=3,min_instruments=2):
    """Sample weekly historical closes, each requiring a complete queried trade tape.

    Skip dates whose subsequent 30-day RV could not have fully realised yet.
    Always return auditable source/date rows, including days with no usable IV.
    """
    asof=pd.Timestamp(asof_utc)
    asof=asof.tz_convert('UTC') if asof.tzinfo else asof.tz_localize('UTC')
    if lookback_days<40 or lookback_days>730:raise ValueError('lookback_days must be 40-730')
    if not 1<=step_days<=31 or not 1<=window_hours<=24:raise ValueError('Bad sample interval')
    first=(asof.normalize()-pd.Timedelta(days=lookback_days))
    last=(asof.normalize()-pd.Timedelta(days=32))
    observations=[];audit=[];trade_rows=[]
    # One shared session, managed across the entire request batch.
    with requests.Session() as session:
        for day in pd.date_range(first,last,freq=f'{step_days}D'):
            end=day+pd.Timedelta(days=1)-pd.Timedelta(milliseconds=1)
            start=end-pd.Timedelta(hours=window_hours)
            try:
                trades,meta=fetch_trade_window(start,end,session=session,max_pages=max_pages,
                      page_size=page_size,request_function=request_function)
                if meta['status']=='complete':
                    obs,qualified=build_atm_observations(trades,day.date(),min_trades,min_instruments)
                    if not obs.empty:observations.append(obs)
                    if not qualified.empty:trade_rows.append(qualified)
                    accepted=[] if obs.empty else obs.asset.tolist()
                    audit.append({'sample_day':str(day.date()),'status':'complete',
                                  'trades_fetched':meta['count'],'pages':meta['pages'],
                                  'qualified_assets':','.join(accepted),
                                  'qualified_observations':len(accepted),'error':''})
                else:
                    audit.append({'sample_day':str(day.date()),'status':meta['status'],
                                  'trades_fetched':meta['count'],'pages':meta['pages'],
                                  'qualified_assets':'','qualified_observations':0,'error':''})
            except Exception as exc:
                # Clear audit, not a successful empty day.
                audit.append({'sample_day':str(day.date()),'status':'api_error',
                              'trades_fetched':0,'pages':0,'qualified_assets':'',
                              'qualified_observations':0,'error':str(exc)[:250]})
    out=pd.concat(observations,ignore_index=True) if observations else pd.DataFrame(columns=['asset','asof_utc','atm30_iv'])
    return out,pd.DataFrame(audit),pd.concat(trade_rows,ignore_index=True) if trade_rows else pd.DataFrame()
