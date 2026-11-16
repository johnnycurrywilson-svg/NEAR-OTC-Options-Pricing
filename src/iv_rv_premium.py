"""Peer IV / trailing-RV calibration for a TWO-WAY OTC mid.

The contemporaneous IV/RV ratio is NOT a realised risk premium or an
options-selling signal. We only estimate fair mid, never an option side.

Units: IV/RV decimals annualised (0.70 means 70%); relative premium 0.10 means
10% of RV, while volatility POINTS are (IV-RV)*100.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import math
import numpy as np
import pandas as pd
import requests
from src.model import peer_at
from src.near_data import historical_volatility, _request, BASE

# Check actual market existence with Kraken, fail per asset if unavailable.
KRAKEN_PAIRS = {'HYPE': 'HYPEUSD', 'SOL': 'SOLUSD', 'ZEC': 'ZECUSD',
                'AVAX': 'AVAXUSD', 'TRX': 'TRXUSD'}
DAYS = 365


def fetch_peer_closes(asset: str, session=None, asof=None):
    """Get completed UTC daily closes, no today's unfinished OHLC candle."""
    if asset not in KRAKEN_PAIRS:
        raise ValueError('No configured Kraken pair for ' + str(asset))
    asof = asof or datetime.now(timezone.utc)
    own_session = session is None
    sess = session or requests.Session()
    try:
        data = _request(sess, 'OHLC', {'pair': KRAKEN_PAIRS[asset], 'interval': 1440})
    finally:
        if own_session:
            sess.close()
    keys = [key for key in data if key != 'last']
    if len(keys) != 1:
        raise ValueError(f'Expected one OHLC series for {asset}; got {keys}')
    rows = []
    for candle in data[keys[0]]:
        date = pd.Timestamp(int(candle[0]), unit='s', tz='UTC').normalize()
        if date.date() >= asof.date():
            continue
        px = float(candle[4])
        if np.isfinite(px) and px > 0:
            rows.append({'asset': asset, 'date': date, 'close': px})
    df = pd.DataFrame(rows)
    if df.empty or len(df.drop_duplicates('date')) < 32:
        raise RuntimeError(f'Only {len(df)} valid historical closes for {asset}; need >=32')
    df = df.drop_duplicates('date').sort_values('date').reset_index(drop=True)
    if not (df.date.diff().dropna().dt.days == 1).all():
        raise RuntimeError(f'Daily dates missing for {asset}; cannot calculate contiguous 30-day RV')
    return df


def normalize_custom_closes(df: pd.DataFrame):
    required = {'asset','date','close'}
    if not required.issubset(df.columns):
        raise ValueError('Peer closes CSV requires asset,date,close columns')
    x = df[list(required)].copy()
    x['asset'] = x['asset'].astype(str).str.upper().str.strip()
    x['date'] = pd.to_datetime(x['date'], utc=True).dt.normalize()
    x['close'] = pd.to_numeric(x['close'], errors='coerce')
    x = x.dropna().loc[lambda t:t.close > 0].sort_values(['asset','date'])
    x = x.drop_duplicates(['asset','date'],keep='last')
    return x


def realised_from_prices(prices: pd.DataFrame, windows=(7,30,90)):
    """Return by-asset trailing RV30 and additional windows for diagnostics."""
    rows=[]
    for asset,g in prices.groupby('asset'):
        g=g.sort_values('date')
        if not (g.date.diff().dropna().dt.days == 1).all():
            continue
        rv=historical_volatility(g,windows=tuple(w for w in windows if len(g)>w))
        if 30 not in rv.lookback_days.values: continue
        row={'asset':asset,'rv30':float(rv.loc[rv.lookback_days.eq(30),'realized_vol_decimal'].iloc[0]),
             'last_close_day':g.date.iloc[-1].date().isoformat(),
             'price_source':'Kraken completed UTC closes / supplied CSV',
             'close_observations':int(len(g))}
        for _,r in rv.iterrows():
            row[f'rv{int(r.lookback_days)}']=float(r.realized_vol_decimal)
        rows.append(row)
    return pd.DataFrame(rows)


def _has_30day_coverage(g:pd.DataFrame):
    ds=g['dte'].astype(float).to_numpy()
    return bool(len(ds) and np.min(np.abs(ds-30))<=20)


def make_peer_premium_table(fits:pd.DataFrame, peer_rv:pd.DataFrame, asof_utc:str):
    """Merge each asset ONCE after fitting each venue separately.

    Does not double-count HYPE or SOL. Reject markets without reasonable 30d
    tenor coverage, even if another tenor is technically extrapolatable.
    """
    if peer_rv.empty: return pd.DataFrame()
    rows=[]
    for asset in sorted(set(fits.asset).intersection(peer_rv.asset)):
        g=fits.loc[fits.asset.eq(asset)]
        if not any(_has_30day_coverage(v) for _,v in g.groupby('venue')):
            continue
        r=peer_rv.loc[peer_rv.asset.eq(asset)].iloc[-1]
        # Never compare TODAY's IV with stale realised volatility from another regime.
        if r.last_close_day != 'FICTIONAL':
            try:
                last_day=pd.Timestamp(r.last_close_day, tz='UTC')
                quote_day=pd.Timestamp(asof_utc)
                if quote_day.tz is None: quote_day=quote_day.tz_localize('UTC')
                if not 0 <= (quote_day.normalize()-last_day.normalize()).days <= 3:
                    continue
            except (ValueError,TypeError):
                continue
        rv=float(r.rv30)
        if not (np.isfinite(rv) and .02 < rv < 5): continue
        p=peer_at(fits,asset,30)
        iv=float(p['atm'])
        if not (.02 < iv < 5): continue
        rows.append({'asof_utc':asof_utc, 'asset':asset,
                     'venues':','.join(sorted(g.venue.unique())),
                     'atm30_iv':iv,'trailing30_rv':rv,
                     'iv_minus_rv_vol_points':(iv-rv)*100,
                     'iv_rv_ratio':iv/rv,
                     'relative_premium':iv/rv-1,
                     'log_iv_rv_ratio':math.log(iv/rv),
                     'quality_score':max(.01,min(1.,float(p['reliability']))),
                     'last_close_day':r.last_close_day,
                     'quote_status':'observed IV vs TRAILING RV (not future performance)'})
    return pd.DataFrame(rows)


def weighted_median(values,weights):
    x=np.asarray(values,dtype=float);w=np.asarray(weights,dtype=float)
    keep=np.isfinite(x)&np.isfinite(w)&(w>0)
    x=x[keep];w=w[keep]
    if not len(x): raise ValueError('No valid premium observations')
    order=np.argsort(x);x=x[order];w=w[order]
    return float(x[np.searchsorted(np.cumsum(w), w.sum()/2, side='left')])


def estimate_neutral_premium(table:pd.DataFrame, min_assets=2):
    """Quality-weighted median log(IV/RV), no positive-premium floor.

    Important: this is a CONTEMPORANEOUS peer IV/RV calibration heuristic.
    It is not an estimate of physical risk premium without forecast RV.
    """
    if table.empty or table.asset.nunique()<min_assets:
        raise ValueError(f'Need >= {min_assets} peer assets with 30-day IV and RV; got {0 if table.empty else table.asset.nunique()}')
    # Trim extremes (in log ratio) for robust fair-mid calibration only;
    # full untrimmed data retained in CSV. A cap is an explicit assumption.
    clipped=np.clip(table.log_iv_rv_ratio.to_numpy(dtype=float),math.log(.55),math.log(1.8))
    w=table.quality_score.to_numpy(dtype=float)
    median=weighted_median(clipped,w)
    return {'premium_ratio': math.expm1(median),
            'log_iv_rv_ratio':median,
            'n_assets':int(table.asset.nunique()),
            'method':'quality-weighted median log(IV30 / trailing RV30), capped ratio 0.55-1.8',
            'warning':'Contemporaneous relative value only; not an ex-ante risk premium or P&L forecast'}


def append_snapshots(path:Path, table:pd.DataFrame):
    """Save immutable timestamped observations for later forward-RV evaluation."""
    if table.empty:return
    path.parent.mkdir(parents=True,exist_ok=True)
    existing=pd.read_csv(path) if path.exists() else pd.DataFrame()
    data=(pd.concat([existing,table],ignore_index=True)
          if not existing.empty else table.copy())
    data=data.drop_duplicates(['asof_utc','asset'],keep='last')
    data.to_csv(path,index=False)


def evaluate_matured_snapshots(path:Path, prices:pd.DataFrame, asof=None):
    """Historical *subsequent* 30 calendar-day close-to-close RV; no lookahead.

    Uses quote-day UTC close as proxy for forward-window start, even if the
    quote occurred earlier that day. Only scores once subsequent day +30
    is a COMPLETED daily candle, and requires exact dates, 31 closes.
    """
    columns=['asof_utc','asset','atm30_iv','trailing30_rv','future30_rv',
             'iv_minus_future_rv_points','iv_minus_trailing_rv_points','evaluation_status']
    if not path.exists() or prices.empty:return pd.DataFrame(columns=columns)
    hist=pd.read_csv(path)
    if hist.empty:return pd.DataFrame(columns=columns)
    # Retain one observation per asset per day to avoid overcounting intraday refreshes.
    hist['snapshot_date']=pd.to_datetime(hist.asof_utc,utc=True).dt.normalize()
    hist=hist.sort_values('asof_utc').drop_duplicates(['asset','snapshot_date'],keep='last')
    lookup={(a,pd.Timestamp(dt).normalize()):float(close) for a,dt,close in prices[['asset','date','close']].itertuples(index=False,name=None)}
    rows=[]
    for r in hist.itertuples(index=False):
        start=pd.Timestamp(r.snapshot_date)
        if start.tz is None: start=start.tz_localize('UTC')
        dates=pd.date_range(start, periods=31, freq='D')
        px=[lookup.get((r.asset,dt)) for dt in dates]
        if any(p is None or p<=0 for p in px):
            continue
        rv=float(np.std(np.diff(np.log(px)),ddof=1)*np.sqrt(DAYS))
        rows.append({'asof_utc':r.asof_utc,'asset':r.asset,'atm30_iv':r.atm30_iv,
                     'trailing30_rv':r.trailing30_rv,'future30_rv':rv,
                     'iv_minus_future_rv_points':100*(r.atm30_iv-rv),
                     'iv_minus_trailing_rv_points':100*(r.atm30_iv-r.trailing30_rv),
                     'evaluation_status':'complete daily-close proxy; ex post, not hedge P&L'})
    return pd.DataFrame(rows,columns=columns)
