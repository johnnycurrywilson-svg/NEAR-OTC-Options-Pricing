"""One-command refreshed market and NEAR OTC research output.

Runs the existing live Derive/Deribit/Kraken model; does not substitute stale/synthetic
quotes when an exchange or NEAR historical feed is unavailable. Archives the
previous model snapshot and creates a dated changes-vs-baseline audit.
"""
from __future__ import annotations
import json, sys, subprocess, shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
BASE=ROOT/'data'/'reference_2026-10-30.json'

def enrich(meta, prices, base):
    now=meta.get('asof')
    if not now or meta.get('mode')!='live':
        raise RuntimeError('Expected a genuine current live model run, not synthetic demo inputs.')
    prices=prices.copy()
    prices['date']=pd.to_datetime(prices['date'],utc=True,errors='raise')
    prices=prices.sort_values('date').drop_duplicates('date')
    prices['close']=pd.to_numeric(prices['close'],errors='raise')
    if prices['close'].le(0).any():raise RuntimeError('Non-positive history prices')
    if prices['date'].diff().dropna().dt.days.ne(1).any():raise RuntimeError('Missing daily closes')
    last=prices['date'].max();asof=pd.Timestamp(now)
    if (asof-last).total_seconds()>3*86400:
        raise RuntimeError('NEAR history is stale (>3 days); refusing market refresh')
    spot=float(meta['near_spot']); recent={}
    for days in (1,7,30,90):
        target=(asof-pd.Timedelta(days=days)).floor('D')
        rows=prices.loc[prices['date']<=target,'close']
        if not rows.empty:
            recent[f'return_{days}d_pct']=float(100*(spot/float(rows.iloc[-1])-1))
    historical_spot=float(base['near_spot'])
    change=float(100*(spot/historical_spot-1))
    baseline={r['asset']:r for r in base.get('premium_peer_rows',[])}
    comparisons=[]
    for p in meta.get('premium_peer_rows',[]):
        old=baseline.get(p['asset'])
        if not old:continue
        comparisons.append({'asset':p['asset'],
                            'current_atm30_iv_pct':100*p['atm30_iv'],
                            'reference_atm30_iv_pct':100*old['atm30_iv'],
                            'change_iv_vol_pts':100*(p['atm30_iv']-old['atm30_iv']),
                            'source_note':'peer ATM derived from asset-level fitted Derive/Deribit sources; quote mix can differ'})
    meta['market_update']={
        'type':'full_live_refresh','spot_asof_utc':meta.get('kraken_history',{}).get('spot_asof_utc',now),
        'last_complete_close_utc':str(last.date()),'returns':recent,
        'reference_date_utc':base['asof'],'reference_spot':historical_spot,
        'spot_change_from_reference_pct':change,
        'reference_near_atm30_iv':base['near_atm_iv_30'],
        'iv_change_from_reference_vol_pts':100*(meta['near_atm_iv_30']-base['near_atm_iv_30']),
        'reference_near_forward_rv30':base['near_forward_rv_30'],
        'rv_forecast_change_vol_pts':100*(meta['near_forward_rv_30']-base['near_forward_rv_30']),
        'peer_iv_comparison':comparisons,
        'cautions':'Peer composition and options quote quality can change. Historical 30 Oct inputs are frozen reference, not current marks.'}
    return meta

def plot_market(prices,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    prices=prices.copy();prices['date']=pd.to_datetime(prices['date'],utc=True)
    tail=prices.sort_values('date').tail(75)
    fig,ax=plt.subplots(figsize=(9,2.55))
    ax.plot(tail.date,tail.close,color='#31546a',lw=1.8)
    ax.set_ylabel('NEAR / USD',fontsize=10)
    ax.grid(alpha=.18); ax.spines[['right','top']].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.autofmt_xdate(rotation=0);fig.tight_layout()
    fig.savefig(output,dpi=150,bbox_inches='tight');plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    baseline=json.loads(BASE.read_text(encoding='utf-8'))
    existing=OUT/'model_run.json'
    if existing.exists():
        previous=json.loads(existing.read_text(encoding='utf-8'))
        if previous.get('mode')=='live':
            dt=str(previous.get('asof','unknown')).replace(':','-').replace('+','_')[:19]
            arc=ROOT/'data'/'prior_live_runs';arc.mkdir(parents=True,exist_ok=True)
            shutil.copy2(existing,arc/f'{dt}.json')
    passthrough=sys.argv[1:]
    cmd=[sys.executable,str(ROOT/'run.py'),'--mode','live','--venues','derive,deribit','--near-auto-history','--pdf']+passthrough
    print('Fetching fresh Derive, Deribit and Kraken inputs (no synthetic fallback).',flush=True)
    subprocess.run(cmd,check=True)
    meta=json.loads(existing.read_text(encoding='utf-8'))
    if not (str(meta['venue_status'].get('derive','')).startswith('OK:') and str(meta['venue_status'].get('deribit','')).startswith('OK:')):
        raise RuntimeError('One venue missing. Existing partial report remains, but refreshed desk report is not certified.')
    prices=pd.read_csv(OUT/'near_daily_closes.csv')
    meta=enrich(meta,prices,baseline)
    (OUT/'model_run.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    m=meta['market_update']
    pd.DataFrame([{'asof':meta['asof'],'spot':meta['near_spot'],**m['returns'],
                   'change_vs_oct30_pct':m['spot_change_from_reference_pct'],
                   'model_atm_iv_pct':meta['near_atm_iv_30']*100,
                   'forecast_rv_pct':meta['near_forward_rv_30']*100}]).to_csv(OUT/'market_movement_summary.csv',index=False)
    pd.DataFrame(m['peer_iv_comparison']).to_csv(OUT/'peer_iv_changes_since_oct30.csv',index=False)
    plot_market(prices,OUT/'04_near_recent_price.png')
    subprocess.run([sys.executable,str(ROOT/'make_pdf.py'),'--out',str(OUT)],check=True)
    print('DONE: Refreshed market, surfaces, volatility, two-way quotes and three-page PDF.')
    print('Market movement:',m['returns'])
    print('Output:',OUT/'NEAR_OTC_Case_Study_LIVE.pdf')
if __name__=='__main__':main()
