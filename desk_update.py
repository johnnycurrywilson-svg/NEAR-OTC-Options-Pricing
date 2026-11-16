"""Pinned 12-Nov-2026 NEAR client OTC case study, reproducible and two-sided.

Usage:
  python desk_update.py                       # uploaded 12 Nov frozen snapshot only
  python desk_update.py --fetch-extra         # try historical Coinbase/hourly and live Hyperliquid, with provenance
  python desk_update.py --fetch-extra --history-deribit   # expensive historical trade backfill
  python desk_update.py --snapshot outputs/model_run.json # use own newer snapshot

Refuses to silently replace history with synthetic or modern trading data.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from src.model import black76
from src.desk_calibration import forecast_near_rv,rolling_forward_rv_backtest,historical_iv_vs_future_rv
from src.research_inputs import coinbase_candles,hourly_diagnostics,hyperliquid_market_snapshot
from src.extra_research import build_supplemental_scenarios

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
PINNED=ROOT/'data'/'snapshot_2026-11-12.json'


def estimate_two_way(spot, strike, right, fair_iv, days=30, execution_bps=15., hedges=30,
                     uncertainty_pts=1.0,jump_pts=1.0):
    """Cost-aware neutral two-sided Black-76 quote; no directional vol position."""
    t=days/365
    model=black76(spot,strike,t,fair_iv,right,0)
    delta=abs(model['forward_delta']); gamma=model['forward_gamma'];vega=model['vega_per_1volpt']
    dyn_turnover=gamma*spot*fair_iv*math.sqrt(2*t/math.pi)*math.sqrt(hedges)
    gross_turnover=delta+dyn_turnover
    cost_usd=spot*gross_turnover*execution_bps/10000
    cost_pts=cost_usd/vega
    half=cost_pts+uncertainty_pts+jump_pts
    if fair_iv-half/100<.05:raise ValueError('Bid IV below 5%; unsupported risk parameters')
    bid_iv=fair_iv-half/100;ask_iv=fair_iv+half/100
    return {'strike':strike,'right':right,'fair_iv_pct':fair_iv*100,
            'bid_iv_pct':bid_iv*100,'offer_iv_pct':ask_iv*100,
            'bid_usd':black76(spot,strike,t,bid_iv,right)['price'],
            'fair_usd':model['price'],'offer_usd':black76(spot,strike,t,ask_iv,right)['price'],
            'delta':model['forward_delta'],'gamma':gamma,'vega_usd_per_vol_point':vega,
            'execution_cost_usd_per_option_per_side':cost_usd,
            'execution_vol_points_per_side':cost_pts,
            'uncertainty_vol_points_per_side':uncertainty_pts,
            'jump_vol_points_per_side':jump_pts,
            'total_half_spread_vol_points':half,'total_spread_vol_points':2*half,
            'expected_gross_delta_hedge_turnover_near_per_option':gross_turnover,
            'execution_bps_per_hedge':execution_bps,'assumed_hedges_per_month':hedges,
            'estimated_not_measured':True}


def nonoverlap_scores(prices):
    """Non-overlapping evaluation dates from original lookahead-safe model.

    Descriptive only: a 30-day target with a 30-day stride has few observations.
    Never use this data to auto switch pricing method.
    """
    from src.vol_forecasting import compare_forecasts
    predictions,scores,summary=compare_forecasts(prices)
    predictions['date_dt']=pd.to_datetime(predictions.forecast_date_utc,utc=True)
    results=[]
    for seg in ('selection','holdout'):
        part=predictions[predictions.segment==seg]
        dates=sorted(part.date_dt.unique()); chosen=[]
        for day in dates:
            if not chosen or (day-chosen[-1]).days>=30:chosen.append(day)
        subset=part[part.date_dt.isin(chosen)]
        for method,g in subset.groupby('method'):
            results.append({'segment':seg,'method':method,'n_nonoverlap':len(g),
                            'mae_vol_points':float(g.absolute_error_vol_points.mean()),
                            'bias_vol_points':float(g.error_vol_points.mean()),
                            'note':'descriptive; one chronological split and small nonoverlapping sample'})
    return pd.DataFrame(results),summary,scores


def get_history(m,fetch=False):
    result={'historical_daily':'not_requested','historical_hourly':'not_requested',
            'perp_book_and_funding':'not_requested'}
    if not fetch:return result
    asof=pd.Timestamp(m['asof']).tz_convert('UTC')
    # completed daily close at Nov 11 for frozen Nov 12 snapshot. No later candles.
    end=(asof.floor('D'))
    for label,granularity,days,filename in [
        ('historical_daily',86400,1095,'near_extended_daily_closes.csv'),
        ('historical_hourly',3600,180,'near_hourly_closes.csv')]:
        try:
            df=coinbase_candles(end-pd.Timedelta(days=days),end,granularity)
            df.to_csv(OUT/filename,index=False)
            result[label]={'status':'ok','source':'Coinbase NEAR-USD','observations':int(len(df)),
                           'first_date':str(df.date.iloc[0]),'last_date':str(df.date.iloc[-1]),
                           'file':filename}
            if label=='historical_hourly':result['hourly_rv']=hourly_diagnostics(df)
            if label=='historical_daily':
                scores,model_info,weekly=nonoverlap_scores(df[['date','close']])
                scores.to_csv(OUT/'forecast_nonoverlap_scores.csv',index=False)
                weekly.to_csv(OUT/'forecast_long_history_weekly_scores.csv',index=False)
                holdout=scores[scores.segment=='holdout']
                clean_scores=[{'model':str(r.method),'n':int(r.n_nonoverlap),
                               'mae_vol_pts':round(float(r.mae_vol_points),2),
                               'bias_vol_pts':round(float(r.bias_vol_points),2)}
                              for r in holdout.itertuples(index=False)]
                result['extended_forecast_validation']={'status':'ok',
                    'n_nonoverlap_holdout':int(holdout.n_nonoverlap.max()),
                    'candidate_selected_early':model_info['selected_method'],
                    'holdout_scores':clean_scores,
                    'pricing_policy':'fixed blend until independently validated; never auto-adopt HAR'}
        except Exception as exc:
            result[label]={'status':'unavailable','reason':str(exc)[:280]}
    try:
        context=hyperliquid_market_snapshot(notional_usd=20000,asof=asof)
        result['perp_book_and_funding']={'status':'ok',**context}
        (OUT/'perp_hedge_context.json').write_text(json.dumps(context,indent=2))
    except Exception as exc:
        result['perp_book_and_funding']={'status':'unavailable','reason':str(exc)[:280]}
    return result


def backfill_archive(m,history_days=365):
    """Deribit trade tape, not historical chain midpoint; only matured observations."""
    from src.historical_deribit import backfill_history
    obs,audit,detail=backfill_history(m['asof'],lookback_days=history_days,step_days=7)
    audit.to_csv(OUT/'historical_trade_fetch_audit.csv',index=False)
    if not obs.empty:obs.to_csv(OUT/'historical_trade_iv_observations.csv',index=False)
    if not detail.empty:detail.to_csv(OUT/'historical_qualified_trade_detail.csv',index=False)
    peer=OUT/'peer_daily_closes.csv'
    if not obs.empty and peer.exists():
        evaluated=historical_iv_vs_future_rv(obs,pd.read_csv(peer))
        evaluated.to_csv(OUT/'historical_trade_iv_vs_future_rv.csv',index=False)
        n=len(evaluated)
    else:n=0
    return {'attempted':True,'qualifying_trade_iv_obs':int(len(obs)),
            'n_forward_rv_evaluated':int(n),'complete_windows':int((audit.status=='complete').sum()),
            'rejected_windows':int((audit.status!='complete').sum()),
            'note':'Trade-derived IV proxy; evaluate only if peer_daily_closes.csv exists; no executable mid'}


def build_review(snapshot_path=PINNED,fetch_extra=False,do_history=False):
    OUT.mkdir(exist_ok=True)
    m=json.loads(Path(snapshot_path).read_text())
    if m.get('mode')!='live':raise ValueError('A live source snapshot is required')
    asof=pd.Timestamp(m['asof'])
    if asof.tzinfo is None:raise ValueError('Timezone-aware market snapshot required')
    if asof.date().isoformat() != '2026-11-12':
        raise ValueError('This case study is locked to 12 Nov 2026; do not label another day as 12 Nov')
    spot=float(m['near_spot']); premium=float(m['vrp_ratio'])
    rv_inputs={int(k):float(v) for k,v in m['forward_forecast']['rv_inputs'].items()}
    fixed=forecast_near_rv(rv_inputs)['forecast_rv']
    fair_atm=fixed*(1+premium)
    selected=float(m['near_forward_rv_30']); challenger={'har_rv_pct':100*selected,'policy':'experimental, not auto-priced'}
    options={}; tca=[]
    for name,origin in m['option_tickets'].items():
        iv=fair_atm+(float(origin['iv'])-float(m['near_atm_iv_30']))
        options[name]=estimate_two_way(spot,float(origin['strike']),origin['right'],iv)
    for label,bps,hedges in [('Low friction',8,12),('Base case',15,30),('Stressed',30,60)]:
        item=estimate_two_way(spot,spot,'C',fair_atm,execution_bps=bps,hedges=hedges)
        tca.append({'scenario':label,'execution_bps_per_hedge':bps,'rehedges_30d':hedges,
                    'spread_vol_points':item['total_spread_vol_points'],
                    'dealer_bid_usd':item['bid_usd'],'dealer_offer_usd':item['offer_usd'],
                    'execution_cost_usd_per_option_per_side':item['execution_cost_usd_per_option_per_side']})
    extra=get_history(m,fetch_extra)
    supplemental=build_supplemental_scenarios(extra,atm_spot=spot,atm_fair_iv=fair_atm,
                                              quote_func=estimate_two_way)
    if 'book_informed_tca' in supplemental:
        bk=supplemental['book_informed_tca']
        tca.append({'scenario':'Perp book (single snapshot)','execution_bps_per_hedge':bk['cost_bps_per_rehedge'],
                    'rehedges_30d':30,'spread_vol_points':bk['full_spread_vol_pts'],
                    'dealer_bid_usd':bk['atm_bid_usd'],'dealer_offer_usd':bk['atm_offer_usd'],
                    'execution_cost_usd_per_option_per_side':None,
                    'note':'Measured one-book impact + assumed 5 bps fees + 5 bps slippage, not executable'} )
    archive=backfill_archive(m) if do_history else {'attempted':False,'reason':'Optional historical backfill not run; 11 previously qualified observations retained as research context'}
    summary={'asof_utc':m['asof'],'research_basis':'12 Nov 2026 recorded live source snapshot; no new invented IV',
             'spot':spot,'rv_7':rv_inputs.get(7),'rv_30':rv_inputs.get(30),'rv_90':rv_inputs.get(90),
             'fixed_forward_rv':fixed,'peer_relative_adjustment':premium,
             'fair_atm_iv':fair_atm,'experimental_har':challenger,
             'all_forecast_models':m.get('near_forecast_model_comparison',{}).get('live_model_forecasts',{}),
             'live_peer_iv':[{k:r[k] for k in ['asset','atm30_iv','trailing30_rv','quality_score']} for r in m['premium_peer_rows']],
             'ticket_quotes':options,'tca':tca,'extra_research':extra,
             'supplemental_assessment':supplemental,'historical_trade_archive':archive,
             'historical_iv_future_rv_reference':m['historical_iv_forward_evaluation'],
             'model_validation_warning':'25 overlapping holdout outcomes, HAR bias, no independent validation; no automatic HAR deployment',
             'pricing_assumptions':'Black-76 European USD cash settlement, zero forward basis/carry and interest, delta-neutral inventory, trading friction 15 bps/rebalance and 30 hedge trades/month',
             'wing_calibration':'Wing IV offsets from the 12 Nov exported surface are shifted to the fixed-blend ATM; no fresh SVI refit in offline mode'}
    (OUT/'desk_research_summary.json').write_text(json.dumps(summary,indent=2))
    pd.DataFrame([{'name':name,**r} for name,r in options.items()]).to_csv(OUT/'desk_two_way_quotes.csv',index=False)
    pd.DataFrame(tca).to_csv(OUT/'transaction_cost_scenarios.csv',index=False)
    if 'book_informed_tca' in supplemental:
        pd.DataFrame([supplemental['book_informed_tca']]).to_csv(OUT/'perp_book_tca_scenario.csv',index=False)
    if supplemental.get('extended_holdout_scores'):
        pd.DataFrame(supplemental['extended_holdout_scores']).to_csv(OUT/'extended_holdout_summary.csv',index=False)
    from make_desk_pdf import create_pdf
    create_pdf(summary,OUT/'NEAR_12Nov_Client_OTC_Pricing_Study.pdf')
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',type=Path,default=PINNED,help='Use the exact live snapshot to study')
    p.add_argument('--fetch-extra',action='store_true',help='Fetch historical hourly/daily NEAR and live perp TCA if date aligned')
    p.add_argument('--history-deribit',action='store_true',help='Optional expensive backfill for historical traded-IV proxy')
    a=p.parse_args()
    r=build_review(a.snapshot,a.fetch_extra,a.history_deribit)
    print('12-Nov snapshot:' if '2026-11-12' in r['asof_utc'] else 'Snapshot:',r['asof_utc'])
    print('Baseline NEAR forecast RV / fair IV: {:.1f}% / {:.1f}%'.format(100*r['fixed_forward_rv'],100*r['fair_atm_iv']))
    print('HAR research only:',r['experimental_har'])
    print('ATM two-way:',r['ticket_quotes']['ATM call']['bid_usd'],r['ticket_quotes']['ATM call']['offer_usd'])
    print('Extra research:',{k:(v.get('status') if isinstance(v,dict) else v) for k,v in r['extra_research'].items()})
    print('Report:',OUT/'NEAR_12Nov_Client_OTC_Pricing_Study.pdf')
if __name__=='__main__':main()
