"""Market-neutral research: peer premium calibration, forward RV forecasting,
lookahead-safe historical diagnostics and cost-aware two-way OTC width.

All cash costs and jump/uncertainty reserves are CONFIGURABLE ASSUMPTIONS,
not observed executable quotes. No systematic long/short volatility bias.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from src.model import black76

WINDOW_WEIGHTS={7:0.20, 30:0.45, 90:0.35}

def _validate_vol(x):
    x=float(x)
    if not np.isfinite(x) or not 0.02 < x < 6: raise ValueError('Bad annualized volatility')
    return x

def forecast_near_rv(window_rv, weights=WINDOW_WEIGHTS):
    """Transparent, fixed-weight blend of VARIANCES; deliberately not a fitted ML forecast."""
    found={int(k):_validate_vol(v) for k,v in window_rv.items() if int(k) in weights}
    if 30 not in found:raise ValueError('Forecast requires RV30')
    w={k:v for k,v in weights.items() if k in found}
    total=sum(w.values())
    rv=math.sqrt(sum((v/total)*found[k]**2 for k,v in w.items()))
    return {'forecast_rv':rv,'weights':{str(k):v/total for k,v in w.items()},
            'rv_inputs':{str(k):found[k] for k in found},
            'method':'fixed 7/30/90-day blend of trailing variances (research heuristic; test OOS)',
            'assumptions':'weights are fixed subjective coefficients, not estimated from option market'}

def rolling_forward_rv_backtest(closes, step_days=7, horizon=30):
    """At each forecast day use ONLY closes through day t; score realised t..t+30.

    Weekly overlapping future returns; metrics describe forecast validation, NOT IV performance.
    """
    x=closes.copy().sort_values('date')
    if 'asset' in x: x=x.loc[x.asset.eq('NEAR')].copy()
    x['date']=pd.to_datetime(x['date'],utc=True)
    x=x.drop_duplicates('date').sort_values('date')
    if len(x)<125:return pd.DataFrame()
    if not (x.date.diff().dropna().dt.days==1).all():return pd.DataFrame()
    p=x.close.to_numpy(dtype=float)
    if not np.all(np.isfinite(p)) or np.any(p<=0):return pd.DataFrame()
    lr=np.diff(np.log(p)); rows=[]
    for end in range(90,len(p)-horizon,step_days):
        hist={w:float(np.std(lr[end-w:end],ddof=1)*np.sqrt(365)) for w in WINDOW_WEIGHTS}
        f=forecast_near_rv(hist)['forecast_rv']
        future=float(np.std(lr[end:end+horizon],ddof=1)*np.sqrt(365))
        rows.append({'forecast_date_utc':x.date.iloc[end].date().isoformat(),
                     'forecast_rv30':f,'subsequent_rv30':future,
                     'error_vol_points':100*(f-future),'absolute_error_vol_points':100*abs(f-future),
                     'window_days':horizon,'is_iv_backtest':False})
    return pd.DataFrame(rows)

def _huber_location(x,w,k=0.125,iterations=50):
    """Huber location in log(IV/RV), fixed 12.5-log-point clipping.

    Small samples have wide uncertainty. Location can be negative.
    """
    x=np.asarray(x,dtype=float);w=np.asarray(w,dtype=float)
    keep=np.isfinite(x)&np.isfinite(w)&(w>0)
    x,w=x[keep],w[keep];w=w/w.sum()
    if len(x)<2:raise ValueError('Need at least two peer assets')
    m=float(np.median(x))
    for _ in range(iterations):
        res=x-m
        rw=w*np.minimum(1,k/np.maximum(np.abs(res),1e-12))
        nxt=float(np.dot(rw,x)/rw.sum())
        if abs(nxt-m)<1e-12:break
        m=nxt
    return m

def robust_peer_premium(table, huber_clip_log=.125):
    """Return central estimate, leave-one-asset-out diagnostics, no bias to positive IV/RV."""
    t=table.copy().drop_duplicates('asset')
    if len(t)<3:raise ValueError('Require >=3 distinct peers for robust estimate and leave-one-out')
    t=t.sort_values('asset')
    x=np.log(t.atm30_iv.astype(float).to_numpy()/t.trailing30_rv.astype(float).to_numpy())
    w=np.maximum(0.01,t.quality_score.astype(float).to_numpy())
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(w)):raise ValueError('Nonfinite peer inputs')
    central=_huber_location(x,w,huber_clip_log)
    arithmetic=float(np.average(x,weights=w))
    simple=float(np.median(x))
    loo=[]
    for ix,r in enumerate(t.itertuples(index=False)):
        keep=np.arange(len(x))!=ix
        estimate=_huber_location(x[keep],w[keep],huber_clip_log)
        loo.append({'excluded_asset':r.asset,'ratio_without_asset':math.expm1(estimate),
                    'log_without_asset':estimate})
    loo_pd=pd.DataFrame(loo)
    return {'premium_ratio':math.expm1(central),'log_premium':central,
            'weighted_mean_ratio':math.expm1(arithmetic),'simple_median_ratio':math.expm1(simple),
            'leave_one_out_min':float(loo_pd.ratio_without_asset.min()),
            'leave_one_out_max':float(loo_pd.ratio_without_asset.max()),
            'leave_one_out':loo,
            'n_assets':int(len(x)),
            'method':'quality-weighted Huber M-estimator of log(IV30 / trailing RV30), 0.125 log clipping',
            'warning':'Small contemporaneous sample, NOT a forward IV risk premium. LOO range is sensitivity, not confidence interval.'}

def cost_aware_two_way_quote(surface,spot,strike,days,right,fee_bps=5.,half_spread_bps=5.,
                            slippage_bps=5.,hedges_per_month=30,
                            uncertainty_vol_pts=1.0,jump_vol_pts=1.0,
                            carry=0.,rate=0.,iv_spread_cap=None):
    """Symmetrical bid & offer anchored on fair IV, with cost/risk-based HALF width.

    Costs per option *side* = expected gross delta-hedge turnover * all-in execution bps.
    Future hedge turnover based on gamma, approximate Brownian total variation.
    Caps and model buffers must be disclosed as assumptions. Does not infer live order depth.
    """
    if any(v<0 for v in [fee_bps,half_spread_bps,slippage_bps,hedges_per_month,uncertainty_vol_pts,jump_vol_pts]):
        raise ValueError('Costs and reserves must be nonnegative')
    t=days/365
    f=spot*math.exp(carry*t)
    iv=surface.iv(f,strike,days)
    model=black76(f,strike,t,iv,right,rate)
    delta=model['forward_delta']*f/spot
    gamma=model['forward_gamma']*(f/spot)**2
    vega1=model['vega_per_1volpt']
    if vega1<=1e-8:raise ValueError('Tiny vega; cannot express cost as IV spread')
    n=max(1,int(round(hedges_per_month*days/30)))
    initial_turnover_near=abs(delta)
    dynamic_turnover_near=abs(gamma)*spot*iv*math.sqrt(2*t/math.pi)*math.sqrt(n)
    exec_rate=(fee_bps+half_spread_bps+slippage_bps)/10000
    execution_cost_per_option=(initial_turnover_near+dynamic_turnover_near)*spot*exec_rate
    cost_vol_points=execution_cost_per_option/vega1
    half_width=cost_vol_points+uncertainty_vol_pts+jump_vol_pts
    if iv_spread_cap is not None:half_width=min(half_width,iv_spread_cap/2)
    half_width=min(half_width,iv*100-5) # keep bid IV at least 5%
    if half_width<0:raise ValueError('IV too low for spread')
    bidiv=iv-half_width/100
    askiv=iv+half_width/100
    bid=black76(f,strike,t,bidiv,right,rate)['price']
    ask=black76(f,strike,t,askiv,right,rate)['price']
    return {'spot':spot,'forward':f,'strike':strike,'days':days,'right':right,
            'iv':iv,'bid_iv':bidiv,'ask_iv':askiv,'bid':bid,'mid':model['price'],'ask':ask,
            'fwd_delta':model['forward_delta'],'spot_delta':delta,'gamma_spot':gamma,
            'vega_per_1volpt':vega1,'outside_fit_range':abs(math.log(strike/f)/(surface.at(days)['atm']*math.sqrt(t)))>2.0,
            'spread_vol_pts':2*half_width,'half_width_vol_pts':half_width,
            'cost_vol_pts_per_side':cost_vol_points,'uncertainty_vol_pts_per_side':uncertainty_vol_pts,
            'jump_vol_pts_per_side':jump_vol_pts,
            'initial_hedge_near_per_option':initial_turnover_near,
            'expected_dynamic_hedge_turnover_near_per_option':dynamic_turnover_near,
            'execution_cost_usd_per_option_side':execution_cost_per_option,
            'hedge_trades_assumed':n,'assumptions_are_observed':False}

def historical_iv_vs_future_rv(iv_observations,closes,horizon=30):
    """Compare dated historical 30d IV with exactly 30 *subsequent* daily returns.

    Start from the completed UTC close *on* the observation day. For historical
    options sampled close to UTC midnight this is a documented daily-close
    approximation to an intraday 30-day future window. The quote itself never
    depends on future prices. Dates lacking all 31 daily closes are excluded.
    Note that trade-derived IV is neither executable option midpoint nor a
    delta-hedged P&L result.
    """
    required={'asset','asof_utc','atm30_iv'}
    if not required.issubset(iv_observations):raise ValueError('Missing asset,asof_utc,atm30_iv')
    px=closes.copy()
    px['date']=pd.to_datetime(px.date,utc=True).dt.normalize()
    lookup={(asset,date):float(close) for asset,date,close in px[['asset','date','close']].itertuples(index=False,name=None)}
    obs=iv_observations.copy()
    obs['asof_utc']=pd.to_datetime(obs.asof_utc,utc=True)
    obs['asof_day']=obs.asof_utc.dt.normalize()
    obs=obs.sort_values('asof_utc').drop_duplicates(['asset','asof_day'],keep='last')
    rows=[]
    for r in obs.itertuples(index=False):
        if not (np.isfinite(float(r.atm30_iv)) and .02<float(r.atm30_iv)<5):continue
        day=r.asof_day
        dates=pd.date_range(day,periods=horizon+1,freq='D')
        values=[lookup.get((r.asset,d)) for d in dates]
        if any(v is None or v<=0 or not np.isfinite(v) for v in values):continue
        future=float(np.std(np.diff(np.log(values)),ddof=1)*np.sqrt(365))
        iv=float(r.atm30_iv)
        rows.append({'asset':r.asset,'asof_utc':r.asof_utc.isoformat(),
                     'iv30_at_snapshot':iv,'subsequent30_rv':future,
                     'iv_minus_subsequent_rv_vol_pts':100*(iv-future),
                     'iv_squared_minus_future_rv_squared':iv*iv-future*future,
                     'window_start_completed_close':day.date().isoformat(),
                     'window_end_close':dates[-1].date().isoformat(),
                     'interpretation':'ex-post IV minus subsequent RV; not hedge P&L or traded edge'})
    return pd.DataFrame(rows)
