"""12 Nov NEAR desk exercise: conservative peer SVI strike-skew diagnostic.
Run in the existing NEAR_OTC_Surface_Project directory after run_peer_svi_smile.py.
Inputs MUST be dated raw two-sided source option observations, never synthetic data.
Options IV: Derive/Deribit. Coinbase/Kraken: underlying spot/RV only.
ZEC 43-day slice is reported separately, never disguised as a 30-day surface.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

from src.peer_svi_surface import (prepare_nodes, fit_peer_slices, extract_venue_smiles,
                                   near_smile_from_peers, static_checks, TGT_T)

ASOF='2026-11-12T07:41:41.643402+00:00'
NEAR_ATM=1.1890025903880814
NEAR_SPOT=3.51
MONEYNESS=np.array([.85,.90,.95,1.0,1.05,1.10,1.15,1.20])


def delta_with_atm(z, iv, atm):
    k=z*atm*math.sqrt(TGT_T)
    d1=(-k+0.5*iv*iv*TGT_T)/(iv*math.sqrt(TGT_T))
    return float(norm.cdf(d1))


def delta_points(asset_rows, atm_iv):
    # Asset-level 30d standardized-z curves, no wing extrapolation.
    a=asset_rows.sort_values('z').drop_duplicates('z')
    z=a.z.to_numpy(float)
    v=atm_iv*a.iv_ratio.to_numpy(float)
    def f(zz,target):
        vi=float(np.interp(zz,z,v))
        return delta_with_atm(zz,vi,atm_iv)-target
    def solve(target,side):
        ii=np.where(z<0)[0] if side=='put' else np.where(z>0)[0]
        for i,j in zip(ii[:-1],ii[1:]):
            if f(z[i],target)*f(z[j],target)<=0:
                root=float(brentq(lambda zz:f(zz,target),float(z[i]),float(z[j])))
                return root,float(np.interp(root,z,v))
        return float('nan'),float('nan')
    zp,vp=solve(.75,'put')
    zc,vc=solve(.25,'call')
    if not np.isfinite(vp) or not np.isfinite(vc):
        return dict(put25_iv_pct=np.nan,call25_iv_pct=np.nan,rr25_pts=np.nan,bf25_pts=np.nan,
                    note='Insufficient observed strike support for true 25-delta wings')
    return dict(put25_iv_pct=vp*100,call25_iv_pct=vc*100,rr25_pts=(vc-vp)*100,
                bf25_pts=(.5*(vc+vp)-atm_iv)*100,
                note='30d fitted peer SVI, not direct executable 25-delta quotes')


def run(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--nodes',default='outputs/source_peer_option_nodes.csv')
    parser.add_argument('--asof',default=ASOF)
    parser.add_argument('--out',default='outputs')
    parser.add_argument('--max-svi-rmse-pts',type=float,default=2.0,
                        help='Conservative gate, previous script admitted 3.5-pt RMSE')
    args=parser.parse_args(argv)
    o=Path(args.out);o.mkdir(exist_ok=True,parents=True)
    if not Path(args.nodes).exists():
        raise SystemExit('Missing raw dated option-chain CSV. No synthetic substitution allowed.')
    df=prepare_nodes(pd.read_csv(args.nodes),args.asof)
    fitted,base_audit=fit_peer_slices(df)
    raw_audit=base_audit.copy()
    raw_audit['strict_status']=np.where((raw_audit.status=='accepted') &
                          (raw_audit.rmse_vol_pts<=args.max_svi_rmse_pts),'accepted','rejected')
    raw_audit['strict_reason']=np.where((raw_audit.status=='accepted') &
                 (raw_audit.rmse_vol_pts>args.max_svi_rmse_pts),
                 'SVI residual above tighter desk threshold',raw_audit.reason)
    raw_audit.to_csv(o/'peer_svi_strict_slice_audit.csv',index=False)
    qualified=[s for s in fitted if s.rmse_vol_pts<=args.max_svi_rmse_pts]

    # Dec-25 ZEC is useful but NOT enough to interpolate to a 30-day ZEC smile.
    zec=[r for r in fitted if r.asset=='ZEC' and r.expiry=='2026-12-25']
    zec_audit={'dec25_slice_present':bool(zec),
               'dec25_rmse_vol_pts':zec[0].rmse_vol_pts if zec else None,
               'dec25_qualifies_strict':bool(zec and zec[0].rmse_vol_pts<=args.max_svi_rmse_pts),
               'nov27_usable':any(s.asset=='ZEC' and s.expiry=='2026-11-27' for s in qualified),
               'include_in_30d_baseline':False,
               'reason':'No valid bracketing ZEC near expiry. Dec 25 is a separate 43-day check, not a 30-day input.'}
    (o/'zec_dec25_tenor_check.json').write_text(json.dumps(zec_audit,indent=2))

    z_target=np.log(MONEYNESS)/(NEAR_ATM*np.sqrt(TGT_T))
    z=np.unique(np.r_[np.linspace(-1.4,1.65,612),z_target,0.])
    obs,venue_audit=extract_venue_smiles(qualified,z)
    venue_audit.to_csv(o/'peer_svi_strict_venue_audit.csv',index=False)
    if obs.empty:raise SystemExit('No supported 30d asset smiles after strict filtering.')
    assets=sorted(set(obs.asset))
    peer=pd.DataFrame(obs.groupby(['asset','z'],as_index=False).iv_ratio.median())
    atm_by_asset=venue_audit[venue_audit.status=='accepted'].groupby('asset').atm_iv_pct.median().div(100).to_dict()
    peer['peer_atm_pct']=peer.asset.map({x:100*v for x,v in atm_by_asset.items()})
    peer['peer_model_iv_pct']=peer.iv_ratio*peer.peer_atm_pct
    peer['near_anchored_iv_pct']=peer.iv_ratio*NEAR_ATM*100
    peer.to_csv(o/'peer_svi_individual_30d_curves.csv',index=False)
    points=[]
    for asset in assets:
        p=peer[peer.asset==asset]
        result=delta_points(p,atm_by_asset[asset])
        result.update(asset=asset,atm_iv_pct=100*atm_by_asset[asset],n_z_points=len(p))
        points.append(result)
    pd.DataFrame(points).to_csv(o/'peer_svi_25delta_skew.csv',index=False)

    # A single asset contributes at most one curve. No hidden weighting from multi-venue peers.
    tgt=peer[peer.z.round(8).isin(np.round(z_target,8))].copy()
    # Avoid floating/rounding errors when matching the requested strike grid.
    rows=[]
    for m,zt in zip(MONEYNESS,z_target):
        sub=peer[np.isclose(peer.z,float(zt),atol=5e-8)]
        vals=sub.groupby('asset').iv_ratio.median()
        if len(vals)<3:
            raise SystemExit(f'Only {len(vals)} independent assets at {m:.0%} strike after stricter fit screening; no new NEAR quote ladder produced.')
        for a,v in vals.items():rows.append({'strike_over_forward':m,'asset':a,'z':zt,'near_anchored_iv_pct':100*NEAR_ATM*v,'iv_ratio':v})
    sampled=pd.DataFrame(rows)
    sampled.to_csv(o/'peer_svi_30d_cross_asset_strike_comparison.csv',index=False)
    near=near_smile_from_peers(obs,NEAR_ATM,NEAR_SPOT,MONEYNESS)
    flags=static_checks(near)
    near.to_csv(o/'peer_svi_strict_near_smile.csv',index=False)
    if flags:
        raise SystemExit('Strict NEAR smile failed discrete static-arbitrage screen: '+repr(flags)+'; do not quote.')

    # Reprice the 9 model contracts using ONLY the revised strict-fit curve.
    # Existing cost / jump reserves remain the same modelling assumptions.
    from scipy.interpolate import PchipInterpolator
    from src.desk_calibration import cost_aware_two_way_quote
    class Strict30dSlice:
        def __init__(self,df):
            self.interpolator=PchipInterpolator(np.log(df.strike_over_forward.to_numpy(float)),
                 df.model_iv_pct.to_numpy(float)/100,extrapolate=False)
        def iv(self,f,k,days):
            if abs(days-30)>1e-7 or abs(f-NEAR_SPOT)>1e-7:
                raise ValueError('Only the dated 30d slice is supported')
            v=float(self.interpolator(np.log(k/f)))
            if not np.isfinite(v):raise ValueError('Strike outside supported interpolation range')
            return v
        def at(self,days):return {'atm':NEAR_ATM}
    sl=Strict30dSlice(near)
    ladd=[]
    for r in near.itertuples(index=False):
        for right in (['C','P'] if abs(r.strike_over_forward-1)<1e-8 else
                      ['P'] if r.strike_over_forward<1 else ['C']):
            q=cost_aware_two_way_quote(sl,NEAR_SPOT,float(r.strike_usd),30,right)
            ladd.append({'right':right,'strike_over_forward':r.strike_over_forward,
                 'strike_usd':float(r.strike_usd),'fair_iv_pct':100*q['iv'],
                 'bid_iv_pct':100*q['bid_iv'],'offer_iv_pct':100*q['ask_iv'],
                 'bid_usd_per_near':q['bid'],'fair_usd_per_near':q['mid'],
                 'ask_usd_per_near':q['ask'],'delta':q['spot_delta'],
                 'gamma':q['gamma_spot'],'vega_usd_per_volpoint':q['vega_per_1volpt'],
                 'full_spread_vol_pts':q['spread_vol_pts'],
                 'estimated_hedging_cost_usd_per_side':q['execution_cost_usd_per_option_side'],
                 'source_status':'MODEL_RESEARCH_NOT_EXECUTABLE'})
    newlad=pd.DataFrame(ladd)
    newlad.to_csv(o/'peer_svi_strict_9_contract_quotes.csv',index=False)
    covered=newlad[(newlad.right=='C') & (np.isclose(newlad.strike_over_forward,1.2))].iloc[0]
    hedge={'rfq_calls':10000,'strike_usd':float(covered.strike_usd),
      'dealer_purchase_bid_usd_per_call':float(covered.bid_usd_per_near),
      'client_premium_usd':float(covered.bid_usd_per_near*10000),
      'dealer_long_call_delta_near':float(covered.delta*10000),
      'initial_dealer_short_underlying_near':float(-covered.delta*10000),
      'dealer_gamma_per_dollar':float(covered.gamma*10000),
      'dealer_vega_per_vol_pt_usd':float(covered.vega_usd_per_volpoint*10000),
      'note':'Fixed IV research Greeks, hedge costs assumed; not a real RFQ'}
    (o/'peer_svi_strict_covered_call.json').write_text(json.dumps(hedge,indent=2))

    # Plot peers individually, and show ZEC as a SEPARATE 43d visual screen in a second panel.
    fig,ax=plt.subplots(figsize=(9.0,4.9),dpi=190)
    pal={'AVAX':'#b07b4a','HYPE':'#5b7188','SOL':'#518876','ZEC':'#7d6b8e'}
    for a in assets:
        ss=sampled[sampled.asset==a].sort_values('strike_over_forward')
        ax.plot(100*ss.strike_over_forward,ss.near_anchored_iv_pct,
                '.-',lw=1.8,label=a+' 30d (ATM-normalized)',color=pal.get(a))
    ax.plot(100*near.strike_over_forward,near.model_iv_pct,'o-',lw=2.9,
            label='NEAR transferred median',color='#17354a')
    ax.fill_between(100*near.strike_over_forward,near.iv_sensitivity_min_pct,
                     near.iv_sensitivity_max_pct,color='#c5dbe3',alpha=.3,label='Across-peer range (not CI)')
    ax.axvline(100,color='#9ba6b0',ls='--',lw=1)
    ax.set_xlabel('Strike as % of 30d forward');ax.set_ylabel('IV re-anchored to NEAR ATM (%)')
    ax.set_title('30d peer SVI shapes transferred to the same 118.9% ATM anchor',loc='left',fontsize=11)
    ax.grid(alpha=.16);ax.legend(fontsize=8,ncol=2,loc='best');fig.tight_layout()
    fig.savefig(o/'peer_svi_individual_30d_comparison.png',dpi=190,bbox_inches='tight');plt.close(fig)

    # Visual cross-check from supplied screenshots; bid-vs-mark IVs MUST NOT be interpreted as common executable mid marks.
    shots=pd.DataFrame([
      {'asset':'HYPE','tenor_days':43,'spot_reference':83.2918,'strike':70,'displayed_iv_pct':59.45,'iv_type':'screenshot mark IV'},
      {'asset':'HYPE','tenor_days':43,'spot_reference':83.2918,'strike':75,'displayed_iv_pct':57.09,'iv_type':'screenshot mark IV'},
      {'asset':'HYPE','tenor_days':43,'spot_reference':83.2918,'strike':80,'displayed_iv_pct':55.49,'iv_type':'screenshot mark IV'},
      {'asset':'HYPE','tenor_days':43,'spot_reference':83.2918,'strike':84,'displayed_iv_pct':54.40,'iv_type':'screenshot mark IV'},
      {'asset':'ZEC','tenor_days':43,'spot_reference':990.96,'strike':700,'displayed_iv_pct':112.7,'iv_type':'screenshot put bid IV'},
      {'asset':'ZEC','tenor_days':43,'spot_reference':990.96,'strike':800,'displayed_iv_pct':110.4,'iv_type':'screenshot put bid IV'},
      {'asset':'ZEC','tenor_days':43,'spot_reference':990.96,'strike':900,'displayed_iv_pct':108.6,'iv_type':'screenshot put bid IV'},
      {'asset':'ZEC','tenor_days':43,'spot_reference':990.96,'strike':1000,'displayed_iv_pct':107.4,'iv_type':'screenshot put bid IV'}])
    shots['atm_reference_iv_pct']=shots.asset.map({'HYPE':54.4,'ZEC':107.4})
    shots['standardized_z_approx']=np.log(shots.strike/shots.spot_reference)/(
                shots.atm_reference_iv_pct/100*np.sqrt(43/365))
    shots['vol_pts_vs_atm']=shots.displayed_iv_pct-shots.atm_reference_iv_pct
    shots.to_csv(o/'screenshot_43day_skew_crosscheck.csv',index=False)

    summary={'asof':args.asof,'fit_gate_vol_pts':args.max_svi_rmse_pts,
      'qualified_30d_assets':assets,'qualifying_slices_after_strict_gate':len(qualified),
      'near_atm_iv_pct':NEAR_ATM*100,'zec_dec_25':zec_audit,
      'near_grid_iv_pct':list(map(float,near.model_iv_pct)),
      'strict_rfq_reference':hedge,
      'notes':['Option IV only from Derive/Deribit raw chains','Coinbase/Kraken prices support spot/RV, not option IV',
               'ZEC Dec 25 screenshot is a bid-IV crosscheck for ~43 days, not a two-sided full 30-day surface',
               '25-delta risk reversals from model fit, only where both wings observed',
               'Dated 12 Nov snapshot; no claim of live market conditions on later dates',
               'Quote ladder and full dynamic P&L must be recalculated after any new final smile']}
    (o/'peer_svi_diagnostics_summary.json').write_text(json.dumps(summary,indent=2))
    print('Qualified 30d independent peers:', ', '.join(assets))
    print('Stricter accepted slices:', len(qualified))
    print('ZEC Dec25 eligible for 30d: NO (nearer expiry insufficient)')
    print('25-delta skews:')
    print(pd.DataFrame(points)[['asset','rr25_pts','bf25_pts','note']].to_string(index=False))
    print('Strict-fit indicative 20% OTM covered call: bid ${:.3f}; initial short {:,.0f} NEAR'.format(hedge['dealer_purchase_bid_usd_per_call'], -hedge['initial_dealer_short_underlying_near']))
    print('Updated NEAR curve:')
    print(near[['strike_over_forward','model_iv_pct']].to_string(index=False))
    print('All diagnostics saved in',o.resolve())

if __name__=='__main__': run()
