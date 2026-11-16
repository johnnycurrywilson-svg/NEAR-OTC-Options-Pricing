"""Run from the existing NEAR_OTC_Surface_Project folder (Python 3.13)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.peer_svi_surface import prepare_nodes,fit_peer_slices,extract_venue_smiles,near_smile_from_peers,static_checks,TGT_T

ASOF='2026-11-12T07:41:41.643402+00:00'
MONEINESS=[.85,.90,.95,1.,1.05,1.10,1.15,1.20]


def execute(argv=None):
    parser=argparse.ArgumentParser(description='Evidence-gated NEAR 30d smile from live peer option bid/ask')
    parser.add_argument('--nodes',default='outputs/source_peer_option_nodes.csv',help='RAW LIVE Derive/Deribit nodes')
    parser.add_argument('--asof',default=ASOF,help='Same timestamp as the fixed pricing snapshot (UTC)')
    parser.add_argument('--near-atm-iv',type=float,default=118.90025903880814,help='Annualised percent, 12-Nov fixed fair ATM research IV')
    parser.add_argument('--near-spot',type=float,default=3.51)
    parser.add_argument('--out',default='outputs')
    args=parser.parse_args(argv)
    path=Path(args.nodes)
    if not path.exists():
        raise SystemExit(f'RAW CHAIN MISSING: {path}. Regenerate the LIVE source_peer_option_nodes.csv first.')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    # A failed or sparse fresh run must not leave an obsolete 'successful' chart.
    for stale in ('peer_svi_near_smile.csv','peer_svi_near_smile.png',
                  'peer_svi_9_contract_quote_ladder.csv','peer_svi_methodology_audit.json'):
        (out/stale).unlink(missing_ok=True)
    df=prepare_nodes(pd.read_csv(path),args.asof)
    slices,audit=fit_peer_slices(df)
    audit.to_csv(out/'peer_svi_slice_audit.csv',index=False)
    z=np.log(np.array(MONEINESS))/(args.near_atm_iv/100*np.sqrt(TGT_T))
    observed,venue_audit=extract_venue_smiles(slices,z)
    venue_audit.to_csv(out/'peer_svi_venue_audit.csv',index=False)
    if not observed.empty:observed.to_csv(out/'peer_svi_normalized_observations.csv',index=False)
    if observed.empty:
        raise SystemExit('NO ACCEPTABLE 30-DAY PEER SMILES: See peer_svi_slice_audit.csv and peer_svi_venue_audit.csv. No curve fabricated.')
    try:
        surface=near_smile_from_peers(observed,args.near_atm_iv/100,args.near_spot,MONEINESS)
    except ValueError as ex:
        raise SystemExit(f'NOT ENOUGH INDEPENDENT PEER WING DATA: {ex}. No NEAR smile chart or quote ladder created.')
    flags=static_checks(surface)
    if flags:
        surface.to_csv(out/'peer_svi_rejected_surface.csv',index=False)
        raise SystemExit(f'NO-ARBITRAGE SCREEN FAILED: {flags}. See peer_svi_rejected_surface.csv; no quote ladder created.')
    surface.to_csv(out/'peer_svi_near_smile.csv',index=False)
    # If the smile has sufficient support and passes static checks, turn it
    # into a NINE-CONTRACT MODEL RFQ LADDER using the existing hedge cost model.
    # Quote widths are still assumptions, never executable NEAR prices.
    from src.desk_calibration import cost_aware_two_way_quote
    from scipy.interpolate import PchipInterpolator
    class SlicedSurface:
        def __init__(self,curve):
            self.strikes=curve.strike_usd.to_numpy(float)
            self.ivs=curve.model_iv_pct.to_numpy(float)/100.
            self.fwd=args.near_spot
            self.spline=PchipInterpolator(np.log(self.strikes/self.fwd),self.ivs,extrapolate=False)
        def iv(self,forward,strike,days):
            if abs(days-30)>1e-6 or abs(forward-self.fwd)>1e-6:
                raise ValueError('This script only has a validated fixed 30-day NEAR slice')
            vol=float(self.spline(np.log(strike/forward)))
            if not np.isfinite(vol):raise ValueError('Strike outside peer-covered grid')
            return vol
        def at(self,days):return {'atm':args.near_atm_iv/100}
    theoretical=SlicedSurface(surface)
    ladder=[]
    for r in surface.itertuples(index=False):
        m=float(r.strike_over_forward)
        for side in (['C','P'] if abs(m-1)<1e-8 else ['P'] if m<1 else ['C']):
            q=cost_aware_two_way_quote(theoretical,args.near_spot,r.strike_usd,30,side)
            ladder.append({'right':side,'strike_usd':round(r.strike_usd,4),
                'strike_over_forward':m,'fair_iv_pct':100*q['iv'],
                'bid_iv_pct':100*q['bid_iv'],'offer_iv_pct':100*q['ask_iv'],
                'bid_usd_per_near':q['bid'],'fair_usd_per_near':q['mid'],
                'ask_usd_per_near':q['ask'],'delta':q['spot_delta'],
                'gamma':q['gamma_spot'],'vega_usd_per_volpoint':q['vega_per_1volpt'],
                'spread_vol_points':q['spread_vol_pts'],
                'model_quote_only_not_executable':True})
    pd.DataFrame(ladder).to_csv(out/'peer_svi_9_contract_quote_ladder.csv',index=False)
    fig,ax=plt.subplots(figsize=(8.4,4.8))
    ax.plot(surface.strike_over_forward*100,surface.model_iv_pct,'-o',lw=2,label='NEAR research smile (from two-sided peer IV)')
    ax.fill_between(surface.strike_over_forward*100,surface.iv_sensitivity_min_pct,
                    surface.iv_sensitivity_max_pct,alpha=.16,label='Across-peer scenario range (not CI)')
    ax.axhline(args.near_atm_iv,lw=1,ls='--',color='gray',label='NEAR ATM anchor')
    ax.set(xlabel='Strike / forward (%)',ylabel='30-day implied volatility (annualised %)',
           title='30-day NEAR model smile • 12 Nov 2026 snapshot')
    ax.legend(fontsize=8);ax.grid(alpha=.15);fig.tight_layout()
    fig.savefig(out/'peer_svi_near_smile.png',dpi=180);plt.close(fig)
    metadata={'snapshot_utc':args.asof,'source':'two-sided live option IV only; dated Derive and Deribit source rows',
              'model':'separate SVI per peer venue and expiry, total-variance interpolation to 30d, transfer by normalized std moneyness',
              'near_atm_anchor_pct':args.near_atm_iv,'quote_status':'RESEARCH IV CURVE, NOT EXECUTABLE',
              'asset_coverage':sorted(observed.asset.unique().tolist()),
              'qualifying_slices':len(slices),'quote_time_caution':'Deribit book-level update time not independently verified',
              'limitations':['Quote depth and stale-book risk','SVI is screened on sampled grids only','Peer wing IV may not transfer to NEAR','No forward basis, funding or inventory included',
                             'Across-peer range is not a confidence interval'],
              'generated_files':['peer_svi_near_smile.csv','peer_svi_near_smile.png','peer_svi_9_contract_quote_ladder.csv','peer_svi_slice_audit.csv','peer_svi_venue_audit.csv']}
    (out/'peer_svi_methodology_audit.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    print('Qualified peer SVI slices:',len(slices))
    print('Independent assets:',', '.join(metadata['asset_coverage']))
    print(surface[['strike_over_forward','model_iv_pct','independent_peer_assets']].to_string(index=False))
    print('Charts, curve, and audit saved in:',out.resolve())

if __name__=='__main__':execute()
