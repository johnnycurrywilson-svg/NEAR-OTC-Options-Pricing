"""Research reproduction: Derive + Deribit altcoin IV -> NEAR indicative OTC quote.

Examples:
    python run.py --mode demo
    python run.py --mode live --near-spot 4.40 --near-rv 1.10
    python run.py --mode csv --peer-csv data/derive_options.csv --near-spot 4.40 --near-rv 1.10
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.model import NearSurface, fit_peer_smiles, peer_at, hedged_stress, simple_arbitrage_checks
from src.market_data import download_derivexyz, sample_synthetic_peers
from src.deribit_data import download_deribit, sample_synthetic_deribit
from src.near_data import fetch_near_daily_prices, historical_volatility
from src.iv_rv_premium import (fetch_peer_closes, normalize_custom_closes, realised_from_prices,
                               make_peer_premium_table, estimate_neutral_premium,
                               append_snapshots, evaluate_matured_snapshots)
from src.desk_calibration import (forecast_near_rv, rolling_forward_rv_backtest,
                                  robust_peer_premium, cost_aware_two_way_quote,
                                  historical_iv_vs_future_rv)
from src.historical_deribit import backfill_history
from src.vol_forecasting import compare_forecasts, save_holdout_chart

ROOT=Path(__file__).resolve().parent


def charts(fits, surface, spot, out, mode):
    tag='SYNTHETIC DEMONSTRATION - NOT LIVE QUOTES' if mode=='demo' else 'PUBLIC EXCHANGE SNAPSHOTS - NOT EXECUTABLE'
    fig, ax=plt.subplots(figsize=(9,5))
    z=np.linspace(-2,2,100)
    for asset, venue in fits[['asset','venue']].drop_duplicates().itertuples(index=False, name=None):
        sub=fits[(fits.asset==asset)&(fits.venue==venue)]
        dte=sub.dte.to_numpy(dtype=float)
        ivs=sub.atm.to_numpy(dtype=float)
        from src.model import _tenor_iv
        atm=_tenor_iv(dte,ivs,30)
        a=float(np.interp(30,dte,sub.slope))
        b=float(np.interp(30,dte,sub.curvature))
        y=atm*(1+a*z+b*z*z)
        ax.plot(z,100*y,label=f'{asset} / {venue} · ATM {atm*100:.1f}%',lw=1.5,alpha=.85,
                ls='--' if venue=='derive' else '-')
    near=surface.at(30)
    ax.plot(z,100*near['atm']*(1+near['slope']*z+near['curvature']*z*z),
            label=f"NEAR model  ·  ATM {near['atm']*100:.1f}%",color='black',lw=3,ls=':')
    ax.set(xlabel='Standardized log-forward moneyness  z = ln(K/F)/(ATM IV × √T)',
           ylabel='Annualized implied volatility (%)',title='30-day peer smiles → synthetic NEAR smile')
    ax.grid(alpha=.2); ax.legend(fontsize=7,ncol=2,loc='upper center');fig.text(.02,.01,tag,fontsize=9)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(out/'01_peer_and_near_smiles.png',dpi=190);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,5));ds=np.linspace(7,90,70)
    for a in surface.peers:
        ax.plot(ds,[100*peer_at(fits,a,float(d))['atm'] for d in ds],label=a,lw=1.8)
    ax.plot(ds,[100*surface.at(float(d))['atm'] for d in ds],color='black',ls='--',lw=3,label='NEAR model')
    ax.set(xlabel='Days to expiry',ylabel='ATM implied volatility (%)',title='ATM term structures: relative peer slope transferred to NEAR')
    ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=8);fig.text(.02,.01,tag,fontsize=9)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(out/'02_atm_term_structure.png',dpi=190);plt.close(fig)

    tenor=np.linspace(7,90,55); moneyness=np.linspace(.6,1.55,85)
    grid=[]
    for d in tenor:
        params=surface.at(float(d))
        z=np.log(moneyness)/(params['atm']*np.sqrt(d/365.))
        zc=2*np.tanh(z/2)
        grid.append(100*np.maximum(.10,params['atm']*(1+params['slope']*zc+params['curvature']*zc*zc)))
    grid=np.asarray(grid)
    fig,ax=plt.subplots(figsize=(9,5));im=ax.pcolormesh(moneyness,tenor,grid,cmap='viridis',shading='auto')
    ax.set(xlabel='Strike / forward', ylabel='Days to expiry',title='Constructed NEAR volatility surface')
    cbar=fig.colorbar(im,ax=ax);cbar.set_label('Implied volatility (%)')
    ax.axvline(1,color='white',alpha=.7,ls='--',lw=1)
    fig.text(.02,.01,tag,fontsize=9);fig.tight_layout(rect=[0,.04,1,1])
    fig.savefig(out/'03_near_surface_heatmap.png',dpi=190);plt.close(fig)


def format_report(mode,asof,near_spot,near_rv,vrp,weights,fits,surface,tickets,issues,output,spread_vol_pts):
    p30=surface.at(30)
    rows=[]
    for asset in surface.peers:
        p=peer_at(fits,asset,30)
        rows.append(f"| {asset} | {', '.join(sorted(fits.loc[fits.asset==asset, 'venue'].unique()))} | {p['atm']*100:.1f}% | {p['slope']:+.3f} | {p['curvature']:.3f} | {weights[asset]*100:.1f}% |")
    text=f'''# NEAR Protocol OTC options pricing — research case study

**Mode:** `{mode.upper()}` | **As-of:** {asof} | **Source:** Derive and/or Deribit public options APIs (live/csv) or fully labelled synthetic demo (demo).

> **IMPORTANT:** {"This demonstration is NOT based on currently executable Derive/Deribit quotes. Smile slopes, convexity and longer-tenor data are synthetic assumptions. All Deribit demo observations and all smile shapes are illustrative assumptions." if mode=='demo' else "Quote data are venue snapshots: exchange model marks and stale/illiquid points must not be treated as executable OTC prices. Live connector must be verified in your own environment."}

## Trading question

How would an OTC market maker price an option on NEAR, where no robust exchange-traded NEAR options surface is available? Build a **defensible starting surface**, convert it to a two-way vanilla quote, and calculate the subsequent Greeks and hedge exposure. This is a *paper quote*, not a strategy performance or arbitrage claim.

## Step 1 — observe and clean comparables

Fetch HYPE/SOL/ZEC from Derive (`public/get_instruments`, `public/get_tickers`) and AVAX/HYPE/SOL/TRX from Deribit (`public/get_instruments`, `public/get_book_summary_by_currency`). Reject expired/stale rows, price-inconsistent books and invalid IVs. Prefer two-sided quote IV; downweight exchange model marks that cannot be executed. Restrict to OTM put and call wings; record quote classification and time. Never label a model mark as an actual order.

## Step 2 — fit each asset's smile in standardized coordinates

For each expiry set `z = ln(K/F) / (ATM_IV * sqrt(T))`, then fit `IV(K,T) = ATM_IV(T) * [1 + a(T) z + b(T) z^2]`. This makes HYPE/SOL/ZEC strikes comparable despite different USD prices and volatility. `a` controls put-versus-call asymmetry and `b` controls convexity/smile. The two-dimensional fit is deliberately simple and interpretable; it is not a production arbitrage-free SVI implementation.

### Estimated 30-day peer coefficients

| Peer | Venue observations | ATM IV | Slope a | Curvature b | Reliability-adjusted weight |
|:--|:--|--:|--:|--:|--:|
{chr(10).join(rows)}

**Weights are modelling assumptions**, with indicative starting weights HYPE 30%, SOL 25%, ZEC 20%, AVAX 15% and TRX 10%; live weights are attenuated when most peer points are model-only rather than two-sided. Missing peers cannot be silently substituted.

## Step 3 — transfer the *shape*, not the entire volatility level

NEAR has its own realised volatility, event catalysts and hedging liquidity. Rather than saying `NEAR IV = HYPE IV`, anchor NEAR's 30-day ATM IV to its independent volatility estimate plus a disclosed option risk premium:

`NEAR_ATM30 = NEAR_RV30 × (1 + IV/RV premium)`.

In this run: NEAR reference spot **${near_spot:.2f}**, NEAR 30-day RV assumption **{near_rv*100:.1f}%**, premium assumption **{vrp*100:.1f}%**, giving **{surface.atm30*100:.1f}%** ATM IV.

Transfer peer term-structure *ratios*: `NEAR_ATM(T) = NEAR_ATM30 × weighted_peer_ATM(T) / weighted_peer_ATM(30d)`. Transfer the peer-weighted smile `a(T), b(T)` around that NEAR-specific level. ATM total variance is interpolated between observed tenors before deriving IV; outside tenors the model uses flat IV extrapolation.

For the chosen 30-day tenor: **ATM {p30['atm']*100:.1f}%**, slope **{p30['slope']:+.3f}**, curvature **{p30['curvature']:.3f}**.

## Step 4 — turn the surface into an OTC indicative market

Use Black-76 European options on forward `F = spot × exp(carry × T)` with annualised IV from the NEAR surface. The quote widens theoretical IV by **±{spread_vol_pts/2:g} volatility points** (a full {spread_vol_pts:g}-point vol spread), solely as an illustration. This is **not derived from actual NEAR hedge slippage or competitive dealer RFQs**.

| Structure | Strike | Market IV | Bid premium | Mid premium | Offer premium | Delta/token | Vega/token / 1 vol pt |
|:--|--:|--:|--:|--:|--:|--:|--:|
'''
    for label,q in tickets.items():
        text+=f"| {label} | ${q['strike']:.3f} | {q['iv']*100:.1f}% | ${q['bid']:.4f} | ${q['mid']:.4f} | ${q['ask']:.4f} | {q['spot_delta']:+.3f} | ${q['vega_per_1volpt']:.4f} |\n"
    call=tickets['ATM call'];out_call=tickets['20% OTM covered call']
    text+=f'''

A client **buying** the ATM call would pay the hypothetical offer of **${call['ask']:.4f} per NEAR**; a client **selling** that call would receive the bid. A client writing the covered call sells to the desk at its *bid*, rather than receiving the offer. Quote orientation must be explicit.

For **10,000 ATM calls sold to a client** (dealer short calls), the initial hedge is to **buy approximately {call['spot_delta']*10000:,.0f} NEAR** or equivalent linear exposure. The model's delta is local to the assumed forward and surface. Rebalancing introduces realised gamma P&L, slippage and venue risk. A 20% OTM covered call is the separate client-sells-to-dealer example, with theoretical bid premium **${out_call['bid']:.4f}** per token.

## Step 5 — audit the risk

Scenario table `stress_atm_call_short.csv` shows *seven-day* model revaluation of a short 10,000-token ATM call position plus its **fixed initial long-delta hedge**, across spot moves ±15%/±30% and vol shifts ±20 points. These are paper scenario MTM outcomes, **not a dynamic hedging backtest**. They exclude fees, liquidation, funding, market impact, collateral and counterparty risk.

Discrete static-arbitrage screens (fixed-forward call monotonicity, strike convexity, and calendar) returned: **{'PASS on sampled grid' if not issues else '; '.join(issues)}**. Passing these checks is not proof of a fully arbitrage-free surface.

## Further desk checks before any firm quote

1. Source timestamp, tradability and comparison with external venues; validate the live Derive and Deribit adapters and their market-data contracts.
2. Obtain actual NEAR spot/perp order-book depth, impact curves, basis, variable funding, exchange restrictions and inventory limits.
3. Estimate NEAR's conditional forecast volatility using multiple realised-vol windows and event-specific stress.
4. Replace ad hoc 6-point volatility spread with expected delta-hedge slippage, gamma bleed, model uncertainty, tail risk and counterparty/margin capital.
5. Upgrade smooth smile fitting to an arbitrage-aware constrained SVI/SSVI approach, including total-variance/calendar consistency and out-of-sample tests.
6. Perform actual *walk-forward* paper quoting and discrete delta-hedging simulation; compare realised outcomes to ex-ante assumptions.

## What this demonstrates for Granite River

A complete research workflow: retrieve option chains; qualify observed marks; compare cross-asset skew and term structure; construct a NEAR-specific pricing input; generate OTC bid/offer prices; calculate Greeks, hedge sizes and scenario P&L. **It does not claim** a production trading system, a profitable strategy, or personal historic desk experience with this exact Python code.

### Sources and conventions

- Official Derive public API: https://docs.derive.xyz/ and https://api.lyra.finance
- Deribit public market API: https://docs.deribit.com/ and https://www.deribit.com/api/v2/public/
- Derive tracked market ATM reference (demo anchors only, accessed 30 October 2026): https://loris.tools/options/derive
- Independent public Derive parsing example: https://gist.github.com/GeorgL0ngGamma/bc7a49f20d78a60d476036893f5af887
- GRT job brief: https://www.graniteriver.io/careers/quant-options-trader-north-america/
- **Input unit:** IV is decimal annualised; e.g., 1.10 = 110%.
- **Discount rate & carry:** 0 in demo. Settlement is hypothetical USD cash; contract specification must be confirmed before firm quotes.
'''
    (output/'case_study.md').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description='Cross-asset crypto option surfaces -> NEAR hypothetical OTC quote')
    parser.add_argument('--mode',choices=['demo','live','csv'],default='demo')
    parser.add_argument('--peer-csv',type=Path)
    parser.add_argument('--venues',default='derive,deribit',help='Comma-separated subset of derive,deribit; default both')
    parser.add_argument('--near-spot',type=float)
    parser.add_argument('--near-rv',type=float,help='Annualized 30d NEAR volatility DECIMAL: 1.10 = 110%%')
    parser.add_argument('--near-closes-csv',type=Path,help='CSV with chronological close column, >=31 days')
    parser.add_argument('--near-auto-history',action='store_true',help='Fetch NEAR/USD spot and 7/14/30/60/90-day RV from Kraken automatically')
    parser.add_argument('--pdf',action='store_true',help='Regenerate a three-page PDF report after pricing')
    parser.add_argument('--vrp-ratio',type=float,default=None,
                        help='MANUAL override of IV/RV ratio uplift (0.08 = +8%% of RV); default live auto-calibrates from peer IV/RV, demo uses SYNTHETIC 8%%')
    parser.add_argument('--peer-closes-csv',type=Path,
                        help='Optional asset,date,close CSV of peer daily prices; otherwise Kraken peer daily OHLC is requested in live mode')
    parser.add_argument('--premium-archive',type=Path,default=ROOT/'data'/'peer_premium_snapshots.csv',
                        help='Persist dated real peer IV/RV observations across runs for later ex-post evaluation')
    parser.add_argument('--spread-vol-pts',type=float,default=None,help='Legacy full fixed IV spread override in vol points; skips cost model')
    parser.add_argument('--hedge-fee-bps',type=float,default=5.,help='ASSUMED hedge fee, bps of linear turnover')
    parser.add_argument('--hedge-halfspread-bps',type=float,default=5.,help='ASSUMED half bid/ask spread of NEAR linear hedge, bps')
    parser.add_argument('--hedge-slippage-bps',type=float,default=5.,help='ASSUMED slippage, bps of linear hedge turnover')
    parser.add_argument('--hedges-per-month',type=int,default=30,help='ASSUMED discrete rehedges per 30 days')
    parser.add_argument('--uncertainty-vol-pts',type=float,default=1.0,help='ASSUMED model reserve in vol points EACH SIDE')
    parser.add_argument('--jump-vol-pts',type=float,default=1.0,help='ASSUMED gap/tail reserve in vol points EACH SIDE')
    parser.add_argument('--historical-iv-csv',type=Path,help='Optional GENUINE timestamped historical IV observations CSV asset,asof_utc,atm30_iv (decimal); not generated retrospectively')
    parser.add_argument('--backfill-deribit-history',action='store_true',
                        help='ONE-OFF: request past Deribit USDC option trades, derive near-ATM IV proxies; can require many API pages')
    parser.add_argument('--history-lookback-days',type=int,default=150,help='Historical lookback (40-730 days), default 150')
    parser.add_argument('--history-step-days',type=int,default=7,help='Weekly observation spacing, default 7')
    parser.add_argument('--history-window-hours',type=int,default=6,help='Hours of the UTC sample day near close, default 6')
    parser.add_argument('--history-max-pages',type=int,default=7,help='Max Deribit archive pages PER sample day; incomplete dates rejected')
    parser.add_argument('--history-page-size',type=int,default=1000,help='History trade page size (max 10000); default 1000')
    parser.add_argument('--historical-backfill-csv',type=Path,default=ROOT/'data'/'deribit_historical_iv_backfill.csv',
                        help='Persistent archive of genuine reconstructed historical trade-derived IV observations')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    requested=[v.strip().lower() for v in args.venues.split(',') if v.strip()]
    if not requested or any(v not in ('derive','deribit') for v in requested) or len(set(requested))!=len(requested):
        parser.error('--venues must be derive,deribit or either individual venue')
    if args.backfill_deribit_history and args.mode!='live':
        parser.error('--backfill-deribit-history requires --mode live')
    status={}
    if args.mode=='demo':
        if args.near_spot is None:args.near_spot=4.40
        if args.near_rv is None:args.near_rv=1.10
        pieces=[]
        if 'derive' in requested:
            pieces.append(sample_synthetic_peers());status['derive']='SYNTHETIC DEMO (not market data)'
        if 'deribit' in requested:
            pieces.append(sample_synthetic_deribit());status['deribit']='SYNTHETIC DEMO (not market data)'
        nodes=pd.concat(pieces,ignore_index=True)
        asof='2026-10-30'
    elif args.mode=='live':
        pieces=[]
        for venue in requested:
            try:
                piece=download_derivexyz() if venue=='derive' else download_deribit()
                pieces.append(piece)
                failed=piece.attrs.get('failures',{})
                status[venue]=f'OK: {len(piece)} nodes' + (f'; asset failures: {failed}' if failed else '')
            except Exception as exc:
                status[venue]=f'UNAVAILABLE: {exc}'
                print(f'WARNING: {venue}: {exc}',file=sys.stderr)
        if not pieces:
            raise RuntimeError('Both requested venues unavailable; no synthetic substitute. Check internet and APIs.')
        nodes=pd.concat(pieces,ignore_index=True)
        asof=datetime.now(timezone.utc).isoformat()
    else:
        if not args.peer_csv:parser.error('--mode csv requires --peer-csv')
        nodes=pd.read_csv(args.peer_csv)
        status={'csv':str(args.peer_csv)}
        asof=f'imported CSV {args.peer_csv}'
    if 'venue' not in nodes.columns:
        parser.error('Peer CSV must include venue column (derive or deribit).')
    if nodes.asset.nunique() < 2:
        raise RuntimeError('Need at least 2 DIFFERENT peer assets; available: ' + str(nodes.asset.unique()))
    print('Venue status:',status)
    kraken_meta = None
    rv_table = None
    near_prices = None
    if args.near_auto_history and args.near_closes_csv:
        parser.error('Choose --near-auto-history OR --near-closes-csv')
    if args.near_auto_history:
        prices, spot, kraken_meta = fetch_near_daily_prices()
        near_prices=prices
        rv_table = historical_volatility(prices)
        prices.to_csv(args.out/'near_daily_closes.csv',index=False)
        rv_table.to_csv(args.out/'near_historical_volatility.csv',index=False)
        if args.near_spot is None:args.near_spot = float(spot)
        if args.near_rv is None:args.near_rv = float(rv_table.loc[rv_table['lookback_days']==30,'realized_vol_decimal'].iloc[0])
        print('Kraken NEAR historical volatility (% annualised):')
        print(rv_table[['lookback_days','realized_vol_percent']].to_string(index=False))
    if args.near_closes_csv:
        near_prices=pd.read_csv(args.near_closes_csv)
        closes=near_prices['close'].astype(float).to_numpy()
        if len(closes)<31:parser.error('Need at least 31 consecutive daily closes')
        if any(closes<=0):parser.error('All closes must be positive')
        args.near_rv=float(np.std(np.diff(np.log(closes[-31:])),ddof=1)*np.sqrt(365))
        if args.near_spot is None:args.near_spot=float(closes[-1])
    if args.near_spot is None or args.near_rv is None:
        parser.error('Live/CSV mode needs --near-spot and --near-rv (or --near-closes-csv). No silent synthetic fallbacks.')
    if args.near_spot<=0:parser.error('spot must be positive')

    nodes.to_csv(args.out/'source_peer_option_nodes.csv',index=False)
    fits=fit_peer_smiles(nodes)
    fits.to_csv(args.out/'peer_smile_fits.csv',index=False)
    # Fair-mid calibration from peer IV versus TRAILING RV, NOT a directional trading signal.
    premium_source = 'manual input' if args.vrp_ratio is not None else 'not calculated'
    premium_estimate = None
    premium_table = pd.DataFrame()
    forward_scores = pd.DataFrame()
    all_peer_prices = pd.DataFrame()
    peer_history_errors = {}
    if args.mode == 'demo':
        # Fully fictitious peer RVs to exercise end-to-end premium calibration.
        # These are NOT historical observations or market premium estimates.
        fictional_relative={'HYPE':.12,'SOL':.10,'ZEC':.18,'AVAX':.13,'TRX':.08}
        rv_rows=[]
        for asset in sorted(fits.asset.unique()):
            if asset in fictional_relative:
                iv30=peer_at(fits,asset,30)['atm']
                rv_rows.append({'asset':asset,'rv30':iv30/(1+fictional_relative[asset]),
                                'last_close_day':'FICTIONAL'})
        premium_table=make_peer_premium_table(fits,pd.DataFrame(rv_rows),asof)
        if not premium_table.empty:
            premium_table.to_csv(args.out/'peer_current_iv_rv_premiums.csv',index=False)
            premium_estimate=robust_peer_premium(premium_table)
        if args.vrp_ratio is None:
            args.vrp_ratio=premium_estimate['premium_ratio']
            premium_source='SYNTHETIC peer-relative calibration (fictional peer RV; NOT market evidence)'
    if args.mode == 'live':
        peer_closes=[]
        if args.peer_closes_csv:
            peer_closes=[normalize_custom_closes(pd.read_csv(args.peer_closes_csv))]
        else:
            for asset in sorted(fits.asset.unique()):
                try:
                    peer_closes.append(fetch_peer_closes(asset))
                except Exception as exc:
                    peer_history_errors[asset]=str(exc)
                    print(f'WARNING peer historical data {asset}: {exc}',file=sys.stderr)
        if peer_closes:
            all_peer_prices=pd.concat(peer_closes,ignore_index=True)
            all_peer_prices.to_csv(args.out/'peer_daily_closes.csv',index=False)
            rv_peers=realised_from_prices(all_peer_prices)
            rv_peers.to_csv(args.out/'peer_historical_volatility.csv',index=False)
            premium_table=make_peer_premium_table(fits,rv_peers,asof)
            if not premium_table.empty:
                premium_table.to_csv(args.out/'peer_current_iv_rv_premiums.csv',index=False)
                # Only store real observations. Same asset is represented once across venues.
                append_snapshots(args.premium_archive,premium_table)
                # Genuine subsequent-RV test uses the last completed close before quote day below.
                print('Peer IV30 versus TRAILING RV30 (NOT forward outperformance):')
                print(premium_table[['asset','venues','atm30_iv','trailing30_rv',
                                     'iv_minus_rv_vol_points','relative_premium']].to_string(index=False))
                try:
                    premium_estimate=robust_peer_premium(premium_table)
                except ValueError as exc:
                    print('WARNING peer calibration:',exc,file=sys.stderr)
        if args.vrp_ratio is None:
            if premium_estimate is None:
                raise RuntimeError('Automatic peer premium needs 2+ qualifying peer price series. '
                                   'Use --peer-closes-csv PATH with asset,date,close; or explicitly '
                                   'set --vrp-ratio 0.08 as a documented manual assumption. '
                                   'No silent 8% fallback.')
            args.vrp_ratio=premium_estimate['premium_ratio']
            premium_source='peer-calibrated: quality-weighted Huber estimate of current IV30 / trailing RV30'
        else:
            premium_source='manual user override (peer table computed for reference)'
    elif args.mode == 'csv' and args.vrp_ratio is None:
        raise RuntimeError('CSV mode requires --vrp-ratio decimal; no invented historic premium')
    if not -0.5 < args.vrp_ratio < 1.0:
        raise RuntimeError(f'Peer/manual relative uplift {args.vrp_ratio:.2%} outside model range (-50%, +100%). Review your data.')
    # NEAR forward-RV is now explicitly estimated from historical 7/30/90d variance.
    trailing_rv30=float(args.near_rv)
    window_inputs={30:trailing_rv30}
    if rv_table is not None:
        window_inputs.update({int(r.lookback_days):float(r.realized_vol_decimal) for r in rv_table.itertuples(index=False)})
    elif near_prices is not None and {'date','close'}.issubset(near_prices.columns):
        from src.near_data import historical_volatility as hv_func
        tr=hv_func(near_prices,windows=(7,30,90))
        window_inputs.update({int(r.lookback_days):float(r.realized_vol_decimal) for r in tr.itertuples(index=False)})
    # Retain the old fixed blend as a transparent benchmark rather than silently
    # treating its subjective weights as an optimised forward-vol forecast.
    forward_forecast=forecast_near_rv(window_inputs)
    forward_rv30=forward_forecast['forecast_rv']
    forecast_comparison=None
    rv_forecast_backtest=pd.DataFrame()
    if near_prices is not None and {'date','close'}.issubset(near_prices.columns):
        rv_forecast_backtest=rolling_forward_rv_backtest(near_prices)
        if not rv_forecast_backtest.empty:
            rv_forecast_backtest.to_csv(args.out/'near_forward_rv_backtest.csv',index=False)
        try:
            comparison, scores, forecast_comparison=compare_forecasts(near_prices)
            comparison.to_csv(args.out/'near_forecast_comparison.csv',index=False)
            scores.to_csv(args.out/'near_forecast_model_scores.csv',index=False)
            save_holdout_chart(scores,args.out/'05_near_forecast_holdout.png',forecast_comparison['deployed_method'])
            (args.out/'near_forecast_selected.json').write_text(
                json.dumps(forecast_comparison,indent=2), encoding='utf-8')
            # Research-only candidate; pricing stays with original, disclosed fixed blend.
            forward_rv30=forecast_near_rv(window_inputs)['forecast_rv']
            forecast_comparison['pricing_policy']='fixed_blend_baseline; experimental HAR not auto-deployed'
            forward_forecast={**forward_forecast,
                'forecast_rv':forward_rv30,
                'method':'fixed 7/30/90 variance blend; HAR only research challenger',
                'fixed_blend_reference':forecast_near_rv(window_inputs)['forecast_rv'],
                'selection_basis':'conservative fixed baseline; research challenger evaluated separately',
                'assumptions':'Earlier dates select a candidate; later holdout is an acceptance gate, not untouched final validation. Overlapping outcomes; no option IV information used.'}
            print('Forecast model comparison (later holdout; lower MAE is better):')
            print(scores[scores.segment.eq('holdout')][['method','n','mae_vol_points','bias_vol_points']].to_string(index=False))
            print('Early candidate:',forecast_comparison['selected_method'],
                  '| Deployed after holdout gate:',forecast_comparison['deployed_method'],
                  '| Holdout improvement vs old blend:',
                  f"{forecast_comparison['holdout_improvement_vs_fixed_vol_points']:+.2f} vol points")
        except ValueError as err:
            # Fallback is declared explicitly; never claim a calibrated forecast.
            forecast_comparison={'status':'not_available', 'reason':str(err),
                                'selected_method':'Fixed blend (fallback)',
                                'forecast_rv':forward_rv30}
            print('WARNING: walk-forward model comparison unavailable:',err,file=sys.stderr)
    # Optional one-off, rate-bounded historical trade backfill. Does not
    # relabel trade IV as historical option-chain mid quotes.
    historical_iv_scores=pd.DataFrame()
    backfill_stats={'attempted':False,'status':'not_requested','qualified_observations':0,
                    'successful_windows':0,'partial_windows_rejected':0,'error_windows':0}
    historical_iv_source='No historical IV observations loaded; genuine ex-post test pending.'
    archives=[]
    if args.backfill_deribit_history:
        print('Historical Deribit backfill: collecting completed UTC sampling windows; may take several minutes.')
        observations,audit,trade_detail=backfill_history(
            asof_utc=asof,lookback_days=args.history_lookback_days,
            step_days=args.history_step_days,window_hours=args.history_window_hours,
            max_pages=args.history_max_pages,page_size=args.history_page_size)
        audit.to_csv(args.out/'historical_deribit_fetch_audit.csv',index=False)
        if not trade_detail.empty:
            trade_detail.to_csv(args.out/'historical_deribit_qualified_trades.csv',index=False)
        if not observations.empty:
            args.historical_backfill_csv.parent.mkdir(parents=True,exist_ok=True)
            old=pd.read_csv(args.historical_backfill_csv) if args.historical_backfill_csv.exists() else pd.DataFrame()
            comb=pd.concat([old,observations],ignore_index=True)
            comb=comb.drop_duplicates(['asset','source_window_day'],keep='last')
            comb.to_csv(args.historical_backfill_csv,index=False)
        backfill_stats={'attempted':True,'status':'complete' if not audit.status.eq('api_error').any() else 'partial_api_errors',
            'qualified_observations':int(len(observations)),
            'successful_windows':int(audit.status.eq('complete').sum()),
            'partial_windows_rejected':int(audit.status.isin(['page_cap_incomplete','timestamp_cursor_ambiguous','pagination_stalled','unsorted_api_response','invalid_timestamp']).sum()),
            'error_windows':int(audit.status.eq('api_error').sum())}
        print('Deribit history audit:',backfill_stats)
    if args.historical_backfill_csv.exists():
        archives.append(pd.read_csv(args.historical_backfill_csv))
        historical_iv_source='Deribit historical option trade IV near-ATM proxy (NOT a historical bid/ask mid)'
    if args.historical_iv_csv:
        archives.append(pd.read_csv(args.historical_iv_csv))
        historical_iv_source+='; plus user-supplied dated historical ATM IV'
    elif args.premium_archive.exists():
        archives.append(pd.read_csv(args.premium_archive))
        historical_iv_source+='; plus forward-collected current market snapshots'
    if archives:
        archive_input=pd.concat(archives,ignore_index=True)
        req={'asset','asof_utc','atm30_iv'}
        if not req.issubset(archive_input):
            raise RuntimeError('Historical IV data require asset,asof_utc,atm30_iv')
        archive_input['asof_utc']=pd.to_datetime(archive_input.asof_utc,utc=True,errors='coerce')
        archive_input['atm30_iv']=pd.to_numeric(archive_input.atm30_iv,errors='coerce')
        archive_input=archive_input.dropna(subset=['asof_utc','atm30_iv']).sort_values('asof_utc')
        # Preserve one per asset per UTC day, prioritizing trade-derived near-ATM
        # source when available (directly traceable back to historical API).
        archive_input['day']=archive_input.asof_utc.dt.normalize()
        archive_input['priority']=archive_input.get('source',pd.Series('',index=archive_input.index)).astype(str).str.contains('historical option trades',case=False).astype(int)
        archive_input=archive_input.sort_values(['priority','asof_utc']).drop_duplicates(['asset','day'],keep='last')
        archive_input.drop(columns=['priority','day']).to_csv(args.out/'historical_iv_observations_used.csv',index=False)
    else:
        archive_input=pd.DataFrame(columns=['asset','asof_utc','atm30_iv'])
    if premium_estimate and premium_estimate.get('leave_one_out'):
        pd.DataFrame(premium_estimate['leave_one_out']).to_csv(args.out/'peer_premium_leave_one_out.csv',index=False)
    if not archive_input.empty and not all_peer_prices.empty:
        historical_iv_scores=historical_iv_vs_future_rv(archive_input,all_peer_prices)
        if not historical_iv_scores.empty:
            historical_iv_scores.to_csv(args.out/'historical_iv_vs_subsequent_rv.csv',index=False)
    hist_summary={'observations':int(len(historical_iv_scores)),
        'assets':sorted(historical_iv_scores.asset.unique().tolist()) if not historical_iv_scores.empty else [],
        'mean_vol_spread_pts':float(historical_iv_scores.iv_minus_subsequent_rv_vol_pts.mean()) if not historical_iv_scores.empty else None,
        'median_vol_spread_pts':float(historical_iv_scores.iv_minus_subsequent_rv_vol_pts.median()) if not historical_iv_scores.empty else None,
        'positive_spread_fraction':float(historical_iv_scores.iv_minus_subsequent_rv_vol_pts.gt(0).mean()) if not historical_iv_scores.empty else None,
        'mean_variance_spread':float(historical_iv_scores.iv_squared_minus_future_rv_squared.mean()) if not historical_iv_scores.empty else None,
        'warning':'Actual traded option IV, not historical two-sided marks; overlapping weekly windows, market regime and selection bias. No profitability inference.'}
    if not historical_iv_scores.empty:
        historical_iv_scores.groupby('asset').agg(
            n=('asset','size'),mean_diff_vol_pts=('iv_minus_subsequent_rv_vol_pts','mean'),
            median_diff_vol_pts=('iv_minus_subsequent_rv_vol_pts','median'),
            mean_variance_diff=('iv_squared_minus_future_rv_squared','mean')
        ).reset_index().to_csv(args.out/'historical_iv_rv_by_asset.csv',index=False)
    surface=NearSurface(fits,forward_rv30,args.vrp_ratio)
    print(f'Fair-mid anchor premium: {args.vrp_ratio:+.2%} of 30d RV | source: {premium_source}')
    spot=args.near_spot
    cost_assumptions={'fee_bps':args.hedge_fee_bps,'half_spread_bps':args.hedge_halfspread_bps,
                      'slippage_bps':args.hedge_slippage_bps,'hedges_per_month':args.hedges_per_month,
                      'uncertainty_vol_pts':args.uncertainty_vol_pts,'jump_vol_pts':args.jump_vol_pts}
    def mk_quote(strike, right):
        if args.spread_vol_pts is not None:
            return surface.quote(spot,strike,30,right,spread_vol_pts=args.spread_vol_pts)
        return cost_aware_two_way_quote(surface,spot,strike,30,right,**cost_assumptions)
    tickets={
        'ATM call':mk_quote(spot,'C'),
        '20% OTM covered call':mk_quote(spot*1.20,'C'),
        '15% OTM cash-secured put':mk_quote(spot*.85,'P')
    }
    pd.DataFrame([{'name':k,**v} for k,v in tickets.items()]).to_csv(args.out/'otc_option_tickets.csv',index=False)
    short_stress=hedged_stress(surface,tickets['ATM call'],10000)
    short_stress.to_csv(args.out/'stress_atm_call_short.csv',index=False)
    # Equal representation of dealer SELL-at-offer and BUY-at-bid scenarios.
    both=short_stress.copy()
    both['dealer_action']='SELL option at offer / buy delta hedge'
    both['entry_premium_per_token']=tickets['ATM call']['ask']
    long=short_stress.copy()
    long['dealer_action']='BUY option at bid / sell delta hedge'
    long['entry_premium_per_token']=tickets['ATM call']['bid']
    # Long option is repriced at same future scenario mark (not negative short P&L,
    # because bid/ask are different entry cash flows).
    long['pnl_before_costs']=(long['option_mark']-tickets['ATM call']['bid'])*10000 - (tickets['ATM call']['spot_delta']*10000*tickets['ATM call']['spot']*long['underlying_move_pct']/100)
    pd.concat([both,long],ignore_index=True).to_csv(args.out/'stress_atm_call_both_sides.csv',index=False)
    issues=simple_arbitrage_checks(surface,spot)
    charts(fits,surface,spot,args.out,args.mode)
    format_report(args.mode,asof,spot,forward_rv30,args.vrp_ratio,surface.weights,fits,surface,
                  tickets,issues,args.out, float(tickets['ATM call'].get('spread_vol_pts',args.spread_vol_pts or 0)))
    if premium_source:
        report_path=args.out/'case_study.md'
        report_text=report_path.read_text(encoding='utf-8')
        report_text=report_text.replace('premium assumption', f'premium calibration ({premium_source})')
        report_text += f'\n\n### IV/RV calibration provenance\n\n{premium_source}. '
        report_text += 'Current IV versus **trailing** RV does not demonstrate that options were overpriced. '
        report_text += 'Both bid and offer are quoted around the same fair surface; spreads are cost- and risk-ASSUMPTION buffers, not a bullish or bearish trade.\n'
        report_text += f'\nForward RV30 proxy: {forward_rv30:.3f} compared with trailing RV30 {trailing_rv30:.3f}. '
        report_text += f'Historical IV vs subsequent RV observations evaluated: {len(historical_iv_scores)}.\n'
        report_path.write_text(report_text,encoding='utf-8')
    meta={'mode':args.mode,'asof':asof,'source':'Derive / Deribit public market data' if args.mode=='live' else 'SYNTHETIC DEMO' if args.mode=='demo' else 'USER CSV',
          'venue_status':status,'venue_assets':{k:sorted(nodes.loc[nodes.venue==k,'asset'].unique().tolist()) for k in nodes.venue.unique()},
          'near_spot':spot,'near_rv_30':trailing_rv30,'near_forward_rv_30':forward_rv30,
          'forward_forecast':forward_forecast,
          'near_forecast_model_comparison':forecast_comparison,
          'near_rv_forecast_backtest':{'n':len(rv_forecast_backtest),
                                     'mae_vol_points':float(rv_forecast_backtest.absolute_error_vol_points.mean()) if not rv_forecast_backtest.empty else None,
                                     'mean_error_vol_points':float(rv_forecast_backtest.error_vol_points.mean()) if not rv_forecast_backtest.empty else None,
                                     'note':'rolling weekly forecast, overlapping 30d future windows, not option IV backtest'},
          'historical_iv_forward_evaluation':{'n':len(historical_iv_scores),'source':historical_iv_source,
               'mean_iv_minus_future_rv_pts':hist_summary['mean_vol_spread_pts'],
               'median_iv_minus_future_rv_pts':hist_summary['median_vol_spread_pts'],
               'assets':hist_summary['assets'],
               'positive_spread_fraction':hist_summary['positive_spread_fraction'],
               'mean_variance_spread':hist_summary['mean_variance_spread'],
               'limitations':hist_summary['warning']},
          'historical_deribit_backfill':backfill_stats,
          'vrp_ratio':args.vrp_ratio,
          'near_atm_iv_30':surface.atm30,'weights':surface.weights,'arbitrage_flags':issues,
          'premium_source':premium_source,'premium_method':premium_estimate,
          'premium_peer_count':int(len(premium_table)),'premium_peer_rows':premium_table.to_dict(orient='records') if not premium_table.empty else [],
          'premium_archive':str(args.premium_archive),'historical_forward_eval_rows':int(len(historical_iv_scores)),
          'peer_history_errors':peer_history_errors,
          'spread_vol_pts':float(tickets['ATM call'].get('spread_vol_pts',args.spread_vol_pts or 0)),
          'spread_mode':'fixed override' if args.spread_vol_pts is not None else 'cost / risk assumption-based, symmetric by side',
          'hedge_cost_assumptions':cost_assumptions,
          'quoting_policy':'neutral two-way: same fair IV mid; cost/risk half width on each side; inventory tilt zero',
          'option_tickets':tickets,'kraken_history':kraken_meta,
          'historical_vol_windows':rv_table.to_dict(orient='records') if rv_table is not None else None,
          'disclaimer':'Indicative research prototype. NOT executable market quote.'}
    (args.out/'model_run.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('NEAR forecast RV30:',f'{forward_rv30*100:.2f}%','vs trailing RV30:',f'{trailing_rv30*100:.2f}%')
    print('Historical IV vs SUBSEQUENT RV score count:',len(historical_iv_scores),'(not a performance claim)')
    print(f'[{args.mode.upper()}] NEAR ATM30 IV: {surface.atm30*100:.2f}%, peers {surface.weights}')
    print(pd.read_csv(args.out/'otc_option_tickets.csv')[['name','strike','iv','bid','mid','ask','spot_delta']].to_string(index=False))
    print('Arbitrage grid flags:', issues or 'none detected on sampled grid')
    print('Generated:',args.out)
    if args.pdf:
        subprocess.run([sys.executable, str(ROOT/'make_pdf.py'), '--out', str(args.out)], check=True)

if __name__=='__main__':main()
