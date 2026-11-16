"""Optional observed-data diagnostics for the 12-Nov NEAR client case study.

All outputs distinguish measured historical/market data from model assumptions.
No external data automatically overrides the pinned fair-volatility forecast.
"""
from __future__ import annotations
from typing import Any
import math


def build_supplemental_scenarios(extra: dict, *, atm_spot: float, atm_fair_iv: float,
                                  quote_func, fee_bps: float = 5.0,
                                  contingency_bps: float = 5.0) -> dict:
    """Converts a contemporaneous two-sided perp book into an *indicative* hedge scenario.

    Buy/sell book impact already includes the half-spread, so do NOT add a second
    half-spread. Fee and slippage contingency remain explicit assumptions.
    This is ONE observed order-book snapshot, not verified hedging P&L.
    """
    result: dict[str, Any] = {'use_in_fair_iv': False, 'use_in_baseline_quote': False,
                              'policy': 'extra feeds are diagnostics and a separate TCA scenario; baseline quote unchanged'}
    hour = extra.get('hourly_rv', {})
    if isinstance(hour, dict) and math.isfinite(float(hour.get('close_to_close_hourly_rv_30d_pct', 'nan'))):
        result['hourly_rv_30_pct'] = float(hour['close_to_close_hourly_rv_30d_pct'])
        result['hourly_method'] = 'Coinbase completed hourly close-to-close returns; annualised sqrt(24*365)'
    validation = extra.get('extended_forecast_validation', {})
    if isinstance(validation,dict) and validation.get('status')=='ok':
        result['extended_holdout_scores'] = validation.get('holdout_scores', [])
        result['nonoverlap_holdout_dates'] = validation.get('n_nonoverlap_holdout')
    perp = extra.get('perp_book_and_funding', {})
    if not isinstance(perp, dict) or perp.get('status')!='ok':
        result['perp_tca_status'] = 'unavailable'
        return result
    try:
        b = float(perp['buy_impact_bps'])
        s = float(perp['sell_impact_bps'])
        if not (math.isfinite(b) and math.isfinite(s)):
            raise ValueError('nonfinite buy/sell impact')
        if b<0 or s<0:raise ValueError('negative book impact')
        if float(perp['rfq_hedge_notional_usd'])<=0:raise ValueError('invalid notional')
        # Use worse of the two directions to preserve neutral two-way quote policy.
        measured_book_impact_bps=max(b,s)
        assumed_fee=float(fee_bps); assumed_buffer=float(contingency_bps)
        total_execution_bps=measured_book_impact_bps+assumed_fee+assumed_buffer
        q=quote_func(atm_spot,atm_spot,'C',atm_fair_iv,
                     execution_bps=total_execution_bps,hedges=30)
        result['perp_tca_status']='indicative_book_informed_scenario'
        result['book_informed_tca']={
            'source':perp.get('source','Hyperliquid NEAR perp'),
            'observation_time_utc':perp.get('book_time_utc'),
            'hedge_notional_usd':float(perp['rfq_hedge_notional_usd']),
            'buy_impact_bps_including_spread':b,
            'sell_impact_bps_including_spread':s,
            'worse_direction_impact_bps':measured_book_impact_bps,
            'assumed_fee_bps':assumed_fee,
            'assumed_extra_slippage_bps':assumed_buffer,
            'cost_bps_per_rehedge':total_execution_bps,
            'hedges_in_30d_assumed':30,
            'atm_bid_usd':q['bid_usd'],'atm_offer_usd':q['offer_usd'],
            'full_spread_vol_pts':q['total_spread_vol_points'],
            'last_30d_observed_funding_sum_pct':perp.get('past_30d_funding_sum_pct'),
            'perp_basis_vs_oracle_bps':perp.get('perp_basis_vs_oracle_bps'),
            'status_note':'one L2 snapshot; NOT a calibrated 30-day execution cost, not executable option spread; baseline price unchanged'
        }
    except (ValueError,TypeError,KeyError) as exc:
        result['perp_tca_status']='unusable_book_data'
        result['perp_tca_reason']=str(exc)
    return result
