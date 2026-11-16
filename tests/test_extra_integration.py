import json
from pathlib import Path
from src.extra_research import build_supplemental_scenarios
from desk_update import estimate_two_way


def test_book_informed_symmetric_neutral_quote():
    extra={'perp_book_and_funding': {'status':'ok','source':'Hyperliquid NEAR',
        'buy_impact_bps':8,'sell_impact_bps':12,'rfq_hedge_notional_usd':20000,
        'book_time_utc':'2026-11-12T08:00:00+00:00'}}
    r=build_supplemental_scenarios(extra,atm_spot=3.51,atm_fair_iv=1.189,
                                    quote_func=estimate_two_way)
    t=r['book_informed_tca']
    assert t['cost_bps_per_rehedge']==22  # 12 observed incl spread + 5 assumed fee + 5 buffer
    assert t['atm_bid_usd'] < t['atm_offer_usd']
    assert r['use_in_fair_iv'] is False and r['use_in_baseline_quote'] is False


def test_incomplete_book_does_not_create_quote():
    extra={'perp_book_and_funding':{'status':'ok','buy_impact_bps':None,
            'sell_impact_bps':5,'rfq_hedge_notional_usd':20000}}
    r=build_supplemental_scenarios(extra,atm_spot=3.51,atm_fair_iv=1.189,
                                    quote_func=estimate_two_way)
    assert r['perp_tca_status']=='unusable_book_data'
    assert 'book_informed_tca' not in r


def test_hourly_and_longer_backtest_are_research_only():
    extra={'hourly_rv':{'close_to_close_hourly_rv_30d_pct':103.2},
        'extended_forecast_validation':{'status':'ok','n_nonoverlap_holdout':6,
            'holdout_scores':[{'model':'Fixed blend','mae_vol_pts':24.5},
                              {'model':'HAR ridge','mae_vol_pts':25.2}]}}
    r=build_supplemental_scenarios(extra,atm_spot=3.51,atm_fair_iv=1.189,
                                    quote_func=estimate_two_way)
    assert r['hourly_rv_30_pct']==103.2
    assert len(r['extended_holdout_scores'])==2
    assert r['use_in_fair_iv'] is False
