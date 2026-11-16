"""Read-only Derive v3 options data connector for the NEAR OTC research project.

Derive migrated its public API from https://api.lyra.finance (v2) to
https://api.derive.xyz/v3. v3 has different instrument discovery methods,
and the old endpoint used in the project's earlier version returns HTTP 503.

Documentation: https://docs.derive.xyz/openapi.json
No API key, trading account, signing key or wallet is needed.

This module intentionally never substitutes synthetic values in live mode.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

BASE = 'https://api.derive.xyz/v3'
ASSETS = ('HYPE', 'SOL', 'ZEC')
DAYS = 365.0
# v3 permits fractional strikes with underscore and optional risk-universe suffix.
PAT = re.compile(r'^(HYPE|SOL|ZEC)-(\d{8})-([0-9]+(?:[_.][0-9]+)?)-([CP])(?:-\d+)?$')


def _num(x, default=float('nan')):
    try:
        n = float(x)
        return n if np.isfinite(n) else default
    except (ValueError, TypeError):
        return default


def post_public(session: requests.Session, method: str, params: dict, attempts: int = 3):
    """GET public v3 data by POST. Accept JSON-RPC result envelopes."""
    last = None
    for i in range(attempts):
        try:
            response = session.post(
                f'{BASE}/public/{method}', json=params,
                headers={'User-Agent': 'NEAR-OTC-Surface-Research/1.2',
                         'Accept': 'application/json', 'Content-Type': 'application/json'},
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, (list, dict)):
                raise ValueError(f'Unexpected Derive v3 {method} response type: {type(data)}')
            if isinstance(data, dict):
                if data.get('error') is not None:
                    raise RuntimeError(f'Derive v3 {method} returned {data["error"]}')
                data = data.get('result', data)
            return data
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last = exc
            if i + 1 < attempts:
                time.sleep(0.65 * (i + 1))
    raise RuntimeError(f'Derive v3 public/{method} failed: {last}')


def _expiry_from_name(instrument: str):
    """Conventional 08:00 UTC Derive expiry fallback from instrument's name.
    v3 docs caution that some underlyings use another configured settlement time.
    This project's peers normally use 08:00; treat resulting DTE as approximate.
    """
    m = PAT.fullmatch(instrument)
    if not m:
        return None
    try:
        exp = datetime.strptime(m.group(2), '%Y%m%d').replace(tzinfo=timezone.utc, hour=8)
    except ValueError:
        return None
    return m, exp


def discover_option_expiries(live_names: list[str], asof: datetime):
    """Find required expiries from Derive v3 public/get_all_live_instruments."""
    expiries = {asset: set() for asset in ASSETS}
    for instrument in live_names:
        if not isinstance(instrument, str):
            continue
        parsed = _expiry_from_name(instrument)
        if not parsed:
            continue
        m, exp = parsed
        dte = (exp - asof).total_seconds()/86400
        if 2 <= dte <= 180:
            expiries[m.group(1)].add(m.group(2))
    return expiries


def parse_tickers(payload: dict, asset: str, asof: datetime,
                  max_age_minutes: float = 20.0) -> list[dict]:
    """Normalize v3 slim tickers into the exact schema expected by the model.

    Two-sided markets require both bid/ask sizes, valid non-crossed prices,
    both implied vols, and a width under 40 vol points. Mark-only values
    are explicitly tagged and downweighted, not called executable prices.
    """
    body = payload.get('tickers', {}) if isinstance(payload, dict) else None
    if not isinstance(body, dict):
        raise ValueError('Derive v3 get_tickers returned an unexpected schema (tickers map missing)')
    rows = []
    for ticker_name, raw in body.items():
        if not isinstance(raw, dict):
            continue
        item = raw.get('instrument_ticker', raw)
        if not isinstance(item, dict):
            continue
        instrument = item.get('instrument_name', str(ticker_name))
        parsed = _expiry_from_name(instrument)
        if not parsed:
            continue
        m, exp = parsed
        if m.group(1) != asset:
            continue
        days = (exp - asof).total_seconds()/86400
        if not 2 <= days <= 180:
            continue
        pricing = item.get('option_pricing') or {}
        if not isinstance(pricing, dict):
            continue
        iv = _num(pricing.get('iv', pricing.get('i')))
        if not 0.05 < iv < 5.0:
            continue
        forward = _num(pricing.get('forward_price', pricing.get('f')))
        if not forward > 0:
            forward = _num(item.get('index_price', item.get('I')))
        if not forward > 0:
            continue
        stamp = _num(item.get('timestamp', item.get('t')))
        if not np.isfinite(stamp):
            continue
        ts = stamp / 1000.0 if stamp > 10**11 else stamp
        age_minutes = (asof.timestamp() - ts) / 60
        if age_minutes > max_age_minutes or age_minutes < -10:
            continue
        bid_iv = _num(pricing.get('bid_iv', pricing.get('bi')))
        ask_iv = _num(pricing.get('ask_iv', pricing.get('ai')))
        bid_size = _num(item.get('best_bid_amount', item.get('B')), 0.0)
        ask_size = _num(item.get('best_ask_amount', item.get('A')), 0.0)
        bid_px = _num(item.get('best_bid_price', item.get('b')), 0.0)
        ask_px = _num(item.get('best_ask_price', item.get('a')), 0.0)
        two_sided = (
            bid_size > 0 and ask_size > 0 and bid_px > 0 and ask_px > bid_px
            and np.isfinite(bid_iv) and np.isfinite(ask_iv)
            and bid_iv > 0 and 0 < ask_iv - bid_iv < 0.40
        )
        if two_sided:
            used_iv = (bid_iv + ask_iv) / 2
            quality = 'two_sided'
        elif bid_px > 0 or ask_px > 0:
            used_iv = iv
            quality = 'one_sided'
        else:
            used_iv = iv
            quality = 'mark_only'
        strike = float(m.group(3).replace('_', '.'))
        right = m.group(4)
        if (right == 'C' and strike < forward * 0.999) or (right == 'P' and strike > forward * 1.001):
            continue
        rows.append({
            'asset': asset, 'venue': 'derive', 'expiry': exp.isoformat(),
            'dte': round(days, 4), 'strike': strike, 'right': right,
            'forward': forward, 'iv': float(used_iv), 'mark_iv': iv,
            'bid_iv': bid_iv, 'ask_iv': ask_iv, 'quality': quality,
            'quality_weight': {'two_sided': 1.0, 'one_sided': 0.28, 'mark_only': 0.16}[quality],
            'source': 'derive_v3_public_api', 'asof': asof.isoformat(),
            'instrument': instrument,
            'delta': _num(pricing.get('delta', pricing.get('d'))),
        })
    return rows


def download_derivexyz(asof=None):
    """Download live Derive v3 option marks/quotes. Never fall back to demo data."""
    asof = asof or datetime.now(timezone.utc)
    all_rows = []
    failures = {}
    with requests.Session() as session:
        live_names = post_public(session, 'get_all_live_instruments', {})
        if isinstance(live_names, dict):
            live_names = live_names.get('instruments', live_names.get('instrument_names'))
        if not isinstance(live_names, list):
            raise RuntimeError('Derive v3 discovery did not return a list of live instrument names')
        expiries_by_asset = discover_option_expiries(live_names, asof)
        for asset in ASSETS:
            expiries = sorted(expiries_by_asset[asset])
            if not expiries:
                failures[asset] = 'No live options listed for this asset in the 2-180-day research window'
                continue
            before = len(all_rows)
            for expiry in expiries:
                try:
                    payload = post_public(session, 'get_tickers', {
                        'currency': asset,
                        'instrument_type': 'option',
                        'expiry_date': int(expiry),
                    })
                    all_rows.extend(parse_tickers(payload, asset, asof))
                except Exception as exc:
                    failures[f'{asset}/{expiry}'] = str(exc)
            if len(all_rows) == before:
                failures[asset] = 'Listed option expiries found, but no usable fresh IV points survived filtering'
    if not all_rows:
        raise RuntimeError(f'Derive v3 returned no usable fresh option nodes. Issues: {failures}')
    frame = pd.DataFrame(all_rows)
    frame.attrs['failures'] = failures
    return frame


def sample_synthetic_peers(asof='2026-10-30'):
    """Original clearly synthetic Derive example data for demo mode only."""
    anchor = {
        'HYPE': {'spot': 25., 'atm': (.88, .7984, .77),
                 'a': (.055, .035, .025), 'b': (.065, .055, .055)},
        'SOL': {'spot': 160., 'atm': (.53, .4681, .49),
                'a': (-.015, -.008, -.020), 'b': (.045, .05, .05)},
        'ZEC': {'spot': 250., 'atm': (1.24, 1.1412, 1.05),
                'a': (-.06, -.055, -.04), 'b': (.085, .09, .07)},
    }
    rows = []
    date = pd.Timestamp(asof, tz='UTC')
    for asset, info in anchor.items():
        for i, days in enumerate((7, 30, 90)):
            maturity = date + pd.Timedelta(days=days)
            atm = info['atm'][i]
            for z in (-1.8, -1.4, -1.0, -.65, -.3, 0, .3, .65, 1., 1.4, 1.8):
                for right in (('P', 'C') if z == 0 else ('P',) if z < 0 else ('C',)):
                    strike = info['spot'] * np.exp(z * atm * np.sqrt(days / DAYS))
                    iv = atm * (1 + info['a'][i]*z + info['b'][i]*z*z)
                    rows.append({
                        'asset': asset, 'venue': 'derive', 'expiry': maturity.isoformat(),
                        'dte': days, 'strike': round(strike, 5), 'right': right,
                        'forward': info['spot'], 'iv': iv, 'mark_iv': iv,
                        'bid_iv': np.nan, 'ask_iv': np.nan,
                        'quality': 'synthetic_test', 'quality_weight': 1.,
                        'source': 'SYNTHETIC_DEMO_NOT_REAL_QUOTE', 'asof': asof,
                        'instrument': f'DEMO-{asset}-{days}d-{right}', 'delta': np.nan,
                    })
    return pd.DataFrame(rows)
