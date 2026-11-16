"""Reproduce the strict 12-Nov-2026 peer SVI fit from the ORIGINAL local raw chain.

This file never downloads fresh data and never substitutes an archived/synthetic chain.
The CSV is deliberately local-only (outputs/ is ignored by Git).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

SNAPSHOT = pd.Timestamp('2026-11-12T07:41:41.643402+00:00')
ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / 'examples' / 'derived_2026-11-12'


def require(condition: bool, message: str):
    if not condition:
        raise RuntimeError(message)


def check_data(raw: Path) -> dict:
    require(raw.is_file(), f'Original 12-Nov raw CSV not found: {raw}')
    source = pd.read_csv(raw)
    require('asof' in source.columns, "Raw CSV has no 'asof' column")
    require({'asset', 'source', 'bid_iv', 'ask_iv', 'strike', 'expiry', 'forward'}.issubset(source.columns),
            'Raw file is not a valid original option-chain export')
    dates = pd.to_datetime(source['asof'], utc=True, errors='coerce')
    near = (dates - SNAPSHOT).abs() <= pd.Timedelta(minutes=20)
    require(near.any(), 'No source observations within 20 minutes of the 12-Nov-2026 07:41 UTC snapshot. '
            'Do not use October 30 or a fresh market download.')
    require(near.sum() >= 100, f'Only {int(near.sum())} rows near the original snapshot; '
            'check that you selected the complete source_peer_option_nodes.csv')
    return {'raw_file': str(raw), 'raw_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
            'input_rows': int(len(source)), 'dated_rows_within_20min': int(near.sum()),
            'timestamp_min_utc': dates.min().isoformat(), 'timestamp_max_utc': dates.max().isoformat()}


def check_outputs(output: Path) -> dict:
    result = {}
    audit = pd.read_csv(output / 'peer_svi_strict_slice_audit.csv')
    ref_audit = pd.read_csv(REFERENCE / 'peer_svi_strict_slice_audit.csv')
    accepted = audit[audit.strict_status == 'accepted']
    result['accepted_slices'] = int(len(accepted))
    result['accepted_30d_assets'] = sorted(accepted.asset.unique().tolist())
    require(len(accepted) == 9, f'Expected 9 strict accepted slices, got {len(accepted)}')
    require(set(accepted.asset) == {'AVAX', 'HYPE', 'SOL'},
            f'Unexpected strict assets: {sorted(set(accepted.asset))}')
    require(not ((audit.asset == 'ZEC') & (audit.strict_status == 'accepted')).any(),
            'ZEC unexpectedly passed the strict calibration; investigate before publication')
    keys = ['asset', 'venue', 'expiry']
    both = audit.merge(ref_audit, on=keys, suffixes=('_actual', '_reference'), how='outer', indicator=True)
    require(both['_merge'].eq('both').all(), 'The accepted source slice set changed vs archived 12-Nov audit')
    mismatches = (both.strict_status_actual != both.strict_status_reference).sum()
    require(mismatches == 0, f'{mismatches} source slices changed accepted/rejected status')
    result['source_slice_statuses_match'] = True

    fitted = pd.read_csv(output / 'peer_svi_strict_near_smile.csv')
    reference = pd.read_csv(REFERENCE / 'peer_svi_near_smile.csv')
    near = fitted.merge(reference, on='strike_over_forward', how='outer',
                        suffixes=('_actual', '_reference'), indicator=True)
    require(near['_merge'].eq('both').all(), 'Strikes changed vs published quote ladder')
    diff = np.abs(near.model_iv_pct_actual - near.model_iv_pct_reference)
    max_iv_diff = float(diff.max())
    result['max_abs_smile_diff_vol_points'] = max_iv_diff
    result['atm_iv_pct'] = float(fitted.loc[np.isclose(fitted.strike_over_forward, 1.0), 'model_iv_pct'].iloc[0])
    require(max_iv_diff <= 0.25,
            f'NEAR smile changed by up to {max_iv_diff:.3f} vol points vs saved reference')

    quotes = pd.read_csv(output / 'peer_svi_strict_9_contract_quotes.csv')
    ref_quotes = pd.read_csv(REFERENCE / 'peer_svi_9_contract_quote_ladder.csv')
    require(len(quotes) == 9, f'Expected 9 indicative contract rows, got {len(quotes)}')
    compared = quotes.merge(ref_quotes, on=['right','strike_over_forward'],
                            suffixes=('_actual','_reference'), indicator=True, how='outer')
    require(compared['_merge'].eq('both').all(), 'The quote ladder contracts changed')
    premium_errors = []
    for field in ('bid_usd_per_near', 'fair_usd_per_near', 'ask_usd_per_near'):
        premium_errors.append(np.abs(compared[f'{field}_actual'] - compared[f'{field}_reference']))
    result['max_abs_option_premium_diff_usd'] = float(np.maximum.reduce(premium_errors).max())
    require(result['max_abs_option_premium_diff_usd'] <= 0.005,
            'Strike quote premiums differ by more than half a cent; review inputs and package versions')
    require((quotes.bid_usd_per_near <= quotes.fair_usd_per_near).all() and
            (quotes.fair_usd_per_near <= quotes.ask_usd_per_near).all(),
            'Bid/mid/offer ordering failed')
    result['quotes_matched'] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description='12-Nov-2026 original option-chain SVI reproducibility check')
    parser.add_argument('--raw', default='outputs/source_peer_option_nodes.csv',
                        help='Path to original RAW 12 November source_peer_option_nodes.csv')
    parser.add_argument('--out', default='outputs/strict_reproduction_2026-11-12',
                        help='Separate directory for regenerated diagnostics (will not overwrite archived results)')
    args = parser.parse_args()
    raw = Path(args.raw).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'FAILED', 'snapshot_utc': SNAPSHOT.isoformat()}
    try:
        report.update(check_data(raw))
        cmd = [sys.executable, str(ROOT/'run_peer_svi_diagnostics.py'),
               '--nodes', str(raw), '--asof', SNAPSHOT.isoformat(), '--out', str(output)]
        print('Running strict SVI calibration against:', raw, flush=True)
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        report.update(check_outputs(output))
        report['status'] = 'PASS'
        print('\nPASS: original 12-Nov SVI results reproduced within stated tolerances.')
        print('Max IV difference: {:.4f} vol points'.format(report['max_abs_smile_diff_vol_points']))
        print('Max quote premium difference: ${:.6f}'.format(report['max_abs_option_premium_diff_usd']))
    except (RuntimeError, ValueError, KeyError, subprocess.CalledProcessError, OSError) as exc:
        report['error'] = str(exc)
        print('\nNOT YET VERIFIED:', exc)
        print('Do not publish the repository as a fully reproducible SVI calibration yet.')
    finally:
        (output/'reproducibility_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('Audit:', output/'reproducibility_check.json')
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
