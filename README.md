# NEAR OTC Options Pricing — Research Case Study

**Snapshot:** 12 November 2026, 07:41 UTC  
**Status:** Research prototype; **not** an executable quote or a production trading system.

## Overview

Python research workflow for indicative NEAR options prices where a sufficiently liquid NEAR listed-volatility market is unavailable. The study:

1. Estimates a baseline NEAR 30-day realised-volatility forecast using 7/30/90-day variance inputs and compares several forecast alternatives.
2. Calculates a peer-market ATM adjustment from HYPE, SOL, AVAX and ZEC 30-day implied volatility relative to their own trailing realised volatility. This *contemporaneous* IV/RV adjustment is not an observed variance risk premium.
3. Fits SVI slices to **two-sided Derive/Deribit peer options quotes**, normalises peer smiles by volatility-adjusted moneyness, and transfers a **provisional** strike shape to a 118.9% NEAR ATM anchor.
4. Converts strike-level theoretical IVs to Black-76 premiums/Greeks and models indicative two-way prices, hedge turnover and execution-cost sensitivities.
5. Discusses a 10,000-NEAR covered-call client RFQ and residual gamma, vega, theta, funding and counterparty/inventory risk.

See [the four-page case study](reports/NEAR_OTC_Case_Study_2026-11-12.pdf) for assumptions and limitations.

## Key research caveats

- The final smile is **not an observed NEAR surface**. HYPE, AVAX and SOL supplied qualifying 30-day peer fits, and the pointwise median selected **AVAX** at every reported strike. ZEC's December smile is only a longer-tenor cross-check.
- **118.9%** ATM IV is a research anchor, conditional on a subjective forecast and an imperfect peer adjustment. It is not an executable option mid.
- Historical 30-day IV-versus-subsequent-RV evaluation has 11 overlapping trade-derived observations; results are not an options-P&L backtest.
- Delta/gamma/vega/theta are Black-76 *fixed-IV* Greeks. The model does not estimate smile dynamics under spot moves.
- Spread and hedge-cost allowances are scenario assumptions; one perp book cannot calibrate full-lifecycle hedging costs.
- Code was developed as a learning/research prototype with AI-assisted development and should be independently reviewed before reuse.

## What is in this repository

- `src/model.py`, `src/desk_calibration.py`: option pricing, surface and two-way cost/risk calculations.
- `src/peer_svi_surface.py`: SVI slices, quote filters and peer smile transfer.
- `src/vol_forecasting.py`, `forecast_check.py`: forecast-method comparison and backtesting.
- `run_peer_svi_smile.py`, `run_peer_svi_diagnostics.py`: dated raw-chain SVI calibration and stricter diagnostics.
- `run.py`, `desk_update.py`, `refresh_market.py`: earlier data/pricing pipelines retained for auditability.
- `tests/`: unit tests for components.
- `data/`: pinned 12 November summary inputs (not the original option-chain download).
- `examples/derived_2026-11-12/`: saved derived results; not raw exchange data.
- `reports/`: submitted research note.

## Reproducing and verifying

Python 3.13 was used during development. From the project root:

```bash
python -m venv .venv
```

Windows:

```cmd
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest -q
```

To reproduce the strict SVI fit you also need the original **12 November 2026, 07:41 UTC** two-sided Derive/Deribit observations saved as `outputs/source_peer_option_nodes.csv` (not included in this reconstructed repository). Then run:

```cmd
.venv\Scripts\python.exe run_peer_svi_smile.py
.venv\Scripts\python.exe run_peer_svi_diagnostics.py
```

The scripts use pinned timestamps and will fail if given improperly dated or insufficient data. The derived audit tables are included in `examples/derived_2026-11-12/` for inspection; they **cannot substitute for the raw quotes for reproducibility**.

The baseline illustrative 12 November report can be attempted from the archived model inputs with:

```cmd
.venv\Scripts\python.exe desk_update.py
```

The PDF packaged in `reports/` is the final presentation note and has been copied from the submission version; this standalone repository has not yet been independently reconciled end to end against the user's latest Windows project. Run the tests and compare regenerated outputs before publication.

## Source and permissions

Options: Derive and Deribit public market data. Underlying price histories: Kraken and Coinbase public endpoints. Perp hedging context: Hyperliquid public market-data snapshot. Confirm redistribution rights before publishing any raw vendor option-chain captures. There are no credentials in this package by design. No license has been selected yet.
