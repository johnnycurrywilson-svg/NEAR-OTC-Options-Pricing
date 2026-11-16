"""Unauthenticated Kraken NEAR/USD snapshots and completed UTC daily prices.

Only daily candles closed before the current UTC day are included in RV.
No synthetic fallback. Kraken can change availability and public API schema.
"""
from __future__ import annotations
from datetime import datetime, timezone
import math
import numpy as np
import pandas as pd
import requests

BASE = "https://api.kraken.com/0/public"


def _request(session, path, params):
    r = session.get(f"{BASE}/{path}", params=params, timeout=20)
    r.raise_for_status()
    body = r.json()
    if body.get("error"):
        raise RuntimeError(f"Kraken {path} error: {body['error']}")
    return body['result']


def fetch_near_daily_prices():
    """Return (daily completed closes, spot, metadata); all values from Kraken."""
    asof = datetime.now(timezone.utc)
    with requests.Session() as session:
        daily = _request(session, "OHLC", {"pair": "NEARUSD", "interval": 1440})
        pairs = [name for name in daily if name != "last"]
        if len(pairs) != 1:
            raise RuntimeError(f"Unexpected Kraken NEAR OHLC payload keys: {pairs}")
        entries = daily[pairs[0]]
        rows = []
        for item in entries:
            ts = pd.Timestamp(int(item[0]), unit="s", tz="UTC")
            close = float(item[4])
            # Kraken includes a final candle that has not yet closed.
            if ts.date() >= asof.date():
                continue
            if math.isfinite(close) and close > 0:
                rows.append({"date": ts.date().isoformat(), "close": close})
        prices = pd.DataFrame(rows).drop_duplicates("date").sort_values("date")
        if len(prices) < 91:
            raise RuntimeError(f"Only {len(prices)} completed NEAR daily closes; need >=91 for 90d RV")
        prices["date"] = pd.to_datetime(prices["date"], utc=True)
        intervals = prices["date"].diff().dropna().dt.days
        if not (intervals == 1).all():
            raise RuntimeError("Missing NEAR daily dates in Kraken data; refusing inaccurate RV estimate")
        ticker = _request(session, "Ticker", {"pair": "NEARUSD"})
        markets = list(ticker.values())
        if len(markets) != 1:
            raise RuntimeError("Unexpected Kraken NEAR spot ticker payload")
        spot = float(markets[0]["c"][0])
        if not math.isfinite(spot) or spot <= 0:
            raise RuntimeError("Kraken returned invalid NEAR/USD spot price")
        metadata = {"source": "Kraken public REST NEAR/USD", "spot_asof_utc": asof.isoformat(),
                    "last_complete_daily_candle_utc": prices.iloc[-1]["date"].date().isoformat(),
                    "candles_count": int(len(prices))}
        return prices, spot, metadata


def historical_volatility(prices, windows=(7, 14, 30, 60, 90)):
    logp = np.log(prices["close"].astype(float).to_numpy())
    rets = np.diff(logp)
    entries = []
    for n in windows:
        if len(rets) < n:
            raise ValueError(f"Not enough observations for {n}-day volatility")
        sigma = float(np.std(rets[-n:], ddof=1) * np.sqrt(365))
        entries.append({"lookback_days": int(n), "realized_vol_decimal": sigma,
                        "realized_vol_percent": sigma * 100.0})
    return pd.DataFrame(entries)
