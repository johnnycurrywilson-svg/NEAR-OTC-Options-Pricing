"""Quick offline comparison using already-downloaded NEAR daily prices.

Run from project folder: .venv\\Scripts\\python.exe forecast_check.py
No exchange connection, no PDF changes, no trade execution.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from src.vol_forecasting import compare_forecasts, save_holdout_chart

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
DATA=OUT/'near_daily_closes.csv'
if not DATA.exists():
    raise SystemExit('Missing outputs/near_daily_closes.csv. Run refresh_market.py once to download completed NEAR daily candles.')
prices=pd.read_csv(DATA)
predictions, scores, summary=compare_forecasts(prices)
predictions.to_csv(OUT/'near_forecast_comparison.csv',index=False)
scores.to_csv(OUT/'near_forecast_model_scores.csv',index=False)
save_holdout_chart(scores,OUT/'05_near_forecast_holdout.png',summary['deployed_method'])
(OUT/'near_forecast_selected.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('\nHOLDOUT (later unseen dates, lower MAE is better):')
print(scores.loc[scores.segment=='holdout',['method','n','mae_vol_points','bias_vol_points']].to_string(index=False))
print('\nSelected on EARLY dates:',summary['selected_method'])
print('Deployed after later holdout acceptance check:',summary['deployed_method'])
print('Live selected forward RV:',f"{100*summary['forecast_rv']:.1f}%")
print('Holdout selected vs existing fixed blend:',f"{summary['holdout_improvement_vs_fixed_vol_points']:+.2f} volatility points improvement")
print('Saved outputs/near_forecast_model_scores.csv, near_forecast_comparison.csv and near_forecast_selected.json')
