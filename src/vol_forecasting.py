"""Walk-forward 30-day realised-volatility forecasts, without future-data leakage.

All inputs are completed UTC daily closes. For each prediction on date t, only
training labels whose *full subsequent 30-day window* ends by t may be used.
The final chronological holdout measures the method selected on earlier dates.
Methods require only NumPy/pandas (no additional Python packages).

This is an RV forecast study, NOT a historical option-IV or market-making P&L test.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

METHODS = ('Trailing RV30', 'Fixed blend', 'Optimised blend', 'EWMA', 'HAR ridge')
LAMBDAS = (0.90, 0.94, 0.97, 0.985)
FIXED_WEIGHTS = np.array([0.20, 0.45, 0.35])
# 66 combinations, step 10%, sum to one. Explicit, stable and interpretable.
WEIGHT_GRID = np.array([(i/10, j/10, (10-i-j)/10)
                        for i in range(11) for j in range(11-i)], dtype=float)


def prepare_closes(closes: pd.DataFrame) -> pd.DataFrame:
    required = {'date', 'close'}
    if not required.issubset(closes.columns):
        raise ValueError('Historical forecasting requires date and close columns')
    x = closes[['date', 'close']].copy()
    x['date'] = pd.to_datetime(x['date'], utc=True, errors='raise').dt.normalize()
    x['close'] = pd.to_numeric(x['close'], errors='raise')
    x = x.sort_values('date')
    if x['date'].duplicated().any():
        raise ValueError('Duplicate daily dates in NEAR history')
    if (x['date'].diff().dropna().dt.days != 1).any():
        raise ValueError('Missing daily closes: refusing a misleading backtest')
    if len(x) < 100 or (~np.isfinite(x['close'])).any() or (x['close'] <= 0).any():
        raise ValueError('Need at least 100 positive finite daily closes')
    return x.reset_index(drop=True)


def _ewma_return_vol(rets: np.ndarray, lam: float) -> float:
    # All returns preceding the forecast origin (at most 180). No look-ahead.
    a = np.asarray(rets[-180:], dtype=float)
    w = lam ** np.arange(len(a)-1, -1, -1, dtype=float)
    return float(np.sqrt(365 * np.dot(w, a*a)/w.sum()))


def _features(rets: np.ndarray, end: int) -> dict:
    values = {f'rv{window}': float(np.std(rets[end-window:end], ddof=1)*np.sqrt(365))
              for window in (7, 30, 90)}
    values.update({f'ewma_{lam:g}': _ewma_return_vol(rets[:end], lam)
                   for lam in LAMBDAS})
    return values


def make_history(closes: pd.DataFrame, horizon=30, step_days=7) -> pd.DataFrame:
    x = prepare_closes(closes)
    if horizon < 2 or step_days < 1: raise ValueError('Bad horizon or step')
    log_returns = np.diff(np.log(x.close.to_numpy(dtype=float)))
    rows = []
    for origin in range(90, len(x)-horizon, step_days):
        f = _features(log_returns, origin)
        target = float(np.std(log_returns[origin:origin+horizon], ddof=1)*np.sqrt(365))
        rows.append({'forecast_date_utc': x.date.iloc[origin].date().isoformat(),
                     'origin_index': origin, 'outcome_end_index': origin+horizon,
                     'subsequent_rv30': target, **f})
    return pd.DataFrame(rows)


def _optimised_weights(train: pd.DataFrame) -> np.ndarray:
    # Select only on matured prior outcomes, without evaluating today's target.
    var = train[['rv7', 'rv30', 'rv90']].to_numpy(dtype=float)**2
    estimate = np.sqrt(np.maximum(0, var @ WEIGHT_GRID.T))
    scores = np.abs(estimate - train.subsequent_rv30.to_numpy()[:, None]).mean(axis=0)
    # Prefer a blend near the original fixed weights when candidates tie.
    penalty = 0.001*np.abs(WEIGHT_GRID-FIXED_WEIGHTS).sum(axis=1)
    return WEIGHT_GRID[int(np.argmin(scores + penalty))]


def _blend(row, w: np.ndarray) -> float:
    return float(np.sqrt(max(0, np.dot(w, np.array([row['rv7'], row['rv30'], row['rv90']])**2))))


def _optimised_lambda(train: pd.DataFrame) -> float:
    target = train.subsequent_rv30.to_numpy(dtype=float)
    return min(LAMBDAS, key=lambda lam: (float(np.mean(np.abs(train[f'ewma_{lam:g}'].to_numpy()-target))), lam))


def _har_ridge(train: pd.DataFrame, row, ridge=5.0) -> float:
    # HAR-style regression of *log forward variance* on lagged log variances.
    # Standardization + ridge makes the tiny-sample regression less volatile.
    eps = 1e-8
    x = np.log(np.maximum(train[['rv7', 'rv30', 'rv90']].to_numpy(dtype=float)**2, eps))
    y = np.log(np.maximum(train.subsequent_rv30.to_numpy(dtype=float)**2, eps))
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.maximum(std, 0.1)
    z = (x-mean)/std
    beta = np.linalg.solve(z.T@z + ridge*np.eye(3), z.T@(y-y.mean()))
    live_z = (np.log(np.maximum(np.array([row['rv7'], row['rv30'], row['rv90']])**2, eps))-mean)/std
    forecast = float(np.exp(0.5*(y.mean()+live_z@beta)))
    # Guardrail for rare ill-conditioned small samples; disclosed as a model constraint.
    return float(np.clip(forecast, 0.15, 4.0))


def _predict_all(train: pd.DataFrame, row) -> tuple[dict, dict]:
    weights = _optimised_weights(train)
    lam = _optimised_lambda(train)
    preds = {
        'Trailing RV30': float(row['rv30']),
        'Fixed blend': _blend(row, FIXED_WEIGHTS),
        'Optimised blend': _blend(row, weights),
        'EWMA': float(row[f'ewma_{lam:g}']),
        'HAR ridge': _har_ridge(train, row),
    }
    return preds, {'weights': weights.tolist(), 'ewma_lambda': lam}


def compare_forecasts(closes: pd.DataFrame, horizon=30, step_days=7,
                      min_train=20, validation_fraction=.60) -> tuple[pd.DataFrame,pd.DataFrame,dict]:
    """Walk-forward predictions and fixed chronological model selection.

    Earlier OOS origins pick the winner; later OOS origins are a holdout. At
    each date, training may include *only matured* target windows. Outcomes
    overlap in calendar time: scores are not independent confidence intervals.
    """
    x = prepare_closes(closes)
    hist = make_history(x, horizon, step_days)
    if hist.empty: raise ValueError('Insufficient daily history for backtest')
    preds_rows=[]
    for row in hist.to_dict(orient="records"):
        # Data with outcome ending today is known; later outcomes forbidden.
        prior = hist.loc[hist.outcome_end_index<=row["origin_index"]]
        if len(prior)<min_train:continue
        preds,details = _predict_all(prior, row)
        for method, predicted in preds.items():
            preds_rows.append({'forecast_date_utc': row["forecast_date_utc"],
                               'method': method, 'forecast_rv30': predicted,
                               'subsequent_rv30': row["subsequent_rv30"],
                               'error_vol_points': 100*(predicted-row["subsequent_rv30"]),
                               'absolute_error_vol_points': 100*abs(predicted-row["subsequent_rv30"]),
                               'matured_training_samples':len(prior),
                               'ewma_lambda': details['ewma_lambda'] if method=='EWMA' else np.nan,
                               'optimised_weights': '/'.join(f'{v:.1f}' for v in details['weights']) if method=='Optimised blend' else ''})
    predictions = pd.DataFrame(preds_rows)
    dates = sorted(predictions.forecast_date_utc.unique()) if not predictions.empty else []
    if len(dates)<15:
        raise ValueError(f'Only {len(dates)} walk-forward test dates; need at least 15. More historical closes are required.')
    cut=max(8, min(len(dates)-6, int(math.floor(len(dates)*validation_fraction))))
    split_date = dates[cut]
    predictions['segment']=np.where(predictions.forecast_date_utc<split_date,'selection','holdout')
    score_rows=[]
    for segment in ('selection','holdout','all'):
        sub = predictions if segment=='all' else predictions.loc[predictions.segment==segment]
        for method in METHODS:
            q=sub.loc[sub.method==method]
            error=q.error_vol_points.to_numpy(dtype=float)
            score_rows.append({'segment':segment, 'method':method, 'n':len(error),
                               'mae_vol_points':float(np.abs(error).mean()),
                               'rmse_vol_points':float(np.sqrt(np.mean(error**2))),
                               'bias_vol_points':float(error.mean())})
    scores = pd.DataFrame(score_rows)
    selection = scores[scores.segment=='selection'].sort_values(['mae_vol_points','method'])
    winner = str(selection.iloc[0]['method'])
    latest_returns = np.diff(np.log(x.close.to_numpy(dtype=float)))
    latest = _features(latest_returns, len(latest_returns))
    # Every historical target is matured by the most recent completed close.
    final_preds, final_params = _predict_all(hist, latest)
    hold = scores[scores.segment=='holdout'].set_index('method')
    fix_error=float(hold.loc['Fixed blend','mae_vol_points'])
    base_error=float(hold.loc['Trailing RV30','mae_vol_points'])
    winner_error=float(hold.loc[winner,'mae_vol_points'])
    # Predeclared deployment gate: later holdout serves as an acceptance filter,
    # not an untouched final performance confirmation after deployment.
    holdout_gate=(winner=='Fixed blend' or (
        fix_error-winner_error >= 1.0 and winner_error <= 0.95*fix_error))
    deployed=winner if holdout_gate else 'Fixed blend'
    summary={
        'selected_method':winner,
        'selected_by':'lowest MAE on earlier walk-forward dates',
        'deployed_method':deployed,
        'deployment_gate_passed':bool(holdout_gate),
        'deployment_policy':'Early winner must beat old fixed blend on later holdout by >=1 vol point and >=5% MAE, otherwise deploy fixed blend. Holdout is an acceptance filter, not untouched confirmation.',
        'forecast_rv':float(final_preds[deployed]),
        'live_model_forecasts':{k:float(v) for k,v in final_preds.items()},
        'live_optimised_blend_weights':final_params['weights'],
        'live_ewma_lambda':final_params['ewma_lambda'],
        'selection_dates':int(cut), 'holdout_dates':int(len(dates)-cut),
        'selection_end_utc':dates[cut-1], 'holdout_start_utc':dates[cut],
        'holdout_selected_mae_vol_points':winner_error,
        'holdout_fixed_mae_vol_points':fix_error,
        'holdout_trailing30_mae_vol_points':base_error,
        'holdout_improvement_vs_fixed_vol_points':fix_error-winner_error,
        'holdout_improvement_vs_trailing_vol_points':base_error-winner_error,
        'horizon_days':horizon, 'step_days':step_days,
        'min_matured_training_samples':min_train,
        'note':'Weekly overlapping 30-day outcomes; compact sample, regime-dependent; selection vs later holdout is descriptive, not a confidence interval. This forecasts REALISED volatility, not options IV.'
    }
    return predictions, scores, summary


def save_holdout_chart(scores: pd.DataFrame, path, selected_method: str) -> None:
    """Optional compact MAE comparison chart; no forecasting logic depends on it."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    view = scores.loc[scores.segment.eq('holdout')].copy()
    if view.empty:
        raise ValueError('No holdout scores to plot')
    fig, ax = plt.subplots(figsize=(8.3, 3.6))
    bars = ax.barh(view.method, view.mae_vol_points, color=[
        '#2b5f74' if a == selected_method else '#97aab5' for a in view.method
    ])
    ax.invert_yaxis()
    for b, val in zip(bars, view.mae_vol_points):
        ax.text(b.get_width()+0.15, b.get_y()+b.get_height()/2,
                f'{val:.1f}', ha='left', va='center', fontsize=9)
    ax.set(xlabel='MAE, annualised volatility points (lower is better)',
           title='NEAR 30-day RV forecast | later walk-forward holdout')
    ax.set_xlim(0, max(view.mae_vol_points)*1.18)
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)
    fig.text(.02,.015,'Highlight = deployed after holdout gate. Overlapping 30-day windows; not IV or trading P&L.',
             fontsize=8, color='#6a7880')
    fig.tight_layout(rect=[0,.065,1,1])
    fig.savefig(path,dpi=165)
    plt.close(fig)
