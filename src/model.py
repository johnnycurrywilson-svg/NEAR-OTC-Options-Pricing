"""Transparent Black-76, peer smile calibration, and NEAR transfer model.

All vols are decimals (1.10 = 110%), times are ACT/365, prices in USD per token.
This is a research prototype, NOT an execution or hedging system.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import norm

DAYS = 365.0
PEER_BASE_WEIGHTS = {'HYPE': 0.30, 'SOL': 0.25, 'ZEC': 0.20, 'AVAX': 0.15, 'TRX': 0.10}


def black76(forward: float, strike: float, t: float, iv: float, right='C', discount_rate=0.0):
    """European option PV using observable/assumed forward; USD per 1 token."""
    if forward <= 0 or strike <= 0 or iv <= 0 or t <= 0:
        raise ValueError('Forward, strike, IV and time must be positive')
    rt = iv * math.sqrt(t)
    d1 = (math.log(forward / strike) + 0.5 * iv * iv * t) / rt
    d2 = d1 - rt
    df = math.exp(-discount_rate * t)
    if right.upper() == 'C':
        price = df * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
        fwd_delta = df * norm.cdf(d1)
    elif right.upper() == 'P':
        price = df * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))
        fwd_delta = -df * norm.cdf(-d1)
    else:
        raise ValueError('right must be C or P')
    fwd_gamma = df * norm.pdf(d1) / (forward * iv * math.sqrt(t))
    vega_per_vol_point = df * forward * norm.pdf(d1) * math.sqrt(t) * 0.01
    return {'price': float(price), 'forward_delta': float(fwd_delta),
            'forward_gamma': float(fwd_gamma), 'vega_per_1volpt': float(vega_per_vol_point)}


def _quality_weight(status: str):
    return {'two_sided': 1.0, 'synthetic_test': 1.0, 'mark_only': 0.16,
            'one_sided': 0.28}.get(str(status), 0.0)


def _tenor_iv(days: np.ndarray, iv: np.ndarray, target_days: float):
    """Interpolate ATM total variance at target tenor; flat IV extrapolation."""
    idx = np.argsort(days)
    ds, vs = np.asarray(days)[idx], np.asarray(iv)[idx]
    if target_days <= ds[0]: return float(vs[0])
    if target_days >= ds[-1]: return float(vs[-1])
    total_variance = vs**2 * ds / DAYS
    w = np.interp(target_days, ds, total_variance)
    return float(math.sqrt(max(w / (target_days / DAYS), 1e-9)))


def fit_peer_smiles(nodes: pd.DataFrame):
    """Fit IV(z)/ATM = 1 + slope*z + curvature*z^2 on OTM mark points.

    z = log(K/F) / (ATM * sqrt(T)), a scale-normalized moneyness axis.
    At each expiry, ATM is estimated from nearest-to-forward option marks.
    Real quotes are weighted more highly than exchange model marks.
    """
    fits=[]
    if 'venue' not in nodes.columns:
        raise ValueError('A venue column is required; never conflate Derive and Deribit quotes')
    for (asset, venue, dte), g in nodes.groupby(['asset', 'venue', 'dte']):
        g = g.copy()
        if len(g) < 5 or g['strike'].nunique()<4: continue
        g['dist'] = abs(np.log(g.strike / g.forward))
        nearest = g.nsmallest(min(4, len(g)), 'dist')
        atm = float(np.average(nearest.iv, weights=np.maximum(nearest.quality_weight,0.01)))
        if not (0.10 < atm < 4.0): continue
        z=np.log(g.strike.to_numpy()/g.forward.to_numpy())/(atm*math.sqrt(dte/DAYS))
        mask=(abs(z)<2.3)&(g.quality_weight.to_numpy()>0)
        if mask.sum()<5 or not (any(z[mask]<-0.3) and any(z[mask]>0.3)): continue
        zz=z[mask]
        yy=g.iv.to_numpy()[mask]/atm-1.0
        X=np.column_stack([zz,zz*zz])
        weights=g.quality_weight.to_numpy()[mask]
        ridge=np.diag([0.035,0.035])
        coeff=np.linalg.solve(X.T @ (weights[:,None]*X)+ridge, X.T@(weights*yy))
        slope=float(np.clip(coeff[0],-0.30,0.30))
        curvature=float(np.clip(coeff[1],0.0,0.35))
        two_sided=int((g.quality=='two_sided').sum())
        quality_ratio=float(g.quality_weight.mean())
        fits.append({'asset':asset,'venue':venue,'dte':float(dte),'atm':atm,'slope':slope,
                     'curvature':curvature,'n_nodes':int(mask.sum()),'two_sided':two_sided,
                     'quality_score':quality_ratio})
    if not fits: raise RuntimeError('Insufficient peer smile data to fit both wings. No NEAR surface generated.')
    return pd.DataFrame(fits).sort_values(['asset','venue','dte']).reset_index(drop=True)


def peer_at(fits: pd.DataFrame, asset: str, days: float):
    """Each asset gets ONE vote; fuse its venues after fitting separately.

    This prevents HYPE and SOL from automatically receiving twice the weight
    merely because they trade on two exchanges.
    """
    sub=fits[fits.asset==asset]
    if sub.empty:raise KeyError(asset)
    venue_values=[]
    for venue, g in sub.groupby('venue'):
        g=g.sort_values('dte')
        ds=g.dte.to_numpy(dtype=float)
        atm=_tenor_iv(ds,g.atm.to_numpy(dtype=float),days)
        slope=float(np.interp(days,ds,g.slope))
        curvature=float(np.interp(days,ds,g.curvature))
        reliability=float(np.clip(g.quality_score.mean(),.05,1.0))
        venue_values.append({'venue':venue,'atm':atm,'slope':slope,
                             'curvature':curvature,'reliability':reliability})
    vals=np.asarray([v['reliability'] for v in venue_values],dtype=float)
    probs=vals/vals.sum()
    return {'atm':float(sum(w*v['atm'] for w,v in zip(probs,venue_values))),
            'slope':float(sum(w*v['slope'] for w,v in zip(probs,venue_values))),
            'curvature':float(sum(w*v['curvature'] for w,v in zip(probs,venue_values))),
            'reliability':float(np.mean(vals)),
            'venue_count':len(venue_values)}

@dataclass
class NearSurface:
    fits: pd.DataFrame
    rv30: float
    vrp_ratio: float = 0.08

    def __post_init__(self):
        if not 0.10<self.rv30<4.0: raise ValueError('NEAR historical/forecast RV input must be decimal (1.10 means 110%).')
        if not -0.5<self.vrp_ratio<1.0: raise ValueError('VRP ratio outside permitted research range')
        self.peers=[a for a in PEER_BASE_WEIGHTS if a in set(self.fits.asset)]
        if len(self.peers)<2: raise ValueError('Need at least 2 peer markets for cross-asset surface.')
        scores={a:PEER_BASE_WEIGHTS[a]*max(peer_at(self.fits,a,30)['reliability'],0.10) for a in self.peers}
        total=sum(scores.values())
        self.weights={a:round(v/total,6) for a,v in scores.items()}
        # Normalize once more after rounding
        total=sum(self.weights.values());self.weights={a:v/total for a,v in self.weights.items()}
        self.atm30=self.rv30*(1+self.vrp_ratio)

    def at(self, days:float):
        if days<=0: raise ValueError('Tenor days must be positive')
        at30=sum(self.weights[a]*peer_at(self.fits,a,30)['atm'] for a in self.peers)
        atT=sum(self.weights[a]*peer_at(self.fits,a,days)['atm'] for a in self.peers)
        # Peer-relative term structure transferred, NEAR-specific RV anchors the level.
        atm=float(self.atm30*(atT/at30))
        a=sum(self.weights[x]*peer_at(self.fits,x,days)['slope'] for x in self.peers)
        b=sum(self.weights[x]*peer_at(self.fits,x,days)['curvature'] for x in self.peers)
        return {'atm':atm,'slope':a,'curvature':b}

    def iv(self, forward:float, strike:float, days:float):
        p=self.at(days)
        z=math.log(strike/forward)/(p['atm']*math.sqrt(days/DAYS))
        # No wild-wing extrapolation; flag in reporting beyond |z|>2
        zc=float(2.0*np.tanh(z/2.0))
        return max(.10, p['atm']*(1+p['slope']*zc+p['curvature']*zc*zc))

    def quote(self, spot:float, strike:float, days:float, right:str='C',
              carry:float=0.0, rate:float=0.0, spread_vol_pts:float=6.0):
        t=days/DAYS
        forward=spot*math.exp(carry*t)
        iv=self.iv(forward,strike,days)
        mid=black76(forward,strike,t,iv,right,rate)
        half=spread_vol_pts/200  # 6 vol points full spread -> +/- 0.03 annual IV
        bid_iv=max(.05,iv-half)
        ask_iv=iv+half
        bid=black76(forward,strike,t,bid_iv,right,rate)['price']
        ask=black76(forward,strike,t,ask_iv,right,rate)['price']
        return {'spot':spot,'forward':forward,'strike':strike,'days':days,'right':right,
                'iv':iv,'bid_iv':bid_iv,'ask_iv':ask_iv,'bid':bid,'mid':mid['price'],
                'ask':ask,'fwd_delta':mid['forward_delta'],
                'spot_delta':mid['forward_delta']*forward/spot,
                'gamma_spot':mid['forward_gamma']*(forward/spot)**2,
                'vega_per_1volpt':mid['vega_per_1volpt'],
                'outside_fit_range':abs(math.log(strike/forward)/(self.at(days)['atm']*math.sqrt(t)))>2.0}


def hedged_stress(surface:NearSurface, quote:dict, tokens:int, moves=(-.30,-.15,0,.15,.30),
                  vol_changes=(-.20,0,.20), day_step:int=7):
    """Mark-to-market of short option and its initial delta hedge after 7 days.

    Hold initial underlying hedge fixed; do not simulate dynamic gamma scalping.
    No transaction fees, funding, slippage, or counterparty risk included.
    """
    rows=[]; q=quote; days=max(0.1,q['days']-day_step)
    for move in moves:
        new_spot=q['spot']*(1+move)
        new_forward=q['forward']*(1+move)
        for dv in vol_changes:
            new_iv=max(.05,surface.iv(new_forward,q['strike'],days)+dv)
            next_price=black76(new_forward,q['strike'],days/DAYS,new_iv,q['right'])['price']
            pnls=(q['ask']-next_price)*tokens+q['spot_delta']*tokens*(new_spot-q['spot'])
            rows.append({'underlying_move_pct':move*100,'vol_shift_pp':dv*100,
                         'option_mark':next_price,'pnl_before_costs':pnls})
    return pd.DataFrame(rows)


def simple_arbitrage_checks(surface:NearSurface, spot:float, tenors=(7,30,90),
                            multipliers=tuple(np.linspace(.60,1.5,75))):
    """Discrete screens: vertical monotonicity, butterfly convexity, and calendar at constant F.

    These checks are necessary screens, not a guarantee of absence of static arbitrage.
    """
    failures=[]
    for d in tenors:
        strikes=np.asarray(multipliers)*spot
        vals=np.array([black76(spot,float(k),d/DAYS,surface.iv(spot,float(k),d))['price'] for k in strikes])
        if np.any(np.diff(vals)>1e-5): failures.append(f'call price not decreasing in K at {d}d')
        if np.any(np.diff(vals,n=2)<-1e-4): failures.append(f'call price not convex in K at {d}d')
    for k in np.asarray(multipliers)[::7]*spot:
        prices=[black76(spot,float(k),d/DAYS,surface.iv(spot,float(k),d))['price'] for d in tenors]
        if np.any(np.diff(prices)<-1e-5): failures.append(f'calendar at K={k:.2f}')
    return sorted(set(failures))
