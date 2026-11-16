"""Research-only peer SVI smile calibration for NEAR OTC exercise.

Fit independent observed two-sided option-IV slices from Derive/Deribit.
NEVER fit a smile to previously manufactured NEAR surfaces or model marks.
Requires a dated raw source_peer_option_nodes.csv from a LIVE run.
No fallback to synthetic market observations. Copyright-independent implementation
of the publicly documented SVI formula (Gatheral, 2004).
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm

TGT_T = 30.0 / 365.0
PEERS = ('HYPE', 'SOL', 'AVAX', 'ZEC')
K_MIN, K_MAX = -.45, .45


def svi_w(k, p):
    a,b,rho,m,s = p
    k=np.asarray(k,float)
    return a+b*(rho*(k-m)+np.sqrt((k-m)**2+s*s))


def svi_g(k,p):
    """Gatheral-Durrleman sampled butterfly-condition diagnostic."""
    a,b,rho,m,s=p
    x=np.asarray(k,float)-m
    root=np.sqrt(x*x+s*s)
    w=svi_w(k,p)
    wp=b*(rho+x/root)
    wpp=b*s*s/(root**3)
    return (1-np.asarray(k)*wp/(2*w))**2-(wp**2)/4*(1/w+0.25)+wpp/2


@dataclass
class SliceFit:
    asset: str
    venue: str
    expiry: str
    t: float
    p: np.ndarray
    lo: float
    hi: float
    n: int
    rmse_vol_pts: float
    worst_bidask_spread_pts: float

    def iv(self, k):
        w=svi_w(k,self.p)
        return np.sqrt(w/self.t)


def fit_slice(frame: pd.DataFrame, asset: str, venue: str, expiry: str) -> SliceFit:
    if frame.strike.nunique()<8 or len(frame)<8:
        raise ValueError('fewer than 8 two-sided strike observations')
    x=frame.k.to_numpy(dtype=float)
    if min(x)>-.07 or max(x)<.07:
        raise ValueError('both OTM wings not observed')
    v=frame.iv.to_numpy(dtype=float)
    ts=frame.dte.to_numpy(dtype=float)/365.
    T=float(np.median(ts))
    if (np.max(ts)-np.min(ts))>2/365:
        raise ValueError('maturity not synchronized')
    # Quote IV spreads are in decimal volatility; use as relative fit-error scales.
    quoted_spread=np.maximum(frame.ask_iv.to_numpy(float)-frame.bid_iv.to_numpy(float),.015)
    wobs=v*v*T
    scale=2*np.maximum(v,.07)*T*quoted_spread
    med=float(np.median(wobs))
    lo=np.array([-.5, 1e-5,-.96,-.4,.015])
    hi=np.array([2.5, 5.0, .96, .4,.8])
    initial=[[max(med*.65,.001),max(med*.6,.02),r,0.,.15] for r in (-.6,-.25,0.,.25,.6)]
    initial += [[max(med*.4,.001),max(med*.8,.04),-.25,m,.22] for m in (-.1,.1)]
    xgrid=np.linspace(max(K_MIN,min(x)),min(K_MAX,max(x)),61)
    best=None
    for p0 in initial:
        def residual(p):
            model=svi_w(x,p)
            # Soft-l1 loss plus butterfly penalty in sampled fitted domain.
            g=svi_g(xgrid,p) if np.min(svi_w(xgrid,p))>1e-8 else -np.ones_like(xgrid)
            return np.r_[(model-wobs)/scale, 6*np.minimum(g,0),
                         .06*(p[2]),.03*(p[3])]
        fit=least_squares(residual,np.clip(p0,lo+1e-7,hi-1e-7),
                          bounds=(lo,hi),loss='soft_l1',max_nfev=650)
        if not np.all(np.isfinite(fit.x)):continue
        model=svi_w(xgrid,fit.x)
        if model.min()<=0:continue
        rmse=float(np.sqrt(np.mean((np.sqrt(svi_w(x,fit.x)/T)-v)**2))*100)
        if not np.isfinite(rmse):continue
        # Exclude a non-physical or uninformative fit, rather than reporting it.
        if np.min(svi_g(xgrid,fit.x)) < -.005:continue
        score=rmse+0.002*np.linalg.norm(fit.x)
        if best is None or score<best[0]:best=(score,fit.x,rmse)
    if best is None:raise ValueError('all SVI fits failed sampled butterfly test')
    _,p,rmse=best
    if rmse>7.0:raise ValueError(f'SVI residual too wide ({rmse:.1f} vol points)')
    return SliceFit(asset,venue,expiry,T,p,float(min(x)),float(max(x)),len(frame),rmse,
                    float(np.max(quoted_spread)*100))


def prepare_nodes(raw: pd.DataFrame, snapshot_utc: str, max_age_minutes=20.0):
    need={'asset','strike','forward','expiry','dte','iv','bid_iv','ask_iv','right','quality','asof','source'}
    absent=need-set(raw.columns)
    if absent:raise ValueError(f'Missing raw option-chain fields: {sorted(absent)}')
    df=raw.copy()
    if len(df)==0:raise ValueError('Empty source nodes')
    if df.source.astype(str).str.contains('SYNTHETIC|DEMO|SIMULATED',case=False).any():
        raise ValueError('Synthetic option rows are not acceptable for this study')
    df['stamp']=pd.to_datetime(df['asof'],utc=True,errors='coerce')
    target=pd.Timestamp(snapshot_utc)
    if target.tzinfo is None:raise ValueError('Snapshot UTC timestamp required')
    delta=(df.stamp-target).dt.total_seconds().abs()/60.
    df=df[delta<=max_age_minutes].copy()
    for col in ['strike','forward','dte','iv','bid_iv','ask_iv']:
        df[col]=pd.to_numeric(df[col],errors='coerce')
    df=df[(df.asset.isin(PEERS)) & (df.quality=='two_sided')]
    df=df[(df.bid_iv>.05)&(df.ask_iv>df.bid_iv)&((df.ask_iv-df.bid_iv)<.30)]
    df=df[df.iv.between(.07,3.5)&df.dte.between(10,85)&(df.strike>0)&(df.forward>0)].copy()
    df['k']=np.log(df.strike/df.forward)
    df=df[df.k.between(K_MIN,K_MAX)]
    # Use predominantly OTM contracts to avoid deep-ITM microstructure distortion.
    df=df[((df.right=='P')&(df.k<=.015))|((df.right=='C')&(df.k>=-.015))]
    df['venue']=df.source.astype(str).str.lower().map(lambda s:'deribit' if 'deribit' in s else 'derive' if 'derive' in s else 'unknown')
    df=df[df.venue!='unknown'].copy()
    df['expiry_key']=pd.to_datetime(df.expiry,utc=True,errors='coerce').dt.strftime('%Y-%m-%d')
    df=df.dropna(subset=['expiry_key','k','iv','bid_iv','ask_iv'])
    # ATM put and call at one strike can be combined but must not duplicate weight.
    df=df.sort_values(['asset','venue','expiry_key','strike','right'])
    return df


def fit_peer_slices(df):
    slices=[];audit=[]
    for (asset,venue,expiry),part in df.groupby(['asset','venue','expiry_key']):
        try:
            item=fit_slice(part,asset,venue,expiry)
            slices.append(item)
            status='accepted';reason=''
        except Exception as ex:
            status='rejected';reason=str(ex)[:140]
        audit.append({'asset':asset,'venue':venue,'expiry':expiry,'dte':round(part.dte.median(),1),
            'two_sided_rows':len(part),'distinct_strikes':part.strike.nunique(),
            'put_wing_min_log_k':round(float(part.k.min()),3),
            'call_wing_max_log_k':round(float(part.k.max()),3),
            'status':status,'reason':reason,
            'rmse_vol_pts':round(item.rmse_vol_pts,3) if status=='accepted' else np.nan})
    return slices,pd.DataFrame(audit)


def venue_30d_pair(slices):
    """Only 30d interpolation BETWEEN expiry SVI slices; never silent extrapolation."""
    below=sorted([s for s in slices if s.t<=TGT_T],key=lambda s:s.t)
    above=sorted([s for s in slices if s.t>=TGT_T],key=lambda s:s.t)
    if below and above:
        a=below[-1];b=above[0]
        if (TGT_T-a.t)>22/365 or (b.t-TGT_T)>40/365:return None
        if a is b:return (a,b)
        if (b.t-a.t)<1e-7:return (a,b)
        return (a,b)
    # A slice within 1.5 days of 30d is an explicit near-exact exception.
    near=sorted(slices,key=lambda s:abs(s.t-TGT_T))
    if near and abs(near[0].t-TGT_T)<1.5/365:return (near[0],near[0])
    return None


def venue_iv_at_k(pair,k):
    a,b=pair
    if not (max(a.lo,b.lo)<=k<=min(a.hi,b.hi)):return float('nan')
    w1=float(svi_w(k,a.p));w2=float(svi_w(k,b.p))
    if min(w1,w2)<=0:return float('nan')
    if a is b:
        return float(np.sqrt(w1/a.t))
    if w2+1e-5<w1:return float('nan') # calendar-spread no-arbitrage check
    h=(TGT_T-a.t)/(b.t-a.t)
    return float(np.sqrt(((1-h)*w1+h*w2)/TGT_T))


def extract_venue_smiles(slices, z_grid):
    rows=[];audit=[]
    groups={}
    for s in slices:groups.setdefault((s.asset,s.venue),[]).append(s)
    for (asset,venue),ss in groups.items():
        pair=venue_30d_pair(ss)
        if pair is None:
            audit.append({'asset':asset,'venue':venue,'status':'rejected','reason':'No bracketing 30-day slices'})
            continue
        # For each peer, express skew in ATM-standard-deviation moneyness z.
        atm=venue_iv_at_k(pair,0.)
        if not np.isfinite(atm):
            audit.append({'asset':asset,'venue':venue,'status':'rejected','reason':'ATM unsupported by quotes or calendar check'})
            continue
        audit.append({'asset':asset,'venue':venue,'status':'accepted','reason':'30-day total-variance interpolation',
                      'bracket_dte':f'{pair[0].t*365:.1f} / {pair[1].t*365:.1f}', 'atm_iv_pct':atm*100})
        for z in z_grid:
            k=float(z*atm*np.sqrt(TGT_T))
            iv=venue_iv_at_k(pair,k)
            if np.isfinite(iv):
                rows.append({'asset':asset,'venue':venue,'z':round(float(z),7),
                             'peer_strike_over_forward':np.exp(k),'peer_atm_iv':atm,
                             'iv_ratio':iv/atm,'peer_iv':iv})
    return pd.DataFrame(rows),pd.DataFrame(audit)


def near_smile_from_peers(peer_rows: pd.DataFrame, atm_near_iv: float, near_spot: float, moneyness):
    """Median of asset-level, equal-asset weight IV/ATM ratios at equal z.

    Each asset contributes once even if quoted on two venues.
    Qualifying curve requires >=3 independently observed peer assets AT EVERY strike.
    """
    required={'z','asset','venue','iv_ratio'}
    if peer_rows.empty or not required.issubset(peer_rows.columns):
        raise ValueError('No qualified two-sided peer smile data. Cannot create NEAR curve.')
    rows=[]
    z_target=np.log(np.asarray(moneyness,float))/(atm_near_iv*np.sqrt(TGT_T))
    for M,z in zip(moneyness,z_target):
        sub=peer_rows[np.isclose(peer_rows.z,float(z),atol=1e-5)]
        # first combine venue observations within each underlying
        ratios=sub.groupby('asset').iv_ratio.median()
        if len(ratios)<3:raise ValueError(f'Only {len(ratios)} peer assets at {M:.0%} strike: no supported curve')
        ratio=float(ratios.median())
        iv=float(atm_near_iv*ratio)
        rows.append({'strike_over_forward':M,'strike_usd':M*near_spot,
                     'standardized_moneyness_z':z,'model_iv_pct':iv*100,
                     'independent_peer_assets':len(ratios),
                     'peer_iv_ratio_min':float(ratios.min()),
                     'peer_iv_ratio_max':float(ratios.max()),
                     'iv_sensitivity_min_pct':100*atm_near_iv*float(ratios.min()),
                     'iv_sensitivity_max_pct':100*atm_near_iv*float(ratios.max())})
    out=pd.DataFrame(rows)
    if abs(float(out.iloc[np.argmin(abs(out.strike_over_forward-1))].model_iv_pct)-atm_near_iv*100)>1e-7:
        raise ValueError('ATM calibration not preserved')
    return out


def black76(f,k,t,v,right):
    d1=(math.log(f/k)+.5*v*v*t)/(v*math.sqrt(t));d2=d1-v*math.sqrt(t)
    return f*norm.cdf(d1)-k*norm.cdf(d2) if right=='C' else k*norm.cdf(-d2)-f*norm.cdf(-d1)


def static_checks(surface):
    # This is a DISCRETE screen only (not an arbitrage-free theorem).
    q=surface.sort_values('strike_usd')
    strikes=q.strike_usd.to_numpy(float);iv=q.model_iv_pct.to_numpy(float)/100
    calls=np.array([black76(float(q.iloc[0].strike_usd/q.iloc[0].strike_over_forward),float(k),TGT_T,float(v),'C') for k,v in zip(strikes,iv)])
    slopes=np.diff(calls)/np.diff(strikes)
    flags=[]
    if np.any(np.diff(calls)>1e-5):flags.append('calls not decreasing with strike')
    if np.any(np.diff(slopes)<-0.0008):flags.append('negative discrete butterfly convexity')
    return flags
