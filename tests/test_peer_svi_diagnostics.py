import numpy as np
import pandas as pd
from scipy.stats import norm
from run_peer_svi_diagnostics import delta_with_atm, delta_points

def test_delta_25_conventions():
    assert abs(delta_with_atm(0,.6,.6)-norm.cdf(.5*.6*np.sqrt(30/365)))<1e-10

def test_constant_iv_finds_delta_wings():
    z=np.linspace(-1.2,1.2,241)
    r=pd.DataFrame({'z':z,'iv_ratio':np.ones(len(z))})
    q=delta_points(r,.65)
    assert abs(q['rr25_pts'])<1e-7
    assert abs(q['bf25_pts'])<1e-7
    assert np.isfinite(q['put25_iv_pct'])

def test_no_extrapolation_of_25delta():
    z=np.linspace(-.2,.2,30)
    r=pd.DataFrame({'z':z,'iv_ratio':np.ones(len(z))})
    q=delta_points(r,.65)
    assert np.isnan(q['rr25_pts'])
