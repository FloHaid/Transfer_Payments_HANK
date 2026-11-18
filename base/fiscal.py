import numpy as np

def Bplan(dT, rho_B):
    """Determine financing for any desired time path of taxes (spending) dT_t (dG_t),
    solving dB_t = rho_B (dB_{t-1} + dT_t (dG_t))"""
    T = len(dT)
    dB = np.empty(T)
    for t in range(T):
        dB_lag = dB[t-1] if t>0 else 0
        dB[t] = rho_B * (dB_lag + dT[t])   #changed the sign here perviously to reflect the taxes reducing debt. dT now confusingly refers to a spending or rather transfer shock
    return dB

#The only thing I might need to do is change the signs here. dG/dT can be anything, its just the name for unspecified shocks
def Tplan(dG, dB, r):
    """Get tax plan dT_t coresponding to any path dG_t, dB_t"""
    dB_lag = np.concatenate(([0], dB[:-1]))
    dT = dG + (1+r)*dB_lag - dB
    return dT







def compute_multipliers(dY, dT, r):
    """Compute multipliers given impulses dG_t, dY_t"""
    q = (1 + r) ** (-np.arange(len(dY)))

    mult_impact = dY[0] / dT[0]
    mult_5Y     = (q[:5] @ dY[:5]) / (q[:5] @ dT[:5])
    mult_cumul  = (q @ dY) / (q @ dT)

    return mult_impact, mult_5Y, mult_cumul
