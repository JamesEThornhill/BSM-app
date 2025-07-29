import numpy as np
from scipy.optimize import brentq
from black_scholes import BSM


def implied_vol(T, K, S, r, option_price, is_call=True):
    def f(sigma):
        bsm = BSM(T=T, K=K, S=S, r=r, sigma=sigma)
        call, put = bsm.option_prices()
        return (call if is_call else put) - option_price

    try:
        if f(1e-7) * f(5) > 0:
            return np.nan  # No root guaranteed
        return brentq(f, 1e-6, 5)
    except Exception:
        return np.nan
