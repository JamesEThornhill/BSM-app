import numpy as np
from scipy.stats import norm

#Code to calculate call or put price using Black-Scholes model
class BSM:
    def __init__(self, T: float, K: float, S: float, r: float, sigma: float):
        self.T = T
        self.K = K
        self.S = S
        self.r = r
        self.sigma = sigma

    def option_prices(self):
        d_1 = (np.log(self.S / self.K) + (self.r + self.sigma**2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d_2 = d_1 - self.sigma * np.sqrt(self.T)

        call_price = norm.cdf(d_1) * self.S - norm.cdf(d_2) * self.K * np.exp(-self.r * self.T)
        put_price = norm.cdf(-d_2) * self.K * np.exp(-self.r * self.T) - norm.cdf(-d_1) * self.S

        self.call_price = call_price
        self.put_price = put_price

        return call_price, put_price