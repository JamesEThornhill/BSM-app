import numpy as np
from black_scholes import BSM

def option_matrix(T, r, sigma_range, price_range, S_fixed, K_fixed, is_call=True, x_axis="Spot Price"):
    option_prices = np.zeros((len(sigma_range), len(price_range)))

    for i, sigma in enumerate(sigma_range):
        for j, price in enumerate(price_range):
            S_var = price if x_axis == "Spot Price" else S_fixed
            K_var = price if x_axis == "Strike Price" else K_fixed

            bsm_output = BSM(T=T, K=K_var, S=S_var, r=r, sigma=sigma)
            call_price, put_price = bsm_output.option_prices()

            option_prices[i, j] = call_price if is_call else put_price

    return option_prices


def option_visualiser(T_range, r, sigma, price_range, S_fixed, K_fixed, is_call=True, x_axis="Spot Price"):
    option_prices = np.zeros((len(T_range), len(price_range)))

    for i, T in enumerate(T_range):
        for j, price in enumerate(price_range):
            S_var = price if x_axis == "Spot Price" else S_fixed
            K_var = price if x_axis == "Strike Price" else K_fixed

            bsm_output = BSM(T = T, K=K_var, S=S_var, r=r, sigma=sigma)
            call_price, put_price = bsm_output.option_prices()

            option_prices[i, j] = call_price if is_call else put_price

    return option_prices

