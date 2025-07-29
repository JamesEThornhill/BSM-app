import numpy as np

#This is for the payoff of a single option
def compute_option_payoff(strike_price, option_price, option_type, position_type, stock_prices):
    
    if position_type == 'Long':
        if option_type == 'Call':
            payoff = np.maximum(stock_prices - strike_price - option_price, -option_price)
        else:
            payoff = np.maximum(strike_price - option_price - stock_prices, -option_price)
    else:  # Short
        if option_type == 'Call':
            payoff = np.minimum(option_price + strike_price - stock_prices, option_price)
        else:
            payoff = np.minimum(stock_prices - strike_price + option_price, option_price)

    return payoff


