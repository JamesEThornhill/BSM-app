import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap

#import BSM class from external module (returns a call and put price)
from black_scholes import BSM
#import option payoff calculator
from option_payoff import compute_option_payoff
#import option price matrix and option prices for 3d plot
from option_matrix import option_matrix, option_visualiser
#import implied volatility function
from implied_volatility import implied_vol

st.set_page_config(layout="centered")
#--------------------------------------------------------------------------------------------
#create sidebar that allows for different values of variables
##set out default variable values
default = {
    "S": 100.0,
    "K": 90.0,
    "r_pc": 4.0,
    "T": 1.0,
    "sigma_pc": 25.0
}

with st.sidebar:
    st.title("Contents:")
    st.markdown("""
    - [Options Pricing Calculator](#options-pricing-calculator)
    - [Option Payoff Visualizer](#option-payoff-visualizer)
    - [Heatmap of Option Prices](#heatmap-of-option-prices)
    - [BSM Visualization](#bsm-visualization)
    - [Implied Volatility Calculator](#implied-volatility-calculator)
    - [Price Mismatch Heatmap](#price-mismatch-heatmap)
    """, unsafe_allow_html=True)

    st.markdown("-------")
    st.title("Input Variables:")
    ##Reset button
    if st.button("Reset Values"):
        for key, val in default.items():
            st.session_state[key] = val

    S = st.number_input("Spot Price, S", step=0.1, key="S", value=st.session_state.get("S", default["S"]))
    K = st.number_input("Strike Price, K", step=0.1, key="K", value=st.session_state.get("K", default["K"]))
    r_pc = st.number_input("Risk Free Interest Rate, r (%)", step=0.01, key="r_pc", value=st.session_state.get("r_pc", default["r_pc"]))
    T = st.number_input("Time to Maturity, T (years)", step=0.01, key="T", value=st.session_state.get("T", default["T"]))
    sigma_pc = st.number_input("Volatility, σ (%)", step=0.1, key="sigma_pc", value=st.session_state.get("sigma_pc", default["sigma_pc"]))


#Have displayed in sidebar as percentage but then convert to decimal for calc
sigma = sigma_pc/100
r = r_pc/100

#Call and option price
BSM_output = BSM(T,K,S,r,sigma)
C, P = BSM_output.option_prices()

#----------------------------------------------------------------------------------------
#Intro

st.title("Introduction: Black-Scholes Model")
st.info('''This app provides an interactive way to explore the Black-Scholes model - a tool used for pricing stock options. 
        
        Options are financial contracts that give buyers the right (or option) to buy (call option) or sell (put option) an asset at a fixed price before a set date.
        The Black-Scholes formula estimates the fair value of (European style) call and put options based on variables like the underlying asset price, strike price, 
        time to expiry, interest rates and volatility. The model can help traders and investors make better decisions by allowing them to price and hedge options in a 
        consistent way.
        
        Below, this app lets you calculate options prices and visualise how they change with different inputs, helping grasp the model's assumptions, behaviour 
        and value in options trading.
        ''')

#----------------------------------------------------------------------------------------
#Layout of page

st.title("Black-Scholes Model: pricing and visualizer tool")

st.markdown("---")
st.markdown("<h2 id='pricing-calculator'>Options pricing calculator</h2>", unsafe_allow_html=True)
st.info('''
Calculating the fair value for option contracts using Black-Scholes model. Use the sidebar on the left to adjust the variable values.

Model assumptions:
- The option is European - meaning it can only be exercised at expiration
- The returns of the underlying asset are normally distributed
- The risk-free interest rate and the volatility of the underlying asset are known and remain constant
- Markets are random (market movements cannot be predicted)
- No dividends are paid out during life of option
- No transaction costs in buying the option
''')

variable_values = {
    "Variable": ["Spot Price, S", "Strike Price, K", "Time to Maturity, T (years)", "Volatility, σ (%)", "Risk-Free Interest Rate, r (%)"],
    "Value": [round(S,1), round(K,1), round(T,2), f"{round(sigma_pc,1)}%", f"{round(r_pc,2)}%"]
}
variable_data = pd.DataFrame(variable_values)

html = variable_data.to_html(index=False)

styled_html = f"""
<style>
    table {{
        border-collapse: collapse;
        width: 100%;
    }}
    th {{
        font-weight: bold !important;
        color: black !important;
        background-color: #f8f8f8;
        border-bottom: 2px solid #ddd;
        padding: 8px;
        text-align: left !important;
    }}
    td {{
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }}
</style>
{html}
"""

table_col, calc_prices_col = st.columns([2, 1])

with table_col:
    st.markdown(styled_html, unsafe_allow_html=True)

with calc_prices_col:
    st.markdown(f"""
        <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; max-width: 200px; margin-bottom: 10px; margin-left: 40px;">
            <div class="metric-label" style="font-weight: bold; font-size: 18px; color: #333;">Call Option</div>
            <div class="metric-sub-label" style="font-size: 10px; color: #666;">(Calculated price)</div>
            <div class="metric-value" style="font-size: 24px; font-weight: bold; color: #007acc;">{C:.2f}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; max-width: 200px; margin-left: 40px;">
            <div class="metric-label" style="font-weight: bold; font-size: 18px; color: #333;">Put Option</div>
            <div class="metric-sub-label" style="font-size: 10px; color: #666;">(Calculated price)</div>
            <div class="metric-value" style="font-size: 24px; font-weight: bold; color: #007acc;">{P:.2f}</div>
        </div>
    """, unsafe_allow_html=True)

st.info('''
Increasing the risk-free interest rate, r, makes the strike payment cheaper at present value. This helps calls (cheaper to delay paying strike price), but hurts puts (future inflows less valuable today).
        
Increasing the volatility, σ, increases both the prices of a call and put option, since the potential size of the payoff (chance stock price differs significantly from strike price) becomes larger. 

Increasing the time to maturity, T, increases the prices of both call and put options, as it provides more opportunity for the underlying asset price to move favorably, increasing the potential payoff.

        ''')
#----------------------------------------------------------------------------------------------------


# Your existing Streamlit inputs and calculations
st.markdown("---")
st.markdown("<h2 id='option-payoff-visualizer'>Option Payoff Visualizer</h2>", unsafe_allow_html=True)
st.info('''
The plot below helps visualize the profit/loss from an option of a given price, for different stock prices at expiry. 
        
The tool allows you to see what the combined total payoff will be for up to two different options.
''')

num = int(st.radio("**Number of Options:**", ['1', '2'], horizontal=True))

strike_prices = []
option_prices = []
option_types = []
position_types = []

for i in range(num):
    st.markdown(f"#### Option {i + 1}:")

    position_type_col, option_type_col, strike_price_col, option_price_col = st.columns([1, 1, 1, 1])

    with position_type_col:
        position_types.append(st.radio(
            f"**Position Type:**", 
            ['Long', 'Short'], 
            horizontal=True,
            key=f"position_type_{i}")
        )

    with option_type_col:
        option_types.append(st.radio(
            f"**Option Type:**", 
            ['Call', 'Put'], 
            horizontal=True,
            key=f"option_type_{i}")
        )

    with strike_price_col:
        strike_prices.append(st.number_input(
            f"**Strike Price:**", 
            value=100.0, 
            step=1.0, 
            key=f"strike_price_{i}")
        )

    with option_price_col:
        option_prices.append(st.number_input(
            f"**{option_types[i]} Price:**", 
            value=5.00, 
            step=0.1, 
            key=f"option_price_{i}")
        )

K_min, K_max = min(strike_prices), max(strike_prices)
stock_prices = np.linspace(max(0, K_min - 15), K_max + 15, 500)

individual_payoffs = []

for i in range(num):
    payoff = compute_option_payoff(strike_prices[i], option_prices[i], option_types[i], position_types[i], stock_prices)
    individual_payoffs.append(payoff)

total_payoff = np.sum(individual_payoffs, axis=0)

# Plotting with Plotly
title_payoff = "Option Payoff Plot" if num == 1 else "Combined Options Payoff Plot"

# Create the title text
if num == 1:
    title_text = f"{title_payoff}<br>{position_types[0]} a {strike_prices[0]:.0f} {option_types[0]} at {option_prices[0]:.1f}"
elif num == 2:
    title_text = (f"{title_payoff}<br>"
                  f"{position_types[0]} a {strike_prices[0]:.0f} {option_types[0]} at {option_prices[0]:.1f}<br>"
                  f"{position_types[1]} a {strike_prices[1]:.0f} {option_types[1]} at {option_prices[1]:.1f}<br>"
                  )


# Initialize Plotly figure
fig = go.Figure()

# Add individual payoff lines
line_styles = ['dash', 'dot']
for i, payoff in enumerate(individual_payoffs):
    fig.add_trace(go.Scatter(
        x=stock_prices,
        y=payoff,
        mode='lines',
        line=dict(color='blue', dash=line_styles[i]),
        name=f"{position_types[i]} a {strike_prices[i]:.0f} {option_types[i]} at {option_prices[i]:.1f}",
        hovertemplate='Stock Price: %{x:.2f}<br>Payoff: %{y:.2f}<extra></extra>'
    ))

# Add strike price vertical lines
for strike in strike_prices:
    fig.add_vline(x=strike, line=dict(color="black", width=1, dash="dash"), annotation_text=f"K={strike:.0f}")

# Add a dummy trace for the strike price legend
fig.add_trace(go.Scatter(
    x=[None],  # No actual data points
    y=[None],
    mode='lines',
    line=dict(color="black", dash="dash", width=1),
    name="Strike Price",
    showlegend=True
))

# Add total payoff line if two options
if num == 2:
    fig.add_trace(go.Scatter(
        x=stock_prices,
        y=total_payoff,
        mode='lines',
        line=dict(color='red', width=3),
        name="Total Payoff",
        hovertemplate='Stock Price: %{x:.2f}<br>Payoff: %{y:.2f}<extra></extra>'
    ))

# Add zero line
fig.add_hline(y=0, line=dict(color="black", width=1))

# Find break-even points and collect them for a single trace
crossings = np.where(np.diff(np.sign(total_payoff)))[0]
x_crossings = []
y_crossings = []
text_labels = []

for idx in crossings:
    x0, x1 = stock_prices[idx], stock_prices[idx + 1]
    y0, y1 = total_payoff[idx], total_payoff[idx + 1]
    slope = (y1 - y0) / (x1 - x0)
    x_cross = x0 - y0 / slope
    y_cross = 0

    x_crossings.append(x_cross)
    y_crossings.append(y_cross)
    text_labels.append(f"{x_cross:.1f}")

# Add all break-even points as a single trace
if x_crossings:  # Only add the trace if there are break-even points
    fig.add_trace(go.Scatter(
        x=x_crossings,
        y=y_crossings,
        mode='markers+text',
        marker=dict(symbol='x', color='red', size=10),
        text=text_labels,
        textposition="top center",
        name='Break-Even Point',  # Single legend entry
        hovertemplate='Stock Price: %{x:.2f}<br>Payoff: %{y:.2f}<extra></extra>',
        showlegend=True
    ))

# Update layout
fig.update_layout(
    title=dict(text=title_text, x=0.5, xanchor='center', font=dict(size=18)),
    xaxis_title="Stock Price at Expiry",
    yaxis_title="Profit / Loss",
    showlegend=True,
    legend=dict(
    x=0.05,      
    y=0.95,    
    xanchor='left',
    yanchor='top',
    bordercolor='black',
    borderwidth=1),
    hovermode='closest',
    width=800,
    height=800
)

# Display the plot in Streamlit
st.plotly_chart(fig, use_container_width=True)

#-----------------------------------------------------------------------------------------------------
#Display call/put values in heatmap - allow choice for whether table is for call price or put price, and whether x axis is strike or spot price

st.markdown("---")
heatmap_title_col, download_button_col = st.columns([8, 2])

with heatmap_title_col:
    st.header('Heatmap of Option Prices')

# Create an empty container for the download button to fill later
download_button_container = download_button_col.empty()

st.info('''
This heatmap shows the calculated fair value call/put prices for a range of volatilities and spot/strike prices.
        
Terminal condition, when T = 0:
- Call Price = max(S - K, 0)
- Put Price = max(K - S, 0)
''')

#Create radio buttons
option_type_heatmap_col, price_type_heatmap_col = st.columns([1,1], gap="large")

with option_type_heatmap_col: 
    option_type_heatmap = st.radio("**Option Price:**", ['Call', 'Put'], horizontal=True)

with price_type_heatmap_col:
    price_type_heatmap = st.radio("**Stock Price (x-axis of heatmap):**", ['Strike Price', 'Spot Price'], horizontal=True)

# Sliders and number inputs
sigma_range_heatmap_col, stock_range_heatmap_col = st.columns([1,1], gap="large")

with sigma_range_heatmap_col:
    sigma_min, sigma_max = st.slider("**Volatility Range for Heatmap:**", min_value=0.01, max_value=1.0, value=(0.1, 0.3), step=0.01)

with stock_range_heatmap_col:
    stock_min_heatmap_col, stock_max_heatmap_col = st.columns(2)
    with stock_min_heatmap_col:
        price_min_heatmap = st.number_input("**Min Stock Price**", min_value=1, max_value=5000, value=80, step=1)
    with stock_max_heatmap_col:
        price_max_heatmap = st.number_input("**Max Stock Price**", min_value=1, max_value=5000, value=120, step=1)

# Prepare heatmap inputs
price_range_heatmap = np.linspace(price_min_heatmap, price_max_heatmap, 10)
sigma_range_heatmap = np.linspace(sigma_min, sigma_max, 10)

st.write("")

#Creating heatmap
option_prices_heatmap = option_matrix(
    T=T,
    r=r,
    sigma_range=sigma_range_heatmap,
    price_range=price_range_heatmap,
    S_fixed=S,
    K_fixed=K,
    is_call=(option_type_heatmap == 'Call'),
    x_axis=price_type_heatmap
)

# Plot and label heatmap
label_title_heatmap = f"Spot Price = {round(S, 1)}" if price_type_heatmap == "Strike Price" else f"Strike Price = {round(K, 1)}"

title_heatmap = f"{option_type_heatmap.upper()} Option Price Heatmap<br>({label_title_heatmap}; Time to Maturity = {round(T,2)} years; Risk Free Rate = {round(r_pc,2)}%)"

st.markdown(
    f"<h3 style='text-align: center; font-size: 18px; font-weight: 600;'>{title_heatmap}</h3>",
    unsafe_allow_html=True
)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(option_prices_heatmap,
            xticklabels=np.round(price_range_heatmap, 1),
            yticklabels=np.round(sigma_range_heatmap, 2),
            annot=True, fmt=".2f", cmap="viridis", ax=ax)

ax.set_xlabel(price_type_heatmap)
ax.set_ylabel("Volatility")
ax.invert_yaxis()

st.pyplot(fig)

buf = io.BytesIO()
fig.savefig(buf, format="png")
buf.seek(0)

download_button_container.download_button(
    label="Download ⬇",
    data=buf,
    file_name="option_price_heatmap.png",
    mime="image/png"
)

#--------------------------------------------------------------------------------------------
#Create 3d plots to visualise how the option prices change with time and stock price

st.markdown("------")
st.markdown("<h2 id='bsm-visualization'>BSM Visualization</h2>", unsafe_allow_html=True)

st.info('''This 3D plot shows how the theoretical option price varies with time to maturity and strike/spot price. 
        Adjust the values of the risk-free rate, volatilty and spot/strike price using the sidebar on the left.
        ''')

option_type_surface_col, price_type_surface_col = st.columns([1,1], gap="large")

with option_type_surface_col: 
    option_type_surface = st.radio("**Option Price:**", ['Call', 'Put'], horizontal=True, key="option_type_vis")

with price_type_surface_col:
    price_type_surface = st.radio("**Stock Price (x-axis of plot):**", ['Strike Price', 'Spot Price'], horizontal=True, key="price_type_vis")

T_range_surface_col, price_range_surface_col = st.columns([1,1], gap="large")

with T_range_surface_col:
    T_max = st.slider("Time left until maturity, T (years)", min_value = 0.0, max_value = 3.0, value = 1.0, step = 0.01, key = "T_slider")

with price_range_surface_col:
    price_min_surface_col, price_max_surface_col = st.columns(2)
    with price_min_surface_col:
        price_min_surface = st.number_input("**Min Stock Price**", min_value=1, max_value=1000, value=80, step=1, key = "price_min_vis")
    with price_max_surface_col:
        price_max_surface = st.number_input("**Max Stock Price**", min_value=1, max_value=1000, value=120, step=1, key = "price_max_vis")



#3D plot
T_range_surface = np.linspace(0.01, T_max, 100)
price_range_surface = np.linspace(price_min_surface, price_max_surface, 100)

option_prices_surface = option_visualiser(
    T_range=T_range_surface,
    price_range=price_range_surface,
    r=r,
    sigma=sigma,
    S_fixed=S,
    K_fixed=K,
    is_call=(option_type_surface == 'Call'),
    x_axis=price_type_surface
)

option_prices_surface = option_prices_surface.T

st.write("")
st.write("")

label_title_surface = f"Spot Price = {round(S, 1)}" if price_type_surface == "Strike Price" else f"Strike Price = {round(K, 1)}"

st.markdown(
    f"<h3 style='text-align: center; font-size: 18px; font-weight: 600;'>{option_type_surface.upper()} Option Price Surface<br>({label_title_surface}; Risk free Rate = {round(r_pc,2)}%; Volatility = {round(sigma_pc,2)}%)</h3>",
    unsafe_allow_html=True
)

fig = go.Figure(data=[
    go.Surface(
        z=option_prices_surface,
        x=T_range_surface,
        y=price_range_surface,
        colorscale="Viridis",
        showscale=True
    )
])

fig.update_layout(
    scene=dict(
        xaxis=dict(title="Time until maturity, T"),
        yaxis=dict(title=f"{price_type_surface}, {'K' if price_type_surface == 'Strike Price' else 'S'}"),
        zaxis=dict(title=f"{option_type_surface} Price, {'C' if option_type_surface == 'Call' else 'P'}")
    ),
    height=700,
    margin=dict(l=0, r=0, b=0, t=50)
)

st.plotly_chart(fig, use_container_width=True)

#-----------------------------------------------------------------------------------------------------------

st.markdown("------")
st.markdown("<h2 id='implied-volatility-calculator'>Implied Volatility Calculator</h2>", unsafe_allow_html=True)

st.info('''Calculated implied-volatility for a given option price. 
        
        Adjust the input variable values using the sidebar on the left.
        ''')

variable_values_iv = {
    "Variable": ["Spot Price", "Strike Price", "Time to Maturity (years)", "Risk Free Interest Rate"],
    "Value": [round(S,1), round(K,1), round(T,2), f"{round(r_pc,2)}%"]
}
variable_data_iv = pd.DataFrame(variable_values_iv)

html_iv = variable_data_iv.to_html(index=False)

styled_html_iv = f"""
<style>
    table {{
        border-collapse: collapse;
        width: 100%;
    }}
    th {{
        font-weight: bold !important;
        color: black !important;
        background-color: #f8f8f8;
        border-bottom: 2px solid #ddd;
        padding: 8px;
        text-align: left !important;
    }}
    td {{
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }}
</style>
{html_iv}
"""

option_type_iv_col, real_option_price_col = st.columns([1,1])

with option_type_iv_col:
    option_type_iv = st.radio("**Option Type:**", ['Call', 'Put'], key = "option_type_iv", horizontal = True)


with real_option_price_col:
    real_option_price = st.number_input(f"**{option_type_iv} Price:**", value = 20.0, step = 0.1, key = "real_option_price_1")

iv = implied_vol(T, K, S, r, real_option_price, is_call=(option_type_iv == 'Call'))

table_iv_col, iv_col = st.columns([1,1])

with table_iv_col:
        st.markdown(styled_html_iv, unsafe_allow_html=True)

with iv_col:
    st.markdown(f"""
        <div style="display: flex;flex-direction: column;justify-content: space-between;border: 1px solid #ddd;padding: 10px;
                border-radius: 5px; max-width: 400px; margin-bottom: 10px; margin-left: 0px; min-height: 190px;"><div>
        <div class="metric-label" style="font-weight: bold; font-size: 28px; color: #333;">Calculated Implied Volatility</div>
        <div class="metric-sub-label" style="font-size: 14px; color: #666;">({option_type_iv} Option)</div>
        </div>
        <div class="metric-value" style="font-size: 30px; font-weight: bold; color: #007acc;">{iv * 100:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

#--------------------------------------------------------------------------------------------
#Create heatmap to show if an option is over/underpriced and by how much

st.markdown("------")
st.markdown("<h2 id='price-mismatch-heatmap'>Price Mismatch Heatmap</h2>", unsafe_allow_html=True)

st.info('''
Heatmap showing how the real market price of a call option compares to its theoretical Black-Scholes price across a range of assumed volatilities. 
        
For a given option price: 
- Green indicates underpricing: market price is lower than the model price at that volatility (assumed volatility is higher than option's implied volatility)
- Red indicates overpricing: market price is higher than the model price (assumed volatility is lower than option's implied volatility) 
        
This visualisation helps assess the relative mispricing of an option depending on different volatility assumptions 
(negative value indicates how much cheaper real option price is than theoretical option price at given variable values; positive value indicates how much more expensive real is than theoretical).
''')

sigma_range_iv_col, stock_type_iv_col = st.columns([1,1], gap="large")

with sigma_range_iv_col:
    sigma_min_iv, sigma_max_iv = st.slider("**Volatility Range for Heatmap:**", min_value=0.01, max_value=1.0, value=(0.1, 0.3), step=0.01, key = "sigma_range_iv")

with stock_type_iv_col:
    stock_type_iv = st.radio("**Stock Price (x-axis of heatmap):**", ['Strike Price', 'Spot Price'], horizontal=True, key = "stock_price_iv") 



stock_range_iv_col, real_option_price_col = st.columns([1,1], gap = "large")

with stock_range_iv_col:
    stock_min_iv_col, stock_max_iv_col = st.columns(2)
    with stock_min_iv_col:
        price_min_iv = st.number_input(f"**Min {stock_type_iv}:**", min_value=1.0, max_value=1000.0, value=80.0, step=0.1, key = "price_min_iv")
    with stock_max_iv_col:
        price_max_iv = st.number_input(f"**Max {stock_type_iv}:**", min_value=1.0, max_value=1000.0, value=120.0, step=0.1, key = "price_max_iv")

with real_option_price_col:
    real_option_price = st.number_input(f"**{option_type_iv} Price:**", value = 20.0, step = 0.1, key = "real_option_price_2")


price_range_iv = np.linspace(price_min_iv, price_max_iv, 10)
sigma_range_iv = np.linspace(sigma_min_iv, sigma_max_iv, 10)

st.write("")
st.write("")

option_prices_valuemap = option_matrix(
    T=T,
    r=r,
    sigma_range=sigma_range_iv,
    price_range=price_range_iv,
    S_fixed=S,
    K_fixed=K,
    is_call=(option_type_iv == 'Call'),
    x_axis=stock_type_iv
)

#Heatmap and plot
custom_cmap = LinearSegmentedColormap.from_list(
    "mispricing_cmap", ["red", "white", "green"]
)

price_diff = option_prices_valuemap - real_option_price
option_price_mismatch = option_prices_valuemap > real_option_price

# Plot and label heatmap
title_heatmap = f"{option_type_iv.upper()} Option Mispricing Heatmap <br> Real {option_type_iv} Price = {round(real_option_price,1)} <br>(Spot price = {round(S,1)}; Time to Maturity = {round(T,2)} years; Risk Free Rate = {round(r_pc,2)}%)"

st.markdown(
    f"<h3 style='text-align: center; font-size: 18px; font-weight: 600;'>{title_heatmap}</h3>",
    unsafe_allow_html=True
)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(price_diff,
            xticklabels=np.round(price_range_iv, 1),
            yticklabels=np.round(sigma_range_iv, 2),
            annot=(-1 * price_diff), fmt=".1f", cmap=custom_cmap, center = 0, ax=ax, linecolor = "black", linewidths = 0.4, cbar = False)


ax.set_xlabel(f"{stock_type_iv}")
ax.set_ylabel("Theoretical Volatility")
ax.invert_yaxis()

st.pyplot(fig)
