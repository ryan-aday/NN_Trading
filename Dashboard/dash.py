import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
from datetime import datetime
import streamlit as st
import time

st.set_page_config(page_title="Options Trading Dashboard", layout="wide")

# --- Black-Scholes & Greeks ---
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    except:
        return 0

def get_greeks(S, K, T, r, sigma, option_type='call'):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)
        return delta, gamma, theta, vega, rho
    except:
        return 0, 0, 0, 0, 0

def get_alpha_beta(stock, benchmark='^GSPC'):
    try:
        hist = stock.history(period="6mo")['Close'].pct_change().dropna()
        bench = yf.Ticker(benchmark).history(period="6mo")['Close'].pct_change().dropna()
        df = pd.concat([hist, bench], axis=1).dropna()
        df.columns = ['stock', 'bench']
        X = sm.add_constant(df['bench'])
        model = sm.OLS(df['stock'], X).fit()
        alpha, beta = model.params['const'], model.params['bench']
        return alpha * 252, beta
    except:
        return np.nan, np.nan

def get_options_data(ticker, r=0.05):
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")['Close'].iloc[-1]
        expiry_dates = stock.options
        if not expiry_dates:
            return pd.DataFrame()
        expiry = expiry_dates[0]
        options = stock.option_chain(expiry)
        calls = options.calls
        puts = options.puts
        expiration = datetime.strptime(expiry, "%Y-%m-%d")
        T = max((expiration - datetime.now()).days / 365, 1/252)

        hist = stock.history(period="1mo")['Close']
        sigma = np.std(np.log(hist / hist.shift(1)).dropna()) * np.sqrt(252)
        if sigma == 0 or np.isnan(sigma):
            sigma = 0.01

        alpha, beta = get_alpha_beta(stock)

        results = []
        for option_df, otype in zip([calls, puts], ['call', 'put']):
            for _, row in option_df.iterrows():
                if row['volume'] < 100 or pd.isna(row['lastPrice']):
                    continue
                K = row['strike']
                market_price = row['lastPrice']
                theo_price = black_scholes_price(price, K, T, r, sigma, otype)
                if theo_price == 0:
                    continue
                mispricing = theo_price - market_price
                mispricing_pct = mispricing / market_price
                delta, gamma, theta, vega, rho = get_greeks(price, K, T, r, sigma, otype)
                eod_target = price * np.exp(sigma * np.sqrt(1/252))
                expected_option_eod = black_scholes_price(eod_target, K, T - 1/252, r, sigma, otype)
                expected_gain = expected_option_eod - market_price
                risk_score = (0.4 * abs(delta) + 0.2 * abs(gamma) + 0.15 * abs(theta) + 0.15 * abs(vega) + 0.1 * abs(beta))
                results.append({
                    'Expiry': expiry,
                    'Ticker': ticker,
                    'Type': otype,
                    'Strike': K,
                    'Last': market_price,
                    'BS_Price': round(theo_price, 2),
                    '%Mispricing': round(mispricing_pct * 100, 2),
                    'Volume': row['volume'],
                    'OI': row['openInterest'],
                    'Delta': round(delta, 2),
                    'Gamma': round(gamma, 4),
                    'Theta': round(theta, 2),
                    'Vega': round(vega, 2),
                    'Rho': round(rho, 2),
                    'Alpha': round(alpha, 4),
                    'Beta': round(beta, 2),
                    'EOD Target': round(eod_target, 2),
                    'Expected EOD Value': round(expected_option_eod, 2),
                    'Expected Gain': round(expected_gain, 2),
                    'Risk Score': round(risk_score, 3)
                })
        df = pd.DataFrame(results)
        df = df[df['Expected Gain'] > 0].sort_values(by='Expected Gain', ascending=False)
        if not df.empty:
            df['Suggested Weight %'] = round(df['Expected Gain'] / df['Expected Gain'].sum() * 100, 2)
        return df
    except:
        return pd.DataFrame()

# --- UI Layout ---
st.title("📊 Options Trading Dashboard")
st.markdown("Live trading suggestions with theoretical pricing, Greeks, risk scoring, and portfolio weighting.")

sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500_df = pd.read_html(sp500_url)[0]
sp500_tickers = sp500_df['Symbol'].tolist()

selected_tickers = st.multiselect("Choose tickers to evaluate:", sp500_tickers, default=['MSFT', 'META', 'NVDA', 'TSLA', 'MPWR'])
manual_input = st.text_input("Add a custom ticker (e.g., MSFT, META, SPY)")
refresh = st.button("🔄 Refresh Now")

# Filters
min_gain = st.sidebar.slider("Minimum Expected Gain ($)", 0.0, 25.0, 1.0, step=0.5)
max_risk = st.sidebar.slider("Maximum Risk Score", 0.0, 10.0, 1.0, step=0.1)

# Add manual ticker if it's not already selected
if manual_input:
    manual_input = manual_input.upper()
    if manual_input not in selected_tickers:
        test_df = get_options_data(manual_input)
        if test_df.empty:
            st.warning(f"⚠️ Ticker '{manual_input}' could not be added — no options data found.")
        else:
            selected_tickers.append(manual_input)

master_df = pd.DataFrame()
if refresh or True:
    for ticker in selected_tickers:
        with st.spinner(f"Fetching data for {ticker}..."):
            df = get_options_data(ticker)
            if not df.empty:
                master_df = pd.concat([master_df, df])

if master_df.empty:
    st.warning("No viable options found for the selected tickers.")
else:
    master_df = master_df[(master_df['Expected Gain'] >= min_gain) & (master_df['Risk Score'] <= max_risk)]
    st.success(f"Found {len(master_df)} tradeable options.")
    alerts = master_df[master_df['%Mispricing'] > 20]
    if not alerts.empty:
        st.sidebar.markdown("### 🚨 Trade Alerts")
        st.sidebar.dataframe(alerts[['Ticker', 'Type', 'Strike', '%Mispricing', 'Expected Gain']], use_container_width=True)

    st.dataframe(master_df.reset_index(drop=True), use_container_width=True)
    csv = master_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Table as CSV", data=csv, file_name="options_dashboard.csv", mime='text/csv')
