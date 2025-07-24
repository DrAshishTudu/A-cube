# streamlit_app.py

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime
import requests
from streamlit_autorefresh import st_autorefresh
import os

# === CONFIG ===
STOCKS = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
INTERVAL = "15m"
RANGE = "5d"
BB_LENGTH = 20
BB_STD = 2

# Telegram setup
TELEGRAM_TOKEN = os.getenv("8495666727:AAH2qChhVHK85PL0khlSkSHemMFGgWrVpYA")
TELEGRAM_CHAT_ID = os.getenv("Acube3Bot")


# === FUNCTIONS ===
def fetch_data(symbol):
    df = yf.download(symbol, interval=INTERVAL, period=RANGE)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)

    # ✅ Fix for missing Datetime column
    if "Datetime" not in df.columns:
        if "Date" in df.columns:
            df.rename(columns={"Date": "Datetime"}, inplace=True)
        elif "index" in df.columns:
            df.rename(columns={"index": "Datetime"}, inplace=True)

    return df


def add_indicators(df):
    # 🛠 Ensure 'Close' is 1D
    if isinstance(df["Close"].values, np.ndarray) and df["Close"].values.ndim > 1:
        df["Close"] = pd.Series(df["Close"].values.flatten(), index=df.index)

    # ✅ Bollinger Bands
    df["bb_mid"] = df["Close"].rolling(BB_LENGTH).mean()
    df["bb_std"] = df["Close"].rolling(BB_LENGTH).std()
    df["bb_upper"] = df["bb_mid"] + BB_STD * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - BB_STD * df["bb_std"]

    # ✅ RSI
    close_series = pd.Series(df["Close"].values.flatten(), index=df.index)
    df["rsi"] = RSIIndicator(close_series).rsi()

    # ✅ MACD
    macd = MACD(close_series)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    return df


def predict_price(df):
   # 🧼 Step 1: Normalize column names
    df.columns = [str(col).strip().lower() for col in df.columns]
    print("✅ Columns after cleanup:", df.columns.tolist())

    # 🔁 Step 2: Try to rename time-related column to 'timestamp'
    if 'time' in df.columns:
        df.rename(columns={'time': 'timestamp'}, inplace=True)
    elif 'date' in df.columns:
        df.rename(columns={'date': 'timestamp'}, inplace=True)

    # 🧪 Step 3: Show columns again
    print("🛠 Columns after possible renaming:", df.columns.tolist())

    # 🚫 Step 4: Final check
    # Try to auto-rename 'Time' or 'Date' to 'timestamp'
for col in df.columns:
    if "time" in col.lower() or "date" in col.lower():
        df.rename(columns={col: "timestamp"}, inplace=True)

# Try to rename 'Close' if it's something else
for col in df.columns:
    if col.lower() == "close":
        df.rename(columns={col: "close"}, inplace=True)

# Check again
if "timestamp" not in df.columns or "close" not in df.columns:
    print("🧪 DEBUG Columns in df:", df.columns.tolist())
    raise KeyError("❌ Required columns 'timestamp' or 'close' not found in DataFrame.")

    # ✅ Step 5: Drop missing
    df = df.dropna(subset=["timestamp", "close"]).copy()
    
    # Convert timestamp to Unix time
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp"] = df["timestamp"].astype(np.int64) // 10**9

    # Train model
    X = df["timestamp"].values.reshape(-1, 1)
    y = df["close"].values

    model = LinearRegression()
    model.fit(X, y)

    # Predict for next 15 minutes
    next_time = np.array([[X[-1][0] + 900]])
    predicted_price = model.predict(next_time)[0]

    return round(predicted_price, 2)


def check_strategy(df):
    c24 = df.iloc[-2]
    upper_bb = c24["bb_upper"]
    f1 = c24["Open"] > upper_bb
    f2 = c24["Close"] > upper_bb
    f3 = c24["Close"] > c24["Open"]
    return f1 and f2 and f3


def send_telegram(message):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        requests.post(url, data=payload)


def plot_chart(df, symbol):
    fig = go.Figure(data=[go.Candlestick(x=df['Datetime'],
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'],
                                         name="Candlesticks")])
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['bb_upper'], line=dict(color='orange'), name="BB Upper"))
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['bb_lower'], line=dict(color='blue'), name="BB Lower"))
    fig.update_layout(title=symbol, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)


# === STREAMLIT APP ===

st.set_page_config(layout="wide")
st.title("📊 TradingView-Style Scanner with AI & Alerts")

# Refresh every 15 min (900000 ms)
st_autorefresh(interval=900000, key="datarefresh")

for symbol in STOCKS:
    st.header(f"🧠 {symbol}")
    df = fetch_data(symbol)
    df = add_indicators(df)

    predicted_price = predict_price(df)
    strategy_passed = check_strategy(df)

    col1, col2 = st.columns([2, 1])
    with col1:
        plot_chart(df, symbol)

    with col2:
        st.metric("Predicted Close Price (next 15 min)", f"₹{predicted_price}")
        st.metric("RSI", round(df['rsi'].iloc[-1], 2))
        st.metric("MACD", round(df['macd'].iloc[-1], 2))
        st.metric("Signal Line", round(df['macd_signal'].iloc[-1], 2))

        if strategy_passed:
            st.success("✅ Strategy Passed!")
            send_telegram(f"🚨 {symbol} passed the strategy. Consider buying!")
        else:
            st.warning("❌ Strategy Not Passed.")

    st.markdown("---")
