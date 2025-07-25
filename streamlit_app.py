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

    # ✅ Ensure Datetime column exists
    if df.index.name is not None:
        df["Datetime"] = df.index
    else:
        df.reset_index(inplace=True)
        if "index" in df.columns:
            df.rename(columns={"index": "Datetime"}, inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df


def add_indicators(df):
    if "Close" not in df.columns:
        raise KeyError("❌ 'Close' column missing in DataFrame.")

    df["bb_mid"] = df["Close"].rolling(BB_LENGTH).mean()
    df["bb_std"] = df["Close"].rolling(BB_LENGTH).std()
    df["bb_upper"] = df["bb_mid"] + BB_STD * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - BB_STD * df["bb_std"]

    close_series = pd.Series(df["Close"].values.flatten(), index=df.index)
    df["rsi"] = RSIIndicator(close_series).rsi()

    macd = MACD(close_series)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    return df


def predict_price(df):
    df.columns = [str(col).strip().lower() for col in df.columns]

    # Rename if needed
    for col in df.columns:
        if "time" in col or "date" in col:
            df.rename(columns={col: "timestamp"}, inplace=True)
        if "close" in col and col != "close":
            df.rename(columns={col: "close"}, inplace=True)

    if "timestamp" not in df.columns or "close" not in df.columns:
        raise KeyError("❌ Required columns 'timestamp' or 'close' not found in DataFrame.")

    df = df.dropna(subset=["timestamp", "close"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp"] = df["timestamp"].astype(np.int64) // 10**9

    X = df["timestamp"].values.reshape(-1, 1)
    y = df["close"].values

    model = LinearRegression()
    model.fit(X, y)

    next_time = np.array([[X[-1][0] + 900]])  # 15 min later
    predicted_price = model.predict(next_time)[0]

    return round(predicted_price, 2)


def check_strategy(df):
    if df.shape[0] < 2 or "bb_upper" not in df.columns:
        return False
    c24 = df.iloc[-2]
    f1 = c24["Open"] > c24["bb_upper"]
    f2 = c24["Close"] > c24["bb_upper"]
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
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['bb_upper'], line=dict(color='orange'), name="BB Upper"))
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['bb_lower'], line=dict(color='blue'), name="BB Lower"))

    fig.update_layout(title=symbol, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)


# === STREAMLIT APP ===

st.set_page_config(layout="wide")
st.title("📊 TradingView-Style Scanner with AI & Alerts")

st_autorefresh(interval=900000, key="datarefresh")  # 15 min

for symbol in STOCKS:
    st.header(f"🧠 {symbol}")

    try:
        df = fetch_data(symbol)
        df = add_indicators(df)
        predicted_price = predict_price(df)
        strategy_passed = check_strategy(df)

        col1, col2 = st.columns([2, 1])
        with col1:
            if "Datetime" in df.columns:
                plot_chart(df, symbol)
            else:
                st.error(f"❌ Datetime column missing in {symbol}")

        with col2:
            st.metric("Predicted Close Price (next 15 min)", f"₹{predicted_price}")

            if 'rsi' in df.columns:
                st.metric("RSI", round(df['rsi'].iloc[-1], 2))
            if 'macd' in df.columns:
                st.metric("MACD", round(df['macd'].iloc[-1], 2))
            if 'macd_signal' in df.columns:
                st.metric("Signal Line", round(df['macd_signal'].iloc[-1], 2))

            if strategy_passed:
                st.success("✅ Strategy Passed!")
                send_telegram(f"🚨 {symbol} passed the strategy. Consider buying!")
            else:
                st.warning("❌ Strategy Not Passed.")

    except Exception as e:
        st.error(f"💥 Error in {symbol}: {e}")

    st.markdown("---")
