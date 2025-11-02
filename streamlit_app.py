import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="Intraday Buy/Sell Signals", layout="wide")

st.title("📊 Intraday Auto Signal System (Manual Trading)")
st.caption("AI-based logic for BUY/SELL/HOLD — Indian market (1-min data)")

# Sidebar
st.sidebar.header("Settings")

symbols_text = st.sidebar.text_area("Enter Stock Symbols (NSE)", value="RELIANCE.NS\nTCS.NS\nICICIBANK.NS\nHDFCBANK.NS\nINFY.NS")
symbols = [s.strip() for s in symbols_text.splitlines() if s.strip()]

top_n = st.sidebar.slider("Top N stocks", 3, 10, 5)
refresh_btn = st.sidebar.button("🔄 Refresh Data")

# Telegram details (optional)
telegram_token = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
telegram_chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "")
send_alerts = st.sidebar.checkbox("Send Telegram Alerts", value=False)

# Helper functions
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=6):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()

def get_signal(row):
    price, vwap_, ema9, ema21, rsi6, vol_ratio = row
    if np.isnan([price, vwap_, ema9, ema21, rsi6, vol_ratio]).any():
        return "HOLD"
    if price > vwap_ and ema9 > ema21 and 55 <= rsi6 <= 70 and vol_ratio >= 1.5:
        return "BUY"
    elif price < vwap_ and ema9 < ema21 and 30 <= rsi6 <= 45 and vol_ratio >= 1.3:
        return "SELL"
    else:
        return "HOLD"

def send_telegram_message(text):
    if not telegram_token or not telegram_chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        payload = {"chat_id": telegram_chat_id, "text": text}
        requests.post(url, data=payload, timeout=5)
    except:
        pass

# Main logic
if st.button("🚀 Scan Stocks") or refresh_btn:
    with st.spinner("Fetching live data... please wait ⏳"):
        records = []
        end = datetime.utcnow()
        start = end - timedelta(minutes=60)

        for sym in symbols:
            try:
                df = yf.download(sym, interval="1m", start=start, end=end, progress=False, threads=False)
                if df.empty:
                    continue
                df = df.dropna()
                df["VWAP"] = vwap(df)
                df["EMA9"] = ema(df["Close"], 9)
                df["EMA21"] = ema(df["Close"], 21)
                df["RSI6"] = rsi(df["Close"], 6)
                df["VolRatio"] = df["Volume"] / df["Volume"].rolling(20).mean()

                latest = df.iloc[-1]
                signal = get_signal([
                    latest["Close"], latest["VWAP"], latest["EMA9"],
                    latest["EMA21"], latest["RSI6"], latest["VolRatio"]
                ])
                records.append({
                    "Symbol": sym,
                    "Price": round(latest["Close"], 2),
                    "RSI6": round(latest["RSI6"], 1),
                    "VWAP": round(latest["VWAP"], 2),
                    "EMA9": round(latest["EMA9"], 2),
                    "EMA21": round(latest["EMA21"], 2),
                    "VolRatio": round(latest["VolRatio"], 2),
                    "Signal": signal
                })

                if send_alerts and signal in ["BUY", "SELL"]:
                    emoji = "🟢" if signal == "BUY" else "🔴"
                    msg = f"{emoji} {signal} Signal\n{sym}\nPrice ₹{latest['Close']:.2f}"
                    send_telegram_message(msg)
            except Exception as e:
                st.warning(f"Error fetching {sym}: {e}")

        if records:
            df_final = pd.DataFrame(records)
            def color_signal(val):
                if val == "BUY": return "background-color: #b6f5c4"
                if val == "SELL": return "background-color: #ffd6d6"
                return ""
            st.dataframe(df_final.style.applymap(color_signal, subset=["Signal"]))
        else:
            st.error("No data fetched. Try again later.")

st.markdown("---")
st.caption("💡 Tip: Run between 9:30 AM – 3:00 PM IST for best accuracy.")