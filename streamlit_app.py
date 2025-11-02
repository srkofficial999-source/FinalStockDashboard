# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import time
from datetime import datetime, timedelta
import requests
import math
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import os

# --- Telegram helper ---
def send_telegram_message(bot_token: str, chat_id: str, text: str):
    try:
        if not bot_token or not chat_id:
            return False, "No bot token/chat id"
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        resp = requests.post(url, data=payload, timeout=5)
        if resp.status_code == 200:
            return True, resp.text
        else:
            return False, f"{resp.status_code}: {resp.text}"
    except Exception as e:
        return False, str(e)

# --- Utility indicators / processing ---
def compute_indicators(df: pd.DataFrame):
    # df expected to have DatetimeIndex and columns: Open, High, Low, Close, Volume
    if df.empty:
        return df
    # VWAP (pandas_ta has vwap for intraday, but we'll compute custom cumulative typical*vol / cum vol)
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0
    df['TPV'] = tp * df['Volume']
    df['cum_TPV'] = df['TPV'].cumsum()
    df['cum_vol'] = df['Volume'].cumsum()
    df['VWAP'] = df['cum_TPV'] / df['cum_vol']
    # EMAs
    df['EMA9'] = ta.ema(df['Close'], length=9)
    df['EMA21'] = ta.ema(df['Close'], length=21)
    # RSI short
    df['RSI6'] = ta.rsi(df['Close'], length=6)
    # ATR percent (using 14)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ATR_pct'] = (df['ATR'] / df['Close']) * 100
    # rolling avg volume
    df['vol_mean20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
    # volume ratio
    df['vol_ratio'] = df['Volume'] / df['vol_mean20']
    return df

# --- Signal logic (Buy / Sell / Hold) as described ---
def compute_signal(latest_row: pd.Series):
    try:
        price = latest_row['Close']
        vwap = latest_row['VWAP']
        ema9 = latest_row['EMA9']
        ema21 = latest_row['EMA21']
        rsi6 = latest_row['RSI6']
        vol_ratio = latest_row['vol_ratio']

        # Basic NA checks
        if np.isnan([vwap, ema9, ema21, rsi6, vol_ratio]).any():
            return "HOLD", "Insufficient data"

        # Buy conditions
        buy_cond = (
            (price > vwap) and
            (ema9 > ema21) and
            (rsi6 >= 55 and rsi6 <= 70) and
            (vol_ratio >= 1.5)
        )
        # Sell conditions
        sell_cond = (
            (price < vwap) and
            (ema9 < ema21) and
            (rsi6 >= 30 and rsi6 <= 45) and
            (vol_ratio >= 1.3)
        )

        if buy_cond:
            return "BUY", "Above VWAP + EMA9>EMA21 + RSI in [55-70] + Vol>=1.5x"
        elif sell_cond:
            return "SELL", "Below VWAP + EMA9<EMA21 + RSI in [30-45] + Vol>=1.3x"
        else:
            return "HOLD", "Neutral/No clear setup"
    except Exception as e:
        return "HOLD", f"Error in signal: {e}"

# --- Stock selection logic: top movers by volume ratio + ATR% + % change ---
def select_top_movers(symbol_list, period_minutes=30, top_n=8):
    """
    symbol_list: list of symbols (NSE tickers as yfinance expects, e.g., 'RELIANCE.NS')
    period_minutes: how much recent history to consider for intraday activity
    """
    records = []
    end = datetime.utcnow()
    start = end - timedelta(minutes=period_minutes*2)  # fetch a bit more to be safe
    for sym in symbol_list:
        try:
            df = yf.download(sym, interval="1m", start=start, end=end, progress=False, threads=False)
            if df.empty:
                continue
            df = df.dropna()
            df = compute_indicators(df)
            latest = df.iloc[-1]
            # percent change from open of day (approx using earliest fetched)
            percent_change = ((latest['Close'] - df['Open'].iloc[0]) / df['Open'].iloc[0]) * 100
            vol_ratio = latest['vol_ratio'] if 'vol_ratio' in latest else 0
            atr_pct = latest['ATR_pct'] if 'ATR_pct' in latest else 0
            # delivery% not available via yfinance; skip that filter in this simple version
            records.append({
                "symbol": sym,
                "pct_change": percent_change,
                "vol_ratio": vol_ratio,
                "atr_pct": atr_pct,
                "price": latest['Close']
            })
        except Exception as e:
            # skip symbol on error
            continue
    df_rec = pd.DataFrame(records)
    if df_rec.empty:
        return []
    # apply simple thresholds then sort: prefer high vol_ratio and pct_change magnitude and ATR%
    df_rec['score'] = (df_rec['vol_ratio'] * 2.0) + (df_rec['atr_pct'] * 1.0) + (df_rec['pct_change'].abs() * 1.0)
    df_rec = df_rec.sort_values(by='score', ascending=False)
    top = df_rec.head(top_n)
    return top.to_dict('records')

# --- Helper to format Telegram message ---
def format_alert(symbol, signal, reason, price, ts):
    emoji = "🟢" if signal == "BUY" else ("🔴" if signal == "SELL" else "⚪")
    text = f"{emoji} <b>{signal} Signal</b>\nSymbol: <b>{symbol}</b>\nTime: {ts}\nPrice: ₹{price:.2f}\nReason: {reason}"
    return text

# --- Main streamlit app ---
st.set_page_config(page_title="Intraday Signals - Auto Scanner", layout="wide")
st.title("Intraday Signal Scanner — Auto stock pick + BUY/SELL/HOLD")
st.write("Short & simple signals for manual intraday trading. Use paper/live discretion. (No guarantees.)")

# Sidebar inputs / secrets
st.sidebar.header("Settings")
tickers_input = st.sidebar.text_area("Ticker list (one per line, yfinance format, e.g., RELIANCE.NS)", value="""
RELIANCE.NS
TCS.NS
ICICIBANK.NS
HDFCBANK.NS
INFY.NS
HINDUNILVR.NS
LT.NS
AXISBANK.NS
JSWSTEEL.NS
ONGC.NS
""".strip(), height=220)
top_n = st.sidebar.number_input("Top N movers to watch", min_value=3, max_value=15, value=8)
interval_minutes = st.sidebar.selectbox("Indicator period (minutes)", options=[1,5], index=0)
auto_refresh = st.sidebar.checkbox("Auto refresh every 60s", value=False)
refresh_seconds = st.sidebar.slider("Auto refresh interval (sec)", min_value=30, max_value=300, value=60, step=10)

st.sidebar.markdown("---")
st.sidebar.subheader("Telegram Alerts (optional)")
telegram_token = st.secrets.get("TELEGRAM_BOT_TOKEN", "") if "TELEGRAM_BOT_TOKEN" in st.secrets else st.sidebar.text_input("Bot Token (or set in Streamlit secrets)", value="")
telegram_chat_id = st.secrets.get("TELEGRAM_CHAT_ID", "") if "TELEGRAM_CHAT_ID" in st.secrets else st.sidebar.text_input("Chat ID (or set in Streamlit secrets)", value="")
send_alerts = st.sidebar.checkbox("Send Telegram alerts", value=False)

st.sidebar.markdown("---")
st.sidebar.write("Notes:")
st.sidebar.write("- App uses yfinance 1-min data for intraday. Not tick-level. Good for signal generation & manual trades.")
st.sidebar.write("- Avoid trading during extreme news. Use small size while testing.")
st.sidebar.write(" - First 5 minutes after market open may be noisy. Use filters.")

# Prepare symbol list
symbols = [s.strip() for s in tickers_input.splitlines() if s.strip()]
if not symbols:
    st.error("Please enter at least one ticker symbol in the sidebar (example format: RELIANCE.NS).")
    st.stop()

# Top movers selection
st.subheader("Auto Stock Selection (Top movers)")
col1, col2 = st.columns([1,3])

with col1:
    if st.button("Refresh Top Movers Now"):
        st.session_state['refresh_trigger'] = time.time()

with col2:
    st.write("Selects top movers by recent volume spike, ATR% and % change. Updates on refresh or auto-refresh.")

# Run selection
with st.spinner("Selecting top movers..."):
    top_movers = select_top_movers(symbols, period_minutes=30, top_n=top_n)
    if len(top_movers) == 0:
        st.warning("No movers found or data fetch failed. Try fewer symbols or try again.")
    else:
        top_df = pd.DataFrame(top_movers)
        st.dataframe(top_df[['symbol','pct_change','vol_ratio','atr_pct','price']].rename(columns={
            'symbol':'Symbol','pct_change':'%Change','vol_ratio':'VolRatio','atr_pct':'ATR%','price':'Price'
        }), height=250)

# Signal table
st.subheader("Live Signals (for selected top movers)")
signal_rows = []
for rec in top_movers:
    sym = rec['symbol']
    try:
        # fetch recent few minutes
        end = datetime.utcnow()
        start = end - timedelta(minutes=60)  # one hour
        df = yf.download(sym, interval=f"{interval_minutes}m", start=start, end=end, progress=False, threads=False)
        if df.empty:
            continue
        df = df.dropna()
        df = compute_indicators(df)
        latest = df.iloc[-1]
        signal, reason = compute_signal(latest)
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        signal_rows.append({
            "Symbol": sym,
            "Price": latest['Close'],
            "VWAP": latest.get('VWAP', np.nan),
            "EMA9": latest.get('EMA9', np.nan),
            "EMA21": latest.get('EMA21', np.nan),
            "RSI6": latest.get('RSI6', np.nan),
            "VolRatio": latest.get('vol_ratio', np.nan),
            "Signal": signal,
            "Reason": reason,
            "Time": ts
        })
    except Exception as e:
        continue

if signal_rows:
    signals_df = pd.DataFrame(signal_rows)
    # pretty column order
    signals_df = signals_df[["Symbol","Price","VWAP","EMA9","EMA21","RSI6","VolRatio","Signal","Reason","Time"]]
    # color map for signal
    def color_signal(val):
        if val == "BUY":
            return "background-color: #b6f5c4"
        elif val == "SELL":
            return "background-color: #ffd6d6"
        else:
            return ""
    st.dataframe(signals_df, height=320)
else:
    st.write("No live signals available now.")

# Telegram alerts handling and log
if 'alert_log' not in st.session_state:
    st.session_state['alert_log'] = []

if send_alerts and telegram_token and telegram_chat_id:
    st.success("Telegram alerts enabled (will send for newly generated signals when Refresh is clicked or auto-refresh runs).")

# Button to send alerts manually for current signals
if st.button("Send Telegram Alerts for current signals"):
    if not telegram_token or not telegram_chat_id:
        st.error("Set Telegram bot token and chat id first (in sidebar or Streamlit secrets).")
    else:
        sent_count = 0
        for r in signal_rows:
            text = format_alert(r['Symbol'], r['Signal'], r['Reason'], float(r['Price']), r['Time'])
            ok, resp = send_telegram_message(telegram_token, telegram_chat_id, text)
            st.session_state['alert_log'].append({
                "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": r['Symbol'],
                "signal": r['Signal'],
                "sent": ok,
                "resp": str(resp)[:200]
            })
            if ok:
                sent_count += 1
        st.info(f"Alerts attempted: {len(signal_rows)} | Sent: {sent_count}")

# Alert log panel
st.subheader("Alert Log (recent)")
log_df = pd.DataFrame(st.session_state['alert_log'])
if not log_df.empty:
    st.dataframe(log_df.tail(20))
else:
    st.write("No alerts sent yet.")

# Auto-refresh scheduler (background) - only if user enables
if auto_refresh:
    if 'scheduler' not in st.session_state:
        st.session_state['scheduler'] = BackgroundScheduler()
        st.session_state['scheduler'].start()

    def job_refresh():
        # Refresh button action: just trigger a rerun by updating a hidden state
        st.session_state['last_refresh'] = datetime.utcnow().isoformat()
    # remove existing jobs
    try:
        for j in st.session_state['scheduler'].get_jobs():
            st.session_state['scheduler'].remove_job(j.id)
    except Exception:
        pass
    st.session_state['scheduler'].add_job(job_refresh, 'interval', seconds=refresh_seconds, id="auto_refresh_job")
    st.write(f"Auto-refresh running every {refresh_seconds}s. (Background scheduler active.)")

# final note
st.markdown("---")
st.write("App built for scanning signals only. Use proper risk management. Start with paper testing.")