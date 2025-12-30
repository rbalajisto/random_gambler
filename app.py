import streamlit as st
import kagglehub
import os
import random
import pandas as pd
from datetime import timedelta

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Quant Portfolio Simulator",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Quant Strategy – Portfolio Backtest")
st.caption("Randomized stock selection with Reliance trend bias")

# -------------------- USER INPUTS --------------------
st.sidebar.header("⚙️ Strategy Inputs")

start = st.sidebar.date_input(
    "Start Date",
    value=pd.to_datetime("2008-01-31")
)

end = st.sidebar.date_input(
    "End Date",
    value=pd.to_datetime("2011-01-31")
)

n_random = st.sidebar.number_input(
    "Stocks per Day (n_random)",
    min_value=1,
    max_value=10,
    value=3
)

init_wallet = st.sidebar.number_input(
    "Initial Wallet (₹)",
    min_value=10000,
    value=100000,
    step=10000
)

run = st.sidebar.button("🚀 Run Backtest")

# -------------------- LOAD DATA --------------------
@st.cache_data(show_spinner=True)
def load_data():
    DATA_PATH = kagglehub.dataset_download("rohanrao/nifty50-stock-market-data")

    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]
    files.remove("stock_metadata.csv")
    files.remove("NIFTY50_all.csv")

    reliance = os.path.join(DATA_PATH, "RELIANCE.csv")
    rDF = pd.read_csv(reliance, parse_dates=["Date"])
    rDF = rDF.sort_values("Date").set_index("Date")

    return DATA_PATH, files, rDF

# -------------------- HELPERS --------------------
def load_ticker_on_date(DATA_PATH, filename, trade_date):
    df = pd.read_csv(os.path.join(DATA_PATH, filename), parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df.loc[df.index == trade_date]

def strategy(trade_date, wallet, rel, filenames, DATA_PATH, n_random):
    if wallet <= 0:
        return [0] * n_random

    picks = random.sample(filenames, n_random)
    per_trade = wallet / n_random
    pnl = []

    for f in picks:
        df = load_ticker_on_date(DATA_PATH, f, trade_date)
        if df.empty:
            pnl.append(0)
            continue

        open_p = df.iloc[0]["Open"]
        close_p = df.iloc[0]["Last"]

        qty = int(per_trade / open_p)
        trade_val = (close_p - open_p) * qty

        pnl.append(trade_val if sum(rel) >= 0 else -trade_val)

    return pnl

def run_backtest(start, end, init_wallet, n_random):
    DATA_PATH, filenames, rDF = load_data()

    wallet = init_wallet
    peak = init_wallet
    max_dd = 0

    curr = pd.to_datetime(start)
    end = pd.to_datetime(end)
    delta = timedelta(days=1)

    rel = []
    wallet_curve = []
    dates = []

    success, loss = 0, 0

    # Seed Reliance trend
    rel_date = curr - delta
    for _ in range(20):
        try:
            row = rDF.loc[rel_date]
            rel.append(1 if row["Open"] <= row["Last"] else -1)
        except:
            rel.append(0)
        rel_date -= delta

    while curr <= end:
        try:
            row = rDF.loc[curr]
            rel.append(1 if row["Open"] < row["Last"] else -1)
        except:
            curr += delta
            continue

        pnl = strategy(curr, wallet, rel, filenames, DATA_PATH, n_random)

        for p in pnl:
            wallet += p
            success += int(p > 0)
            loss += int(p <= 0)

        peak = max(peak, wallet)
        max_dd = max(max_dd, (peak - wallet) / peak)

        wallet_curve.append(wallet)
        dates.append(curr)

        rel.pop(0)
        curr += delta

    return {
        "dates": dates,
        "wallet": wallet_curve,
        "final": wallet,
        "returns": (wallet - init_wallet) / init_wallet * 100,
        "success": success,
        "loss": loss,
        "peak": peak,
        "max_dd": max_dd * 100
    }

# -------------------- RUN --------------------
if run:
    with st.spinner("Running backtest..."):
        result = run_backtest(start, end, init_wallet, n_random)

    # -------------------- METRICS --------------------
    st.subheader("📌 Performance Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Final Value", f"₹ {result['final']:,.0f}")
    c2.metric("Total Return", f"{result['returns']:.2f} %")
    c3.metric("Max Drawdown", f"{result['max_dd']:.2f} %")
    c4.metric("Peak Wallet", f"₹ {result['peak']:,.0f}")

    # -------------------- CHART --------------------
    st.subheader("📈 Portfolio Value Over Time")

    chart_df = pd.DataFrame({
        "Date": result["dates"],
        "Wallet": result["wallet"]
    }).set_index("Date")

    st.line_chart(chart_df, height=400)

    # -------------------- TRADE STATS --------------------
    st.subheader("📊 Trade Statistics")

    t1, t2 = st.columns(2)
    t1.success(f"✅ Profitable Trades: {result['success']}")
    t2.error(f"❌ Loss Trades: {result['loss']}")

    st.caption("Model uses random stock selection with Reliance trend bias")
