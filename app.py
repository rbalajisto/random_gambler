import streamlit as st
import kagglehub
import os
import random
import pandas as pd
import numpy as np
from datetime import timedelta

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Quant Portfolio Backtest",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Quant Strategy Backtest Dashboard")
st.caption("Randomized portfolio with Reliance trend bias | Professional risk metrics")

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("⚙️ Strategy Parameters")

start = pd.to_datetime(
    st.sidebar.date_input("Start Date", pd.to_datetime("2008-01-31"))
)

end = pd.to_datetime(
    st.sidebar.date_input("End Date", pd.to_datetime("2011-01-31"))
)

n_random = st.sidebar.number_input(
    "Stocks per Day",
    min_value=1,
    max_value=10,
    value=3
)

init_wallet = st.sidebar.number_input(
    "Initial Wallet (₹)",
    min_value=10000,
    step=10000,
    value=100000
)

run = st.sidebar.button("🚀 Run Backtest")

# ---------------- LOAD DATA ----------------
@st.cache_data(show_spinner=True)
def load_data():
    DATA_PATH = kagglehub.dataset_download("rohanrao/nifty50-stock-market-data")

    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]
    files.remove("stock_metadata.csv")
    files.remove("NIFTY50_all.csv")

    reliance = pd.read_csv(
        os.path.join(DATA_PATH, "RELIANCE.csv"),
        parse_dates=["Date"]
    ).sort_values("Date").set_index("Date")

    nifty = pd.read_csv(
        os.path.join(DATA_PATH, "NIFTY50_all.csv"),
        parse_dates=["Date"]
    ).sort_values("Date").set_index("Date")

    return DATA_PATH, files, reliance, nifty

# ---------------- HELPERS ----------------
def load_stock_on_date(DATA_PATH, filename, date):
    df = pd.read_csv(
        os.path.join(DATA_PATH, filename),
        parse_dates=["Date"]
    ).sort_values("Date").set_index("Date")
    return df.loc[df.index == date]

def strategy(date, wallet, rel, filenames, DATA_PATH, n_random):
    if wallet <= 0:
        return [0] * n_random

    picks = random.sample(filenames, n_random)
    per_trade = wallet / n_random
    pnl = []

    for f in picks:
        df = load_stock_on_date(DATA_PATH, f, date)
        if df.empty:
            pnl.append(0)
            continue

        open_p = df.iloc[0]["Open"]
        close_p = df.iloc[0]["Last"]

        qty = int(per_trade / open_p)
        trade_val = (close_p - open_p) * qty

        pnl.append(trade_val if sum(rel) >= 0 else -trade_val)

    return pnl

# ---------------- BACKTEST ENGINE ----------------
def run_backtest(start, end, init_wallet, n_random):
    DATA_PATH, filenames, rDF, nifty = load_data()

    wallet = init_wallet
    peak = init_wallet
    max_dd = 0

    curr = start
    delta = timedelta(days=1)

    wallet_curve = []
    dates = []
    daily_returns = []

    success, loss = 0, 0
    rel = []

    # Seed Reliance momentum
    rel_date = curr - delta
    for _ in range(20):
        try:
            row = rDF.loc[rel_date]
            rel.append(1 if row["Open"] <= row["Last"] else -1)
        except:
            rel.append(0)
        rel_date -= delta

    prev_wallet = wallet

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

        daily_ret = (wallet - prev_wallet) / prev_wallet if prev_wallet > 0 else 0
        daily_returns.append(daily_ret)
        prev_wallet = wallet

        peak = max(peak, wallet)
        max_dd = max(max_dd, (peak - wallet) / peak)

        wallet_curve.append(wallet)
        dates.append(curr)

        rel.pop(0)
        curr += delta

    equity = pd.DataFrame({
        "Date": dates,
        "Wallet": wallet_curve,
        "Returns": daily_returns
    }).set_index("Date")

    # ---------------- METRICS ----------------
    years = (end - start).days / 365.25
    cagr = ((wallet / init_wallet) ** (1 / years) - 1) * 100

    sharpe = (
        np.mean(equity["Returns"]) /
        np.std(equity["Returns"])
    ) * np.sqrt(252) if np.std(equity["Returns"]) != 0 else 0

    # Benchmark CAGR
    nifty_slice = nifty.loc[(nifty.index >= start) & (nifty.index <= end)]
    nifty_cagr = (
        (nifty_slice["Close"].iloc[-1] / nifty_slice["Close"].iloc[0])
        ** (1 / years) - 1
    ) * 100

    alpha = cagr - nifty_cagr

    # Yearly returns heatmap
    yearly_returns = equity["Wallet"].resample("Y").last().pct_change() * 100
    heatmap_df = yearly_returns.to_frame("Return")
    heatmap_df["Year"] = heatmap_df.index.year

    return {
        "equity": equity,
        "final": wallet,
        "total_return": (wallet - init_wallet) / init_wallet * 100,
        "cagr": cagr,
        "sharpe": sharpe,
        "alpha": alpha,
        "max_dd": max_dd * 100,
        "success": success,
        "loss": loss,
        "heatmap": heatmap_df
    }

# ---------------- RUN ----------------
if run:
    with st.spinner("Running quantitative backtest..."):
        result = run_backtest(start, end, init_wallet, n_random)

    # ---------------- METRICS ----------------
    st.subheader("📌 Performance Metrics")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Final Value", f"₹ {result['final']:,.0f}")
    c2.metric("Total Return", f"{result['total_return']:.2f}%")
    c3.metric("CAGR", f"{result['cagr']:.2f}%")
    c4.metric("Sharpe Ratio", f"{result['sharpe']:.2f}")
    c5.metric("Alpha vs NIFTY", f"{result['alpha']:.2f}%")

    # ---------------- EQUITY CURVE ----------------
    st.subheader("📈 Portfolio Equity Curve")
    st.line_chart(result["equity"]["Wallet"], height=400)

    # ---------------- HEATMAP ----------------
    st.subheader("🔥 Yearly Returns Heatmap")

    heatmap = result["heatmap"].pivot_table(
        values="Return",
        index="Year"
    )

    st.dataframe(
        heatmap.style.background_gradient(
            cmap="RdYlGn", axis=0
        ).format("{:.2f}%"),
        use_container_width=True
    )

    # ---------------- TRADE STATS ----------------
    st.subheader("📊 Trade Statistics")
    t1, t2 = st.columns(2)
    t1.success(f"✅ Profitable Trades: {result['success']}")
    t2.error(f"❌ Loss Trades: {result['loss']}")

    st.caption(
        "Metrics: CAGR, Alpha (vs NIFTY 50), Sharpe (risk-free = 0), Max Drawdown"
    )
