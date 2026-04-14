"""
app.py — Mini-Aladdin Trading Dashboard
Simple, clean Streamlit app for Pradeep.
Run with: /opt/homebrew/bin/python3.11 -m streamlit run app.py
"""

from datetime import date, timedelta
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from strategy_simple import STRATEGY_REGISTRY, build_strategy, BUY, SELL
from backtest_simple import run_backtest, compare_strategies, results_to_df, BacktestResult

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mini-Aladdin",
    page_icon="📈",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("📈 Mini-Aladdin")
st.sidebar.caption("Your Personal Trading System")
st.sidebar.divider()

symbol = st.sidebar.text_input("Stock Ticker", value="AAPL").upper().strip()
start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=3*365))
end_date = st.sidebar.date_input("End Date", value=date.today() - timedelta(days=1))

strategy_name = st.sidebar.selectbox("Strategy", list(STRATEGY_REGISTRY.keys()))

# Dynamic params
st.sidebar.subheader("Parameters")
param_defs = STRATEGY_REGISTRY[strategy_name]["param_defs"]
default_params = STRATEGY_REGISTRY[strategy_name]["default_params"]
user_params = {}
for param, pdef in param_defs.items():
    user_params[param] = st.sidebar.slider(
        param, pdef["min"], pdef["max"], default_params[param]
    )

st.sidebar.divider()
capital = st.sidebar.number_input("Capital ($)", min_value=100, value=10000, step=500)
commission = st.sidebar.slider("Commission %", 0.0, 1.0, 0.1, 0.01) / 100
slippage = st.sidebar.slider("Slippage %", 0.0, 1.0, 0.1, 0.01) / 100

st.sidebar.subheader("Risk Management")
stop_loss = st.sidebar.slider("Stop Loss %", 0, 20, 5) / 100
take_profit = st.sidebar.slider("Take Profit %", 0, 50, 10) / 100
run_btn = st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Backtest", "⚖️ Compare All", "⚡ Live Quote"])

# ── Tab 1: Backtest ──────────────────────────────────────────────────────────
with tab1:
    st.header(f"Backtest — {symbol}")

    if run_btn:
        with st.spinner("Running backtest..."):
            try:
                strategy = build_strategy(strategy_name, **user_params)
                result = run_backtest(symbol, strategy, str(start_date), str(end_date), capital, commission, slippage, stop_loss_pct=stop_loss, take_profit_pct=take_profit)
                st.session_state["result"] = result
            except Exception as e:
                st.error(f"Error: {e}")

    result: BacktestResult = st.session_state.get("result")

    if result is None:
        st.info("👈 Set your parameters and click **Run Backtest**")
    else:
        # Metrics
        m = result.metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return", f"{m['total_return']}%")
        c2.metric("CAGR", f"{m['cagr']}%")
        c3.metric("Sharpe Ratio", f"{m['sharpe_ratio']}")
        c4.metric("Max Drawdown", f"{m['max_drawdown']}%")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Win Rate", f"{m['win_rate']}%")
        c6.metric("Profit Factor", f"{m['profit_factor']}x")
        c7.metric("Total Trades", m['total_trades'])
        c8.metric("Final Value", f"${m['final_value']:,.2f}")

        st.divider()

        # Equity curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result.equity_curve.index, y=result.equity_curve.values,
            fill="tozeroy", fillcolor="rgba(0,212,170,0.1)",
            line=dict(color="#00d4aa", width=2), name="Portfolio"
        ))
        fig.add_hline(y=capital, line_dash="dash", line_color="gray", annotation_text="Start")
        fig.update_layout(title="Equity Curve", template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

        # Drawdown
        rolling_max = result.equity_curve.cummax()
        drawdown = (result.equity_curve - rolling_max) / rolling_max * 100
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values,
            fill="tozeroy", fillcolor="rgba(255,75,75,0.15)",
            line=dict(color="#ff4b4b", width=1.5), name="Drawdown"
        ))
        fig2.update_layout(title="Drawdown %", template="plotly_dark", height=220)
        st.plotly_chart(fig2, use_container_width=True)

        # Price + signals
        df = result.signals_df
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df.index, y=df["Close"], line=dict(color="#aaa", width=1), name="Price"))
        buys = df[df["signal"] == BUY]
        sells = df[df["signal"] == SELL]
        fig3.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers",
                                   marker=dict(color="lime", size=10, symbol="triangle-up"), name="BUY"))
        fig3.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers",
                                   marker=dict(color="red", size=10, symbol="triangle-down"), name="SELL"))
        fig3.update_layout(title="Price + Signals", template="plotly_dark", height=320)
        st.plotly_chart(fig3, use_container_width=True)

        # Trades
        st.subheader(f"Trade Log ({m['total_trades']} trades)")
        if not result.trades.empty:
            st.dataframe(result.trades, use_container_width=True)
        else:
            st.warning("No completed trades in this period.")

# ── Tab 2: Compare ───────────────────────────────────────────────────────────
with tab2:
    st.header("Compare All Strategies")
    if st.button("▶️ Run Comparison", type="primary"):
        with st.spinner("Running all strategies..."):
            try:
                results = compare_strategies(symbol, str(start_date), str(end_date), capital)
                st.session_state["compare"] = results
            except Exception as e:
                st.error(f"Error: {e}")

    compare = st.session_state.get("compare")
    if compare:
        df_compare = results_to_df(compare)
        st.dataframe(df_compare, use_container_width=True)

        fig = go.Figure()
        colors = ["#00d4aa", "#ff9f43", "#a29bfe"]
        for i, r in enumerate(compare):
            norm = (r.equity_curve / r.equity_curve.iloc[0] - 1) * 100
            fig.add_trace(go.Scatter(x=norm.index, y=norm.values,
                                      line=dict(color=colors[i], width=2), name=r.strategy_name))
        fig.update_layout(title="Normalized Return % — All Strategies", template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Click **Run Comparison** to see all strategies side by side.")

# ── Tab 3: Live Quote ─────────────────────────────────────────────────────────
with tab3:
    st.header("Live Quote")
    q_symbol = st.text_input("Enter ticker", value="AAPL").upper().strip()
    if st.button("Fetch", type="primary"):
        try:
            ticker = yf.Ticker(q_symbol)
            info = ticker.fast_info
            price = info.last_price
            prev = info.previous_close
            change = price - prev
            change_pct = change / prev * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("Price", f"${price:,.2f}", f"{change_pct:+.2f}%")
            c2.metric("Change", f"${change:+.2f}")
            c3.metric("Volume", f"{int(info.last_volume):,}")

            # 30-day chart
            hist = ticker.history(period="1mo")
            fig = go.Figure(go.Candlestick(
                x=hist.index, open=hist["Open"], high=hist["High"],
                low=hist["Low"], close=hist["Close"]
            ))
            fig.update_layout(title=f"{q_symbol} — Last 30 Days",
                              template="plotly_dark", height=400,
                              xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not fetch quote: {e}")
