"""
dashboard.py — Mini-Aladdin Trading Board
Streamlit dashboard for backtesting, strategy comparison, and live quotes.
Run with: streamlit run dashboard.py
"""

import logging
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from agent import TradingAgent
from backtest import BacktestEngine, BacktestResult, results_to_comparison_df
from data_fetch import get_provider
from strategy import STRATEGY_REGISTRY, build_strategy, BUY, SELL

logging.basicConfig(level=logging.WARNING)

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mini-Aladdin Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Cached resources
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_engine():
    return BacktestEngine()

@st.cache_resource
def get_agent():
    return TradingAgent()

@st.cache_resource
def get_data_provider():
    return get_provider("yfinance")

@st.cache_data(ttl=3600)
def fetch_bars(symbol, timeframe, start, end):
    provider = get_data_provider()
    return provider.get_bars(symbol, timeframe, start, end)

@st.cache_data(ttl=60)
def fetch_quote(symbol):
    provider = get_data_provider()
    return provider.get_latest_quote(symbol)

# ──────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ──────────────────────────────────────────────────────────────────────────────
def plot_equity_curve(result: BacktestResult) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.equity_curve.index,
        y=result.equity_curve.values,
        mode="lines",
        name="Portfolio Value",
        line=dict(color="#00d4aa", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.08)",
    ))
    fig.add_hline(
        y=result.initial_capital,
        line_dash="dash",
        line_color="gray",
        annotation_text="Initial Capital",
        annotation_position="top left",
    )
    fig.update_layout(
        title=f"Equity Curve — {result.symbol} | {result.strategy_name}",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        template="plotly_dark",
        height=400,
    )
    return fig


def plot_drawdown(result: BacktestResult) -> go.Figure:
    equity = result.equity_curve
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        mode="lines",
        name="Drawdown %",
        line=dict(color="#ff4b4b", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255,75,75,0.15)",
    ))
    fig.update_layout(
        title="Drawdown (%)",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        template="plotly_dark",
        height=250,
    )
    return fig


def plot_price_with_signals(result: BacktestResult) -> go.Figure:
    df = result.signals_df
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines", name="Price",
        line=dict(color="#aaaaaa", width=1),
    ))

    buys = df[df["signal"] == BUY]
    sells = df[df["signal"] == SELL]

    fig.add_trace(go.Scatter(
        x=buys.index, y=buys["Close"],
        mode="markers", name="BUY",
        marker=dict(color="lime", size=10, symbol="triangle-up"),
    ))
    fig.add_trace(go.Scatter(
        x=sells.index, y=sells["Close"],
        mode="markers", name="SELL",
        marker=dict(color="red", size=10, symbol="triangle-down"),
    ))

    fig.update_layout(
        title=f"Price Chart + Signals — {result.symbol}",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        template="plotly_dark",
        height=350,
    )
    return fig


def plot_comparison_curves(results: list[BacktestResult]) -> go.Figure:
    fig = go.Figure()
    colors = ["#00d4aa", "#ff9f43", "#a29bfe", "#fd79a8", "#74b9ff", "#55efc4"]

    for i, r in enumerate(results):
        normalized = (r.equity_curve / r.equity_curve.iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized.values,
            mode="lines",
            name=r.strategy_name,
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.update_layout(
        title="Strategy Comparison — Normalized Return (%)",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        hovermode="x unified",
        template="plotly_dark",
        height=400,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Metrics display
# ──────────────────────────────────────────────────────────────────────────────
def display_metrics(metrics: dict):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", f"{metrics['total_return']}%",
                delta=f"{metrics['total_return']}%")
    col2.metric("CAGR", f"{metrics['cagr']}%")
    col3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']}")
    col4.metric("Max Drawdown", f"-{abs(metrics['max_drawdown'])}%",
                delta=f"-{abs(metrics['max_drawdown'])}%", delta_color="inverse")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Win Rate", f"{metrics['win_rate']}%")
    col6.metric("Profit Factor", f"{metrics['profit_factor']}x")
    col7.metric("Total Trades", f"{metrics['total_trades']}")
    col8.metric("Final Value", f"${metrics['final_value']:,.2f}")


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
def sidebar():
    st.sidebar.image("https://img.icons8.com/nolan/96/combo-chart.png", width=64)
    st.sidebar.title("Mini-Aladdin")
    st.sidebar.caption("AI-Powered Trading System")
    st.sidebar.divider()

    symbol = st.sidebar.text_input("Ticker Symbol", value="SPY", max_chars=10).upper().strip()

    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start", value=date.today() - timedelta(days=3*365))
    end_date = col2.date_input("End", value=date.today() - timedelta(days=1))

    strategy_name = st.sidebar.selectbox("Strategy", options=list(STRATEGY_REGISTRY.keys()))

    # Dynamic param inputs
    st.sidebar.subheader("Strategy Parameters")
    param_defs = STRATEGY_REGISTRY[strategy_name]["param_defs"]
    default_params = STRATEGY_REGISTRY[strategy_name]["default_params"]
    user_params = {}

    for param_name, pdef in param_defs.items():
        if pdef["type"] == "int":
            user_params[param_name] = st.sidebar.slider(
                param_name,
                min_value=pdef["min"],
                max_value=pdef["max"],
                value=default_params[param_name],
            )
        elif pdef["type"] == "select":
            user_params[param_name] = st.sidebar.selectbox(
                param_name,
                options=pdef["options"],
                index=pdef["options"].index(default_params[param_name]),
            )

    st.sidebar.divider()
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)", min_value=1000, max_value=10_000_000, value=10_000, step=1000
    )
    commission = st.sidebar.slider("Commission (%)", 0.0, 1.0, 0.1, 0.01) / 100
    slippage = st.sidebar.slider("Slippage (%)", 0.0, 1.0, 0.1, 0.01) / 100

    run_btn = st.sidebar.button("🚀 Run Backtest", use_container_width=True, type="primary")

    return {
        "symbol": symbol,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "strategy_name": strategy_name,
        "user_params": user_params,
        "initial_capital": initial_capital,
        "commission": commission,
        "slippage": slippage,
        "run_btn": run_btn,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Tab 1: Backtest
# ──────────────────────────────────────────────────────────────────────────────
def tab_backtest(cfg: dict):
    st.header("📊 Backtest Results")

    if cfg["run_btn"]:
        with st.spinner(f"Running backtest for {cfg['symbol']}..."):
            try:
                engine = get_engine()
                strategy = build_strategy(cfg["strategy_name"], **cfg["user_params"])
                result = engine.run(
                    cfg["symbol"], strategy,
                    cfg["start_date"], cfg["end_date"],
                    cfg["initial_capital"], cfg["commission"], cfg["slippage"],
                )
                st.session_state["last_result"] = result
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                return

    result: BacktestResult = st.session_state.get("last_result")

    if result is None:
        st.info("Configure your settings in the sidebar and click **Run Backtest** to get started.")
        return

    # Metrics
    st.subheader("Performance Metrics")
    display_metrics(result.metrics)

    st.divider()

    # Charts
    st.plotly_chart(plot_equity_curve(result), use_container_width=True)
    st.plotly_chart(plot_drawdown(result), use_container_width=True)
    st.plotly_chart(plot_price_with_signals(result), use_container_width=True)

    st.divider()

    # Trades table
    st.subheader(f"Trade Log ({result.metrics['total_trades']} trades)")
    if not result.trades.empty:
        st.dataframe(result.trades, use_container_width=True)
    else:
        st.warning("No completed trades in this period.")

    st.divider()

    # AI Analysis
    st.subheader("🤖 AI Analysis")
    if st.button("Analyze with AI"):
        agent = get_agent()
        with st.spinner("Analyzing..."):
            analysis = agent.analyze_backtest(result)
        st.markdown(analysis)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2: Compare Strategies
# ──────────────────────────────────────────────────────────────────────────────
def tab_compare(cfg: dict):
    st.header("⚖️ Strategy Comparison")
    st.caption("Run all strategies on the same symbol and period to find the best performer.")

    if st.button("🔄 Compare All Strategies", type="primary"):
        with st.spinner(f"Backtesting all strategies on {cfg['symbol']}..."):
            try:
                engine = get_engine()
                results = engine.compare_strategies(
                    cfg["symbol"],
                    cfg["start_date"],
                    cfg["end_date"],
                    cfg["initial_capital"],
                )
                st.session_state["compare_results"] = results
            except Exception as e:
                st.error(f"Comparison failed: {e}")
                return

    results = st.session_state.get("compare_results")
    if results is None:
        st.info("Click **Compare All Strategies** to run the analysis.")
        return

    # Ranked table
    st.subheader("Ranked by Sharpe Ratio")
    comparison_df = results_to_comparison_df(results)
    st.dataframe(
        comparison_df.style.format({
            "total_return": "{:.2f}%",
            "cagr": "{:.2f}%",
            "sharpe_ratio": "{:.3f}",
            "max_drawdown": "{:.2f}%",
            "win_rate": "{:.2f}%",
            "profit_factor": "{:.3f}",
        }),
        use_container_width=True,
    )

    st.divider()

    # Overlapping equity curves
    st.plotly_chart(plot_comparison_curves(results), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3: Live Quote
# ──────────────────────────────────────────────────────────────────────────────
def tab_live_quote():
    st.header("⚡ Live Quote")
    st.caption("Fetch the latest price for any US stock or ETF.")

    quote_symbol = st.text_input("Enter ticker", value="AAPL", max_chars=10).upper().strip()

    if st.button("Fetch Quote", type="primary"):
        try:
            with st.spinner("Fetching..."):
                quote = fetch_quote(quote_symbol)
            col1, col2, col3 = st.columns(3)
            col1.metric("Price", f"${quote['price']:,.4f}",
                        delta=f"{quote['change_pct']}%")
            col2.metric("Change", f"${quote['change']:+.4f}")
            col3.metric("Volume", f"{quote['volume']:,}")
            st.caption(f"Last updated: {quote['timestamp']} UTC")
        except Exception as e:
            st.error(f"Could not fetch quote: {e}")

    # 30-day chart
    st.subheader("30-Day Price Chart")
    try:
        from datetime import date, timedelta
        end = date.today().strftime("%Y-%m-%d")
        start = (date.today() - timedelta(days=45)).strftime("%Y-%m-%d")
        df = fetch_bars(quote_symbol, "1d", start, end)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name=quote_symbol,
        ))
        fig.update_layout(
            title=f"{quote_symbol} — Last 30 Days",
            template="plotly_dark",
            height=400,
            xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load chart: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Tab 4: AI Strategy Builder
# ──────────────────────────────────────────────────────────────────────────────
def tab_ai_builder():
    st.header("🤖 AI Strategy Builder")
    st.caption("Describe a strategy idea in plain English and the AI will convert it into a backtest config.")

    user_idea = st.text_area(
        "Describe your strategy idea",
        placeholder='e.g. "momentum strategy on large-cap tech stocks" or "RSI mean reversion on SPY"',
        height=100,
    )

    if st.button("Generate Strategy Config", type="primary") and user_idea:
        agent = get_agent()
        with st.spinner("Thinking..."):
            config = agent.suggest_strategy(user_idea)

        st.subheader("Generated Config")
        st.json(config)

        if st.button("▶️ Run This Backtest"):
            try:
                engine = get_engine()
                strategy = build_strategy(config["strategy"], **config.get("params", {}))
                result = engine.run(
                    config["symbol"], strategy,
                    config["start_date"], config["end_date"],
                )
                st.session_state["last_result"] = result
                st.success("Backtest complete! Switch to the **Backtest** tab to see results.")
            except Exception as e:
                st.error(f"Backtest failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    cfg = sidebar()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Backtest",
        "⚖️ Compare Strategies",
        "⚡ Live Quote",
        "🤖 AI Builder",
    ])

    with tab1:
        tab_backtest(cfg)
    with tab2:
        tab_compare(cfg)
    with tab3:
        tab_live_quote()
    with tab4:
        tab_ai_builder()


if __name__ == "__main__":
    main()
