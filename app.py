"""
app.py — Mini-Aladdin Trading Dashboard
A beginner-friendly trading system for non-technical traders.
Run with: streamlit run app.py
"""

from datetime import date, timedelta
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from strategy_simple import STRATEGY_REGISTRY, build_strategy, BUY, SELL
from backtest_simple import run_backtest, compare_strategies, results_to_df, BacktestResult

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mini-Aladdin Trading System",
    page_icon="📈",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# Theme (Dark/Light)
# ──────────────────────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True

dark_mode = st.session_state["dark_mode"]
chart_theme = "plotly_dark" if dark_mode else "plotly_white"

bg_color = "#0E1117" if dark_mode else "#FFFFFF"
secondary_bg = "#161B22" if dark_mode else "#F3F6FA"
sidebar_bg = "#111827" if dark_mode else "#F7F9FC"
text_color = "#FAFAFA" if dark_mode else "#111827"
muted_text = "#C9D1D9" if dark_mode else "#4B5563"
border_color = "#30363D" if dark_mode else "#D1D5DB"

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .stApp, .stApp p, .stApp span, .stApp label, .stApp div {{
            color: {text_color};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {sidebar_bg};
            color: {text_color};
            border-right: 1px solid {border_color};
        }}
        section[data-testid="stSidebar"] * {{
            color: {text_color};
        }}
        .stMarkdown, .stText, .stCaption, label, p, span {{
            color: {text_color};
        }}
        div[data-testid="stMetric"] {{
            background-color: {secondary_bg};
            border: 1px solid {border_color};
            border-radius: 12px;
            padding: 0.5rem 0.75rem;
        }}
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] div {{
            color: {text_color};
        }}
        .stDataFrame, .stTable {{
            background-color: {secondary_bg};
            color: {text_color};
            border-radius: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: {secondary_bg};
            color: {text_color};
            border-radius: 8px 8px 0 0;
            border: 1px solid {border_color};
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .stSelectbox > div > div,
        .stNumberInput > div > div > input,
        .stTextInput > div > div > input,
        textarea {{
            background-color: {secondary_bg} !important;
            color: {text_color} !important;
            border: 1px solid {border_color} !important;
        }}
        .stSelectbox [data-baseweb="select"] > div {{
            background-color: {secondary_bg} !important;
            color: {text_color} !important;
            border-color: {border_color} !important;
        }}
        section[data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {{
            background-color: {secondary_bg} !important;
            color: {text_color} !important;
            border: 1px solid {border_color} !important;
            border-radius: 10px !important;
            box-shadow: none !important;
        }}
        section[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {{
            background-color: {"#1F2937" if dark_mode else "#E8EEF7"} !important;
            color: {text_color} !important;
        }}
        hr {{ border-color: {border_color}; }}
        .guide-box {{
            background: {secondary_bg};
            color: {text_color};
            border: 1px solid {border_color};
            padding: 16px;
            border-radius: 8px;
            margin: 8px 0;
        }}
        .verdict-strong, .verdict-moderate, .verdict-weak {{
            color: #FFFFFF !important;
            padding: 12px 16px;
            border-radius: 6px;
            margin: 8px 0;
        }}
        .verdict-strong * , .verdict-moderate * , .verdict-weak * {{
            color: #FFFFFF !important;
        }}
        .verdict-strong {{ background: {"#1a4731" if dark_mode else "#166534"}; border-left: 4px solid #00d4aa; }}
        .verdict-moderate {{ background: {"#3d3010" if dark_mode else "#92400e"}; border-left: 4px solid #ffa500; }}
        .verdict-weak {{ background: {"#3d1010" if dark_mode else "#991b1b"}; border-left: 4px solid #ff4b4b; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Quick Start Presets
# ──────────────────────────────────────────────────────────────────────────────
PRESETS = {
    "🚀 NVDA Momentum": {
        "symbol": "NVDA", "strategy": "Momentum",
        "params": {"window": 20}, "capital": 1200,
        "stop_loss": 7, "take_profit": 15,
        "start": date.today() - timedelta(days=2*365),
    },
    "🛡️ AAPL Safe": {
        "symbol": "AAPL", "strategy": "RSIMeanReversion",
        "params": {"rsi_period": 14, "oversold": 30, "overbought": 70}, "capital": 1200,
        "stop_loss": 5, "take_profit": 10,
        "start": date.today() - timedelta(days=3*365),
    },
    "⚡ TSLA Aggressive": {
        "symbol": "TSLA", "strategy": "MACDStrategy",
        "params": {"fast": 12, "slow": 26, "signal": 9}, "capital": 1200,
        "stop_loss": 8, "take_profit": 20,
        "start": date.today() - timedelta(days=2*365),
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("📈 Mini-Aladdin")
st.sidebar.caption("Your Personal AI Trading System")

# Theme toggle
col_t1, col_t2 = st.sidebar.columns([1, 1])
if col_t1.button("☀️ Light" if dark_mode else "🌙 Dark", use_container_width=True):
    st.session_state["dark_mode"] = not st.session_state["dark_mode"]
    st.rerun()

st.sidebar.divider()

# Quick Start presets
st.sidebar.subheader("⚡ Quick Start")
st.sidebar.caption("New here? Pick a preset to get started instantly:")
preset_cols = st.sidebar.columns(3)
selected_preset = None
for i, (name, cfg) in enumerate(PRESETS.items()):
    if preset_cols[i].button(name.split()[0] + "\n" + name.split()[1], use_container_width=True):
        selected_preset = cfg

# Apply preset if selected
if selected_preset:
    st.session_state["preset"] = selected_preset

preset = st.session_state.get("preset", {})

st.sidebar.divider()
st.sidebar.subheader("⚙️ Settings")

# Stock ticker
symbol = st.sidebar.text_input(
    "Stock Ticker",
    value=preset.get("symbol", "AAPL"),
    help="Enter the stock symbol. Examples: AAPL (Apple), NVDA (Nvidia), TSLA (Tesla), MSFT (Microsoft)"
).upper().strip()

# Date range
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("From", value=preset.get("start", date.today() - timedelta(days=3*365)))
end_date = col2.date_input("To", value=date.today() - timedelta(days=1))

# Strategy
with st.sidebar.expander("🧠 Strategy & Parameters", expanded=True):
    strategy_name = st.selectbox(
        "Strategy",
        list(STRATEGY_REGISTRY.keys()),
        index=list(STRATEGY_REGISTRY.keys()).index(preset.get("strategy", "MovingAverageCrossover")) if preset.get("strategy") in STRATEGY_REGISTRY else 0,
        help="The trading rule the system will follow. Each strategy has a different approach to deciding when to buy and sell."
    )

    strategy_descriptions = {
        "MovingAverageCrossover": "📈 Trend following — buys when short-term trend crosses above long-term trend. Good in bull markets.",
        "RSIMeanReversion": "🔄 Mean reversion — buys when stock is oversold, sells when overbought. Good in sideways markets.",
        "MACDStrategy": "⚡ Momentum — uses MACD indicator to catch trend changes. Versatile all-around strategy.",
        "BollingerBands": "📊 Volatility — buys at lower price band, sells at upper band. Good for range-bound stocks.",
        "Momentum": "🚀 Breakout — buys new highs, sells new lows. Best for strong trending stocks like NVDA.",
        "CombinedSignal": "🎯 Smart Vote — requires 2 of 3 strategies to agree before trading. Higher confidence, fewer trades.",
    }
    st.caption(strategy_descriptions.get(strategy_name, ""))

    param_defs = STRATEGY_REGISTRY[strategy_name]["param_defs"]
    default_params = STRATEGY_REGISTRY[strategy_name]["default_params"]
    preset_params = preset.get("params", {})
    user_params = {}

    param_help = {
        "fast_period": "How many days for the fast moving average. Lower = more sensitive to recent price changes.",
        "slow_period": "How many days for the slow moving average. Higher = smoother, catches bigger trends.",
        "ma_type": "EMA reacts faster to recent prices. SMA is more stable and predictable.",
        "rsi_period": "How many days to calculate RSI. Default 14 is the industry standard.",
        "oversold": "RSI below this = stock is beaten down = BUY signal. 30 is standard.",
        "overbought": "RSI above this = stock is overpriced = SELL signal. 70 is standard.",
        "fast": "MACD fast line period. Default 12 days.",
        "slow": "MACD slow line period. Default 26 days.",
        "signal": "MACD signal smoothing. Default 9 days.",
        "window": "Number of days to look back for high/low breakout detection.",
        "required_votes": "How many strategies need to agree before trading. 2 of 3 = more selective.",
        "std_dev": "How many standard deviations wide the bands are. Higher = fewer signals but more reliable.",
    }

    for param, pdef in param_defs.items():
        default_val = preset_params.get(param, default_params[param])
        if pdef["type"] == "int":
            user_params[param] = st.slider(
                param.replace("_", " ").title(),
                pdef["min"], pdef["max"], default_val,
                help=param_help.get(param, "")
            )
        elif pdef["type"] == "select":
            user_params[param] = st.selectbox(
                param.replace("_", " ").title(),
                pdef["options"],
                index=pdef["options"].index(default_val),
                help=param_help.get(param, "")
            )

with st.sidebar.expander("💰 Capital & Risk", expanded=True):
    capital = st.number_input(
        "Your Capital ($)",
        min_value=100, value=preset.get("capital", 1200), step=100,
        help="How much money you're investing. The backtest will simulate trading with this amount."
    )
    stop_loss = st.slider(
        "Stop Loss % 🛑",
        0, 20, preset.get("stop_loss", 5),
        help="Automatically sell if the trade loses this much. 5% means: if you're down $60 on a $1,200 trade, exit automatically. Protects you from big losses."
    ) / 100
    take_profit = st.slider(
        "Take Profit % 🎯",
        0, 50, preset.get("take_profit", 10),
        help="Automatically sell when the trade gains this much. 10% means: if you're up $120, lock in the profit and exit."
    ) / 100

run_btn = st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Backtest",
    "⚖️ Compare All",
    "⚡ Live Quote",
    "🔬 Validate",
    "📖 Guide",
])

# ── Tab 1: Backtest ──────────────────────────────────────────────────────────
with tab1:
    st.header(f"Backtest — {symbol}")
    st.caption("See how this strategy would have performed on historical data.")

    if run_btn:
        with st.spinner(f"Testing {strategy_name} on {symbol}..."):
            try:
                strategy = build_strategy(strategy_name, **user_params)
                result = run_backtest(
                    symbol, strategy, str(start_date), str(end_date),
                    capital, 0.001, 0.001,
                    stop_loss_pct=stop_loss, take_profit_pct=take_profit
                )
                # Fetch buy & hold benchmark
                try:
                    bh_df = yf.Ticker(symbol).history(start=str(start_date), end=str(end_date), auto_adjust=True)
                    if not bh_df.empty:
                        bh_return = (bh_df["Close"].iloc[-1] / bh_df["Close"].iloc[0] - 1) * 100
                        bh_equity = (bh_df["Close"] / bh_df["Close"].iloc[0]) * capital
                        result._bh_return = round(bh_return, 2)
                        result._bh_equity = bh_equity
                    else:
                        result._bh_return = None
                        result._bh_equity = None
                except Exception:
                    result._bh_return = None
                    result._bh_equity = None

                st.session_state["result"] = result
            except Exception as e:
                st.error(f"Something went wrong: {e}")

    result = st.session_state.get("result")

    if result is None:
        st.info("👈 Choose your settings on the left and click **Run Backtest** to see results.")
        st.image("https://img.icons8.com/nolan/128/combo-chart.png", width=80)
    else:
        m = result.metrics
        bh_return = getattr(result, "_bh_return", None)

        # Verdict banner
        sharpe = m["sharpe_ratio"]
        total_ret = m["total_return"]
        if sharpe >= 1.0 and total_ret > 0:
            st.markdown('<div class="verdict-strong">✅ <strong>Strong Strategy</strong> — Good risk-adjusted returns. This strategy performed well on this stock.</div>', unsafe_allow_html=True)
        elif sharpe >= 0.5 and total_ret > 0:
            st.markdown('<div class="verdict-moderate">⚠️ <strong>Moderate Strategy</strong> — Decent results but not exceptional. Consider tweaking parameters or trying a different strategy.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="verdict-weak">❌ <strong>Weak Strategy</strong> — Poor performance on this stock/period. Try a different strategy or different date range.</div>', unsafe_allow_html=True)

        st.divider()

        # Metrics in plain English
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Total Profit/Loss",
            f"{m['total_return']}%",
            delta=f"{m['total_return'] - bh_return:.1f}% vs buy & hold" if bh_return is not None else None,
            help="Did this strategy make or lose money overall?"
        )
        c2.metric(
            "Yearly Growth Rate",
            f"{m['cagr']}%",
            help="If this rate continued every year, how much would your money grow annually?"
        )
        c3.metric(
            "Risk Score",
            f"{m['sharpe_ratio']}",
            help="Higher is better. Above 1.0 = excellent. 0.5-1.0 = decent. Below 0.5 = risky."
        )
        c4.metric(
            "Worst Loss Period",
            f"{m['max_drawdown']}%",
            help="The biggest drop from peak to bottom during the test. This is what you'd have felt in real trading."
        )

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Win Rate", f"{m['win_rate']}%", help="What % of trades were profitable?")
        c6.metric("Win/Loss Ratio", f"{m['profit_factor']}x", help="How much do winning trades earn compared to what losing trades lose? Above 1.5 is good.")
        c7.metric("Total Trades", m['total_trades'], help="How many times did the strategy buy and sell?")
        c8.metric("Final Value", f"${m['final_value']:,.2f}", help=f"Your ${capital:,.0f} turned into this amount.")

        if bh_return is not None:
            diff = m['total_return'] - bh_return
            color = "🟢" if diff >= 0 else "🔴"
            st.caption(f"{color} Buy & Hold {symbol} for same period: **{bh_return}%** | Strategy {'beat' if diff >= 0 else 'underperformed'} by **{abs(diff):.1f}%**")

        st.divider()

        # Equity curve with buy & hold
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result.equity_curve.index, y=result.equity_curve.values,
            fill="tozeroy", fillcolor="rgba(0,212,170,0.1)",
            line=dict(color="#00d4aa", width=2), name="Your Strategy"
        ))
        if hasattr(result, "_bh_equity") and result._bh_equity is not None:
            fig.add_trace(go.Scatter(
                x=result._bh_equity.index, y=result._bh_equity.values,
                line=dict(color="#aaaaaa", width=1.5, dash="dash"), name="Buy & Hold"
            ))
        fig.add_hline(y=capital, line_dash="dot", line_color="gray", annotation_text="Starting Capital")
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date", yaxis_title="Value ($)",
            template=chart_theme, height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
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
        fig2.update_layout(
            title="Worst Loss Periods (Drawdown)",
            xaxis_title="Date", yaxis_title="Loss from Peak (%)",
            template=chart_theme, height=220
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Price + signals
        df = result.signals_df
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df.index, y=df["Close"], line=dict(color="#aaa", width=1), name="Price"))
        buys = df[df["signal"] == BUY]
        sells = df[df["signal"] == SELL]
        fig3.add_trace(go.Scatter(
            x=buys.index, y=buys["Close"], mode="markers",
            marker=dict(color="lime", size=10, symbol="triangle-up"), name="BUY Signal"
        ))
        fig3.add_trace(go.Scatter(
            x=sells.index, y=sells["Close"], mode="markers",
            marker=dict(color="red", size=10, symbol="triangle-down"), name="SELL Signal"
        ))
        fig3.update_layout(
            title="Price Chart with Buy/Sell Signals",
            xaxis_title="Date", yaxis_title="Price ($)",
            template=chart_theme, height=340
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Trade log
        st.subheader(f"📋 Trade History ({m['total_trades']} trades)")
        if not result.trades.empty:
            def color_pnl(val):
                color = "color: #00d4aa" if val > 0 else "color: #ff4b4b"
                return color
            try:
                st.dataframe(
                    result.trades.style.map(color_pnl, subset=["PnL", "Return %"]),
                    use_container_width=True
                )
            except AttributeError:
                st.dataframe(
                    result.trades.style.applymap(color_pnl, subset=["PnL", "Return %"]),
                    use_container_width=True
                )
        else:
            st.warning("No completed trades in this period. Try a longer date range or different strategy.")

# ── Tab 2: Compare ───────────────────────────────────────────────────────────
with tab2:
    st.header("⚖️ Compare All Strategies")
    st.caption(f"Run all strategies on **{symbol}** for the same period and see which performed best.")

    if st.button("▶️ Run Comparison", type="primary"):
        with st.spinner(f"Testing all strategies on {symbol}..."):
            try:
                results = compare_strategies(symbol, str(start_date), str(end_date), capital)
                st.session_state["compare"] = results
            except Exception as e:
                st.error(f"Error: {e}")

    compare = st.session_state.get("compare")
    if compare:
        # Rename columns for readability
        df_compare = results_to_df(compare)
        df_compare = df_compare.rename(columns={
            "total_return": "Total Return %",
            "cagr": "Yearly Growth %",
            "sharpe_ratio": "Risk Score",
            "max_drawdown": "Worst Loss %",
            "win_rate": "Win Rate %",
            "profit_factor": "Win/Loss Ratio",
            "total_trades": "# Trades",
            "final_value": "Final Value $",
            "initial_capital": "Starting Capital $",
        })
        st.dataframe(df_compare, use_container_width=True)

        fig = go.Figure()
        colors = ["#00d4aa", "#ff9f43", "#a29bfe", "#fd79a8", "#74b9ff", "#55efc4"]
        for i, r in enumerate(compare):
            norm = (r.equity_curve / r.equity_curve.iloc[0] - 1) * 100
            fig.add_trace(go.Scatter(
                x=norm.index, y=norm.values,
                line=dict(color=colors[i % len(colors)], width=2),
                name=r.strategy_name
            ))
        fig.update_layout(
            title=f"All Strategies on {symbol} — Return % Over Time",
            template=chart_theme, height=420,
            yaxis_title="Return (%)", xaxis_title="Date"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Click **Run Comparison** to see all strategies side by side.")

# ── Tab 3: Live Quote ─────────────────────────────────────────────────────────
with tab3:
    st.header("⚡ Live Price")
    st.caption("Get the latest price for any US stock.")

    q = st.text_input("Stock Ticker", value="AAPL", help="Enter any US stock symbol").upper().strip()
    if st.button("Fetch Latest Price", type="primary"):
        try:
            ticker = yf.Ticker(q)
            info = ticker.fast_info
            price = info.last_price
            prev = info.previous_close
            change = price - prev
            change_pct = change / prev * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", f"${price:,.2f}", f"{change_pct:+.2f}% today")
            c2.metric("Change Today", f"${change:+.2f}")
            c3.metric("Volume", f"{int(info.last_volume):,}")

            hist = ticker.history(period="1mo")
            fig = go.Figure(go.Candlestick(
                x=hist.index,
                open=hist["Open"], high=hist["High"],
                low=hist["Low"], close=hist["Close"]
            ))
            fig.update_layout(
                title=f"{q} — Last 30 Days",
                template=chart_theme, height=420,
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not fetch price: {e}")

# ── Tab 4: Validate ───────────────────────────────────────────────────────────
with tab4:
    st.header("🔬 Validate Strategy")
    st.caption("Tests if a strategy is **consistently profitable** across different time periods — not just lucky once.")

    st.info("""
    **Why validate?** A strategy that looks great on 3 years of data might have only worked 
    because it caught one lucky big move (like NVDA's AI boom). Walk-forward testing splits 
    the data into multiple smaller periods and checks if the strategy works across ALL of them.
    """)

    col1, col2 = st.columns(2)
    with col1:
        val_symbol = st.text_input("Symbol", value=symbol, key="val_sym")
        val_strategy_name = st.selectbox("Strategy", list(STRATEGY_REGISTRY.keys()), key="val_strat")
    with col2:
        val_splits = st.slider("Number of test periods", 3, 8, 4,
                               help="More periods = more thorough test. 4-5 is recommended.")
        val_capital = st.number_input("Capital per period ($)", value=1200, min_value=500, key="val_cap")

    val_years = st.slider(
        "Years of history to test", 3, 20, 10,
        help="More years = more reliable result. 10+ years is ideal. yfinance has data back to the 1990s for major stocks."
    )

    if st.button("🔬 Run Validation Test", type="primary"):
        with st.spinner("Running validation... testing across multiple time periods..."):
            try:
                from validator import WalkForwardValidator
                validator = WalkForwardValidator()
                val_strategy = build_strategy(val_strategy_name)
                val_results = validator.run(
                    symbol=val_symbol,
                    strategy=val_strategy,
                    start_date=str(date.today() - timedelta(days=val_years*365)),
                    end_date=str(date.today() - timedelta(days=1)),
                    n_splits=val_splits,
                    initial_capital=val_capital,
                )
                st.session_state["validation"] = val_results
            except Exception as e:
                st.error(f"Validation failed: {e}")

    val = st.session_state.get("validation")
    if val:
        if val.get("error"):
            st.error(val["error"])
        else:
            verdict = val.get("verdict", "UNRELIABLE")
            consistency = val.get("consistency_score", 0)

            if verdict == "ROBUST":
                st.markdown(f'<div class="verdict-strong">✅ <strong>ROBUST</strong> — This strategy is consistently profitable. It worked in {consistency:.0%} of test periods. Suitable for paper trading.</div>', unsafe_allow_html=True)
            elif verdict == "MARGINAL":
                st.markdown(f'<div class="verdict-moderate">⚠️ <strong>MARGINAL</strong> — Mixed results. Worked in {consistency:.0%} of periods. Use with caution and small position sizes.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-weak">❌ <strong>UNRELIABLE</strong> — Only worked in {consistency:.0%} of periods. Do NOT trade this with real money.</div>', unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Return", f"{val.get('avg_return', 0):.1f}%")
            c2.metric("Risk Score", f"{val.get('avg_sharpe', 0):.2f}")
            c3.metric("Avg Worst Loss", f"{val.get('avg_max_drawdown', 0):.1f}%")
            c4.metric("Consistency", f"{consistency:.0%}")

            st.subheader("Results by Period")
            splits_data = []
            for s in val.get("splits", []):
                if s.get("metrics"):
                    result_icon = "✅" if s["metrics"]["total_return"] > 0 else "❌"
                    splits_data.append({
                        "Period": s["split"],
                        "Test Dates": f"{s['test_start']} → {s['test_end']}",
                        "Result": result_icon,
                        "Return %": s["metrics"]["total_return"],
                        "Risk Score": s["metrics"]["sharpe_ratio"],
                        "Win Rate %": s["metrics"]["win_rate"],
                        "Worst Loss %": s["metrics"]["max_drawdown"],
                        "# Trades": s["metrics"]["total_trades"],
                    })
            if splits_data:
                st.dataframe(pd.DataFrame(splits_data), use_container_width=True)
    else:
        st.info("Click **Run Validation Test** to check if this strategy is consistently reliable.")

# ── Tab 5: Guide ──────────────────────────────────────────────────────────────
with tab5:
    st.header("📖 How to Use Mini-Aladdin")

    st.markdown('<div class="guide-box">', unsafe_allow_html=True)
    st.subheader("What is this tool?")
    st.write("""
    Mini-Aladdin is a personal trading research system. It tests trading strategies on real 
    historical stock data so you can see how a strategy would have performed **before** risking 
    real money. Think of it as a flight simulator for trading — practice without crashing.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    st.subheader("🚀 How to Use It (5 Steps)")
    steps = [
        ("1️⃣ Pick a stock", "Type the ticker symbol in the sidebar. AAPL = Apple, NVDA = Nvidia, TSLA = Tesla, MSFT = Microsoft."),
        ("2️⃣ Choose a strategy", "Select from the dropdown. Not sure? Start with **AAPL Safe** preset — it's the most conservative."),
        ("3️⃣ Set your capital", "Enter how much money you'd invest. Use your real capital so the numbers feel real."),
        ("4️⃣ Set stop loss", "Keep it at 5%. This means: if your trade loses 5%, the system automatically exits. This saves you from big losses."),
        ("5️⃣ Click Run Backtest", "See the results. Focus on: Total Return, Win Rate, and Worst Loss Period. A good strategy has positive returns and manageable drawdowns."),
    ]
    for title, desc in steps:
        with st.expander(title):
            st.write(desc)

    st.divider()

    st.subheader("📊 What Do the Numbers Mean?")
    metrics_explained = {
        "Total Return %": "How much money you made (or lost) overall. 10% on $1,200 = $120 profit.",
        "Yearly Growth Rate (CAGR)": "If this growth rate continued every year, how much would you earn annually? 20% CAGR doubles your money in ~3.5 years.",
        "Risk Score (Sharpe Ratio)": "The most important number. Above 1.0 = excellent. 0.5-1.0 = decent. Below 0.5 = too risky for the return. Higher is better.",
        "Worst Loss Period (Max Drawdown)": "The biggest drop you would have experienced. If it shows -40%, at some point your account was down 40% before recovering. Can you stomach that?",
        "Win Rate": "What % of individual trades made money. Don't be fooled by low win rates — a strategy with 30% win rate can still be profitable if the wins are much bigger than the losses.",
        "Win/Loss Ratio (Profit Factor)": "How much the winning trades earn compared to losing trades. 2.0x means your wins are twice as big as your losses. Above 1.5 is good.",
        "Total Trades": "How many times the strategy bought and sold. More trades = more commission costs. Fewer, higher-quality trades are usually better.",
    }
    for metric, explanation in metrics_explained.items():
        with st.expander(f"📌 {metric}"):
            st.write(explanation)

    st.divider()

    st.subheader("🎯 Which Strategy Should I Use?")
    st.markdown("""
    ```
    Is the market going UP overall? (S&P 500 near highs?)
    ├── YES → Use MA Crossover or Momentum
    │          (trend-following strategies work best in bull markets)
    └── NO  → Is the stock bouncing between a range?
               ├── YES → Use RSI Mean Reversion or Bollinger Bands
               └── NO  → Market is crashing → DON'T TRADE
                          (no strategy works well in panic selloffs)
    
    Want fewer but higher-confidence trades?
    └── Use Combined Signal (requires 2/3 strategies to agree)
    
    Want to test a specific stock you know well?
    └── Run Compare All to see which strategy fits it best
    ```
    """)

    st.divider()

    st.subheader("❓ Frequently Asked Questions")
    faqs = [
        ("Is this real money?", "No — backtesting uses historical data. No real money moves. When you're confident, you connect Alpaca paper trading to test with simulated live trades."),
        ("Why does the strategy show losses sometimes?", "Every strategy has losing trades. The goal is for wins to outweigh losses over time. A 40% win rate with big winners can be more profitable than 70% win rate with tiny wins."),
        ("The win rate is only 20-30% — should I be worried?", "Not necessarily. Trend-following strategies like MA Crossover often have low win rates but catch massive moves. Check the Win/Loss Ratio — if wins are 3-4x bigger than losses, low win rate is fine."),
        ("What's a good Total Return?", "Beating buy-and-hold (just buying and holding the stock) is the benchmark. If AAPL returned 50% and your strategy returned 40%, your strategy actually underperformed. Always compare to buy & hold."),
        ("Can I trust these backtest results for real trading?", "Be cautious. Backtests always look better than real trading because: fills are cleaner, no gaps, no slippage surprises. Use the Validate tab to stress-test the strategy across multiple periods before trusting it."),
        ("What's the -48% drawdown on NVDA?", "At some point during the backtest, the strategy's portfolio dropped 48% from its peak before recovering. In real trading, most people would have panic-sold at the bottom. Drawdown shows you the emotional test you'd face."),
        ("How much money do I need to start?", "Alpaca allows fractional shares so technically any amount. But with less than $1,000 it's hard to diversify. $1,200 (Pradeep's budget) is workable — keep positions small, max 50% in one stock."),
        ("When should I NOT trade?", "1) Around earnings announcements (stock can gap 10-20% overnight). 2) When the overall market is crashing. 3) When a trade immediately goes against you (trust your stop loss). 4) When you're emotional."),
        ("What's the difference between paper trading and real trading?", "Paper trading = simulated trades with fake money but real market prices. Use it to validate the system for 2-4 weeks before going live. If paper results match backtests, you can consider real money."),
        ("How do I know when to go live?", "When: 1) Strategy shows ROBUST on the Validate tab 2) Paper trading results match backtests for 3-4 weeks 3) You understand WHY the strategy works 4) You can handle the max drawdown emotionally."),
    ]
    for q, a in faqs:
        with st.expander(f"❓ {q}"):
            st.write(a)

    st.divider()

    st.subheader("⚠️ Risk Warning")
    st.warning("""
    **Read this before trading real money:**
    
    - Past performance does NOT guarantee future results. A strategy that worked for 3 years can stop working.
    - Backtests always look better than real trading. Real fills, gaps, and emotions make it harder.
    - Never risk money you cannot afford to lose. Start with your smallest comfortable amount.
    - The stop loss feature helps, but gaps (when a stock jumps overnight) can bypass stop losses.
    - This tool is for research and education. It is NOT financial advice.
    - When in doubt, paper trade first. Always.
    """)
