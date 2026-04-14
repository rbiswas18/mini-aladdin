"""
app.py — Trading Alpha Dashboard
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
try:
    from strategy_pro import PRO_STRATEGY_REGISTRY, build_pro_strategy
    COMBINED_REGISTRY = {**STRATEGY_REGISTRY, **PRO_STRATEGY_REGISTRY}
except ImportError:
    PRO_STRATEGY_REGISTRY = {}
    COMBINED_REGISTRY = STRATEGY_REGISTRY
    build_pro_strategy = None

try:
    from signal_hub import SignalHub
    SIGNAL_HUB_AVAILABLE = True
except ImportError:
    SIGNAL_HUB_AVAILABLE = False

# ── Stock search utility ──────────────────────────────────────────────────────
POPULAR_STOCKS = {
    # Tech
    "Apple": "AAPL", "Microsoft": "MSFT", "NVIDIA": "NVDA", "Alphabet": "GOOGL",
    "Meta": "META", "Amazon": "AMZN", "Tesla": "TSLA", "Netflix": "NFLX",
    "Adobe": "ADBE", "Salesforce": "CRM", "Intel": "INTC", "AMD": "AMD",
    "Qualcomm": "QCOM", "Broadcom": "AVGO", "Texas Instruments": "TXN",
    "Palantir": "PLTR", "Snowflake": "SNOW", "Datadog": "DDOG", "CrowdStrike": "CRWD",
    # Finance
    "JPMorgan Chase": "JPM", "Goldman Sachs": "GS", "Morgan Stanley": "MS",
    "Bank of America": "BAC", "Wells Fargo": "WFC", "Visa": "V", "Mastercard": "MA",
    "PayPal": "PYPL", "Square/Block": "SQ", "Berkshire Hathaway": "BRK-B",
    # Healthcare
    "Johnson & Johnson": "JNJ", "Pfizer": "PFE", "UnitedHealth": "UNH",
    "Abbott Labs": "ABT", "Eli Lilly": "LLY", "Moderna": "MRNA", "Novo Nordisk": "NVO",
    # Consumer
    "Walmart": "WMT", "Costco": "COST", "Home Depot": "HD", "Nike": "NKE",
    "McDonald's": "MCD", "Starbucks": "SBUX", "Coca-Cola": "KO", "PepsiCo": "PEP",
    # Energy
    "ExxonMobil": "XOM", "Chevron": "CVX", "ConocoPhillips": "COP",
    # ETFs
    "S&P 500 ETF": "SPY", "Nasdaq ETF": "QQQ", "Dow Jones ETF": "DIA",
    "Small Cap ETF": "IWM", "Tech ETF": "XLK", "Finance ETF": "XLF",
}

# Reverse lookup: ticker -> name
TICKER_TO_NAME = {v: k for k, v in POPULAR_STOCKS.items()}


def stock_searchbox(label: str, key: str, default_ticker: str = "AAPL") -> str:
    """
    Renders a stock selector that accepts EITHER a company name OR a ticker.
    Returns the ticker symbol.

    Uses a selectbox with all popular stocks + a text input for custom tickers.
    """
    options = [f"{name} ({ticker})" for name, ticker in sorted(POPULAR_STOCKS.items())]
    options.insert(0, "🔍 Enter custom ticker...")

    default_name = TICKER_TO_NAME.get(default_ticker.upper(), "")
    default_option = f"{default_name} ({default_ticker.upper()})" if default_name else options[0]
    default_idx = options.index(default_option) if default_option in options else 0

    selected = st.selectbox(label, options, index=default_idx, key=f"{key}_select")

    if selected == "🔍 Enter custom ticker...":
        custom = st.text_input("Enter ticker symbol", value=default_ticker, key=f"{key}_custom")
        return custom.upper().strip()
    else:
        ticker = selected.split("(")[-1].rstrip(")")
        return ticker.strip()

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trading Alpha System",
    page_icon="📈",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# Theme (Dark Mode Only)
# ──────────────────────────────────────────────────────────────────────────────
dark_mode = True
chart_theme = "plotly_dark"
bg_color = "#0E1117"
secondary_bg = "#161B22"
sidebar_bg = "#111827"
text_color = "#FAFAFA"
muted_text = "#C9D1D9"
border_color = "#30363D"

st.markdown("""
<style>
    /* ===== Global app surfaces ===== */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    .main, section.main, .block-container {
        background: #0E1117 !important;
        color: #FAFAFA !important;
    }
    /* ===== Sidebar ===== */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"] {
        background: #111827 !important;
        color: #FAFAFA !important;
    }
    [data-testid="stSidebar"] *, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div {
        color: #FAFAFA !important;
    }
    /* ===== All inputs ===== */
    [data-testid="stDateInput"] input, [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input, [data-baseweb="input"] input,
    [data-baseweb="base-input"] input, [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        background: #161B22 !important;
        color: #FAFAFA !important;
        -webkit-text-fill-color: #FAFAFA !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }
    [data-testid="stDateInput"] [data-baseweb="input"],
    [data-testid="stNumberInput"] [data-baseweb="input"],
    [data-testid="stTextInput"] [data-baseweb="input"],
    [data-testid="stSidebar"] [data-baseweb="input"],
    [data-testid="stSidebar"] [data-baseweb="base-input"] {
        background: #161B22 !important;
        border: 1px solid #30363D !important;
    }
    [data-testid="stDateInput"] svg, [data-testid="stSidebar"] svg {
        fill: #FAFAFA !important;
    }
    /* ===== Buttons ===== */
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] [data-testid="baseButton-secondary"] {
        background: #161B22 !important;
        color: #FAFAFA !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] button:hover {
        filter: brightness(0.95) !important;
    }
    button[kind="primary"], .stButton > button[kind="primary"] { color: white !important; }
    /* ===== Selectboxes ===== */
    [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #161B22 !important;
        color: #FAFAFA !important;
        border: 1px solid #30363D !important;
    }
    [data-baseweb="select"] * { color: #FAFAFA !important; }
    /* ===== Metrics ===== */
    div[data-testid="stMetric"] {
        background: #161B22 !important;
        border: 1px solid #30363D !important;
        border-radius: 12px !important;
        padding: 0.5rem 0.75rem !important;
    }
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] div { color: #FAFAFA !important; }
    /* ===== Tabs ===== */
    .stTabs [data-baseweb="tab"] {
        background: #161B22 !important;
        color: #FAFAFA !important;
        border-radius: 8px 8px 0 0 !important;
        border: 1px solid #30363D !important;
    }
    .stTabs [aria-selected="true"] { background: #0E1117 !important; }
    /* ===== DataFrames ===== */
    [data-testid="stDataFrame"], [data-testid="stDataFrame"] > div {
        background: #161B22 !important;
        color: #FAFAFA !important;
        border-radius: 10px !important;
    }
    [data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td,
    .stTable th, .stTable td {
        background: #161B22 !important;
        color: #FAFAFA !important;
        border-color: #30363D !important;
    }
    /* ===== Info boxes ===== */
    [data-testid="stInfo"], .stAlert, div[data-baseweb="notification"] {
        background: #161B22 !important;
        color: #FAFAFA !important;
        border: 1px solid #30363D !important;
        border-radius: 10px !important;
    }
    [data-testid="stInfo"] *, .stAlert * { color: #FAFAFA !important; }
    /* ===== Code blocks ===== */
    pre, code, .stCodeBlock, [data-testid="stMarkdownContainer"] pre,
    [data-testid="stMarkdownContainer"] code {
        background: #161B22 !important;
        color: #FAFAFA !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
    }
    /* ===== Expanders ===== */
    [data-testid="stExpander"] details, [data-testid="stExpander"] summary {
        background: #161B22 !important;
        color: #FAFAFA !important;
        border-color: #30363D !important;
    }
    /* ===== Markdown text ===== */
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] strong { color: #FAFAFA !important; }
    /* ===== Plotly containers ===== */
    [data-testid="stPlotlyChart"], [data-testid="stPlotlyChart"] > div,
    [data-testid="stPlotlyChart"] .main-svg { background: transparent !important; }
    /* ===== Hide Streamlit toolbar & branding completely ===== */
    header[data-testid="stHeader"] {
        height: 0px !important;
        min-height: 0px !important;
        visibility: hidden !important;
    }
    #MainMenu, footer, [data-testid="stToolbar"],
    [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
        display: none !important;
        visibility: hidden !important;
    }
    /* ===== Fix tooltip dots ===== */
    [data-testid="stTooltipHoverTarget"] button,
    [data-testid="tooltipHoverTarget"] button,
    button[aria-label="Learn more"],
    [data-baseweb="tooltip"] button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stTooltipHoverTarget"] svg,
    [data-testid="tooltipHoverTarget"] svg {
        fill: #FAFAFA !important;
        color: #FAFAFA !important;
    }
    /* ===== Tooltip icons ===== */
    [data-testid="stTooltipIcon"], [data-testid="stTooltipIcon"] svg,
    button[data-testid="stTooltipIcon"], .stTooltipIcon svg,
    [data-testid="stSidebar"] [data-testid="stTooltipIcon"] {
        color: #FAFAFA !important;
        fill: #FAFAFA !important;
        stroke: #FAFAFA !important;
    }
    /* ===== HR ===== */
    hr { border-color: #30363D; }
    /* ===== Verdict banners ===== */
    .verdict-strong, .verdict-moderate, .verdict-weak {
        color: #FFFFFF !important; padding: 12px 16px; border-radius: 6px; margin: 8px 0;
    }
    .verdict-strong *, .verdict-moderate *, .verdict-weak * { color: #FFFFFF !important; }
    .verdict-strong { background: #1a4731; border-left: 4px solid #00d4aa; }
    .verdict-moderate { background: #3d3010; border-left: 4px solid #ffa500; }
    .verdict-weak { background: #3d1010; border-left: 4px solid #ff4b4b; }
    /* ===== Guide box ===== */
    .guide-box { background: #161B22; color: #FAFAFA; border: 1px solid #30363D; padding: 16px; border-radius: 8px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

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
# ── Top banner ──
st.markdown("""
<div style="
    background: linear-gradient(90deg, #00d4aa 0%, #0066ff 100%);
    padding: 18px 28px;
    border-radius: 12px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
">
    <div>
        <h1 style="color: white; margin: 0; font-size: 28px; font-weight: 800; letter-spacing: -0.5px;">📈 Trading Alpha</h1>
        <p style="color: rgba(255,255,255,0.85); margin: 4px 0 0 0; font-size: 14px;">AI-Powered Personal Trading System</p>
    </div>
    <div style="color: rgba(255,255,255,0.7); font-size: 13px; text-align: right;">
        🌙 Dark Mode
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("📈 Trading Alpha")
st.sidebar.caption("Your Personal AI Trading System")
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
with st.sidebar:
    symbol = stock_searchbox("Stock", "sidebar_symbol", default_ticker=preset.get("symbol", "AAPL"))

# Date range
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("From", value=preset.get("start", date.today() - timedelta(days=3*365)))
end_date = col2.date_input("To", value=date.today() - timedelta(days=1))

# Strategy
with st.sidebar.expander("🧠 Strategy & Parameters", expanded=True):
    strategy_name = st.selectbox(
        "Strategy",
        list(COMBINED_REGISTRY.keys()),
        index=list(COMBINED_REGISTRY.keys()).index(preset.get("strategy", "MovingAverageCrossover")) if preset.get("strategy") in STRATEGY_REGISTRY else 0,
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

    param_defs = COMBINED_REGISTRY.get(strategy_name, {}).get("param_defs", {})
    default_params = COMBINED_REGISTRY.get(strategy_name, {}).get("default_params", {})
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Backtest",
    "⚖️ Compare All",
    "⚡ Live Quote",
    "🔬 Validate",
    "📖 Guide",
    "📡 Signal Hub",
    "🧠 Learn & Optimize",
    "🤖 Portfolio Engine",
])

# ── Tab 1: Backtest ──────────────────────────────────────────────────────────
with tab1:
    st.header(f"Backtest — {symbol}")
    st.caption("See how this strategy would have performed on historical data.")

    if run_btn:
        with st.spinner(f"Testing {strategy_name} on {symbol}..."):
            try:
                if strategy_name in PRO_STRATEGY_REGISTRY and build_pro_strategy:
                        strategy = build_pro_strategy(strategy_name, **user_params)
                else:
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

    q = stock_searchbox("Search Stock", "quote_symbol", "AAPL")
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
        val_symbol = stock_searchbox("Symbol to validate", "val_symbol_search", symbol)
        val_strategy_name = st.selectbox("Strategy", list(COMBINED_REGISTRY.keys()), key="val_strat")
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
    st.header("📖 How to Use Trading Alpha")

    st.markdown('<div class="guide-box">', unsafe_allow_html=True)
    st.subheader("What is this tool?")
    st.write("""
    Trading Alpha is a personal trading research system. It tests trading strategies on real 
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

    st.subheader("🏆 How to Get the Best Results")
    st.markdown("""
    Follow this workflow every time you want to trade:

    **Step 1 — 📡 Signal Hub first**
    Run the Signal Hub scan on your watchlist. Only consider stocks with NSS ≥ 0.30.
    If everything is HOLD, the market is saying "wait."

    **Step 2 — 🔬 Validate the strategy**
    Go to the Validate tab. Run walk-forward validation on your chosen stock + strategy.
    Only trade if the verdict is ROBUST or MARGINAL. Never trade UNRELIABLE strategies.

    **Step 3 — 📊 Backtest with realistic settings**
    Set stop loss to 5-7%, take profit to 10-15%, capital to your real amount.
    Check the equity curve — if the drawdown is more than you can stomach, don't trade it.

    **Step 4 — 🧠 Optimize parameters**
    Run the Learn & Optimize tab to find the best parameters for your stock.
    Use the optimized params in your backtest before going live.

    **Step 5 — 🤖 Paper trade first**
    Use the Portfolio Engine tab in Dry Run mode for at least 2-4 weeks.
    Only go live when paper results match backtest expectations.

    **The golden rule:** If Signal Hub + Validate + Backtest all agree → HIGH CONFIDENCE.
    If any one of them says no → WAIT.
    """
    )

    st.divider()

    st.subheader("📖 Understanding NSS Scores")
    col1, col2, col3 = st.columns(3)
    col1.metric("NSS ≥ 0.50", "STRONG BUY", "High confidence")
    col2.metric("NSS 0.30–0.50", "BUY", "Moderate confidence")
    col3.metric("NSS < 0.30", "HOLD", "Not enough agreement")
    st.caption("NSS = Normalized Signal Score. Combines 9 technical strategies + 5 fundamental providers. Range: -1.0 to +1.0")

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

# ── Tab 6: Signal Hub ────────────────────────────────────────────────────────
with tab6:
    st.header("📡 Signal Hub")
    st.caption("Multi-signal aggregation — only trade when technicals + fundamentals + sentiment all agree")

    if not SIGNAL_HUB_AVAILABLE:
        st.error("Signal Hub not available. Make sure signal_hub.py is in the same folder.")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            watchlist_input = st.text_input(
                "Watchlist (comma-separated tickers)",
                value="AAPL, NVDA, TSLA, MSFT, AMZN",
                help="Enter the stocks you want to scan"
            )
            st.caption("Enter ticker symbols (e.g. AAPL, NVDA) or use the search above to find tickers")
        with col2:
            min_nss = st.slider(
                "Min NSS to trade", 0.1, 0.8, 0.3, 0.05,
                help="Minimum signal strength to consider a trade. 0.30 = 60% of weighted signals agree"
            )

        use_providers = st.checkbox(
            "Include Fundamentals + Sentiment + Valuation", value=True,
            help="Adds fundamental analysis, news sentiment, and valuation checks. Slower but more accurate."
        )

        scan_btn = st.button("🔍 Run Full Signal Scan", type="primary", use_container_width=True)

        if scan_btn:
            symbols = [s.strip().upper() for s in watchlist_input.split(",") if s.strip()]
            with st.spinner(f"Scanning {len(symbols)} stocks... (30-60 seconds)"):
                try:
                    hub = SignalHub(symbols=symbols, use_providers=use_providers)
                    results = hub.scan()
                    st.session_state["hub_results"] = results
                    st.session_state["hub_symbols"] = symbols
                    st.session_state["hub_min_nss"] = min_nss
                    st.session_state["hub_use_providers"] = use_providers
                except Exception as e:
                    st.error(f"Scan failed: {e}")

        results = st.session_state.get("hub_results")
        if results:
            symbols = st.session_state.get("hub_symbols", [])
            min_nss_val = st.session_state.get("hub_min_nss", 0.3)
            use_providers = st.session_state.get("hub_use_providers", use_providers)

            candidates = [
                (sym, data) for sym, data in results.items()
                if abs(data.get("nss", 0)) >= min_nss_val and data.get("should_trade", False)
            ]

            if candidates:
                st.success(f"✅ {len(candidates)} trade candidate(s) found!")
                for sym, data in sorted(candidates, key=lambda x: abs(x[1].get("nss", 0)), reverse=True):
                    nss = data.get("nss", 0)
                    rec = data.get("recommendation", "HOLD")
                    icon = "🟢" if nss > 0 else "🔴"
                    st.markdown(f"### {icon} {sym} — {rec} (NSS: {nss:+.3f})")

                    breakdown = data.get("aggregation", {}).get("breakdown", [])
                    if breakdown:
                        df_breakdown = pd.DataFrame([
                            {
                                "Signal": s.get("name", ""),
                                "Vote": "🟢 BUY" if s.get("signal", 0) == 1 else "🔴 SELL" if s.get("signal", 0) == -1 else "⚪ HOLD",
                                "Weight": s.get("weight", 0),
                                "Detail": s.get("summary", "")[:60]
                            }
                            for s in breakdown
                        ])
                        st.dataframe(df_breakdown, use_container_width=True)
            else:
                st.info(f"No trade candidates found with NSS ≥ {min_nss_val:.2f}. Market conditions not aligned.")

            st.divider()

            st.subheader("📊 All Stocks — Signal Summary")
            summary_rows = []
            for sym, data in results.items():
                nss = data.get("nss", 0)
                agg = data.get("aggregation", {})
                summary_rows.append({
                    "Symbol": sym,
                    "NSS": round(nss, 3),
                    "Signal": data.get("recommendation", "HOLD"),
                    "Bullish": agg.get("bullish_count", 0),
                    "Bearish": agg.get("bearish_count", 0),
                    "Neutral": agg.get("neutral_count", 0),
                    "Regime": data.get("regime", "UNKNOWN"),
                    "Trade?": "✅" if data.get("should_trade", False) else "❌",
                })

            df_summary = pd.DataFrame(summary_rows)
            st.dataframe(df_summary, use_container_width=True)

            fig = go.Figure()
            colors = ["#00d4aa" if r["NSS"] > 0 else "#ff4b4b" for r in summary_rows]
            fig.add_trace(go.Bar(
                x=[r["Symbol"] for r in summary_rows],
                y=[r["NSS"] for r in summary_rows],
                marker_color=colors,
                text=[r["Signal"] for r in summary_rows],
                textposition="outside",
            ))
            fig.add_hline(y=min_nss_val, line_dash="dash", line_color="lime", annotation_text="Buy threshold")
            fig.add_hline(y=-min_nss_val, line_dash="dash", line_color="red", annotation_text="Sell threshold")
            fig.update_layout(
                title="Normalized Signal Score (NSS) by Stock",
                yaxis_title="NSS (-1 to +1)",
                template="plotly_dark",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Full Text Report"):
                try:
                    hub = SignalHub(symbols=list(results.keys()), use_providers=use_providers)
                    report = hub.format_report(results)
                    st.text(report)
                except Exception:
                    st.text("Report generation failed.")
        else:
            st.info("Click **Run Full Signal Scan** to analyze your watchlist.")


# ── Tab 7: Learn & Optimize ──────────────────────────────────────────────────
with tab7:
    st.header("🧠 Learn & Optimize")
    st.caption("The system analyzes your backtests, finds mistakes, and discovers better parameters")

    col1, col2 = st.columns(2)
    with col1:
        learn_symbol = stock_searchbox("Symbol", "learn_symbol_search", "TSLA")
        learn_strategy = st.selectbox("Strategy", list(COMBINED_REGISTRY.keys()), key="learn_strat")
    with col2:
        learn_years = st.slider("Years of history", 3, 15, 9, key="learn_years")

    col3, col4 = st.columns(2)
    run_analysis = col3.button("🔍 Run Full Analysis", type="primary", use_container_width=True)
    run_optimize = col4.button("⚡ Optimize Parameters", use_container_width=True)

    try:
        from adaptive_learner import AdaptiveLearner, LearningJournal
        ADAPTIVE_LEARNER_AVAILABLE = True
    except ImportError:
        ADAPTIVE_LEARNER_AVAILABLE = False
        AdaptiveLearner = None
        LearningJournal = None

    if not ADAPTIVE_LEARNER_AVAILABLE:
        st.error("Learn & Optimize is not available. Make sure adaptive_learner.py is in the same folder.")
    else:
        if run_analysis:
            with st.spinner(f"Analyzing {learn_strategy} on {learn_symbol}..."):
                try:
                    learner = AdaptiveLearner(symbols=[learn_symbol], state_file="learning_journal.json")
                    learn_result = learner.run_full_analysis(learn_symbol, learn_strategy, years_back=learn_years)
                    st.session_state["learn_result"] = learn_result
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

        if run_optimize:
            with st.spinner(f"Optimizing {learn_strategy} on {learn_symbol}..."):
                try:
                    learner = AdaptiveLearner(symbols=[learn_symbol], state_file="learning_journal.json")
                    optimize_result = learner.optimize_and_learn(learn_symbol, learn_strategy, years_back=learn_years)
                    st.session_state["optimize_result"] = optimize_result
                except Exception as e:
                    st.error(f"Optimization failed: {e}")

        learn_result = st.session_state.get("learn_result")
        if learn_result:
            score_data = learn_result.get("strategy_score", {})
            total_score = score_data.get("total_score", learn_result.get("metrics", {}).get("strategy_score", 0))
            grade = score_data.get("grade", learn_result.get("metrics", {}).get("grade", "N/A"))
            grade_colors = {
                "A": "#00d4aa",
                "B": "#4ade80",
                "C": "#facc15",
                "D": "#fb923c",
                "F": "#ff4b4b",
            }
            grade_color = grade_colors.get(str(grade).upper(), "#C9D1D9")

            score_col1, score_col2, score_col3 = st.columns([1, 1, 2])
            score_col1.metric("Strategy Score", f"{total_score}/100")
            score_col2.markdown(
                f"""
                <div style="background:{grade_color};padding:0.85rem 1rem;border-radius:12px;text-align:center;font-weight:800;color:#0E1117;">
                    Grade: {grade}
                </div>
                """,
                unsafe_allow_html=True,
            )
            recommendation = score_data.get("recommendation")
            if recommendation:
                score_col3.info(recommendation)

            failure_patterns = learn_result.get("failure_patterns", [])
            if failure_patterns:
                st.subheader("⚠️ Failure Patterns")
                for pattern in failure_patterns:
                    st.warning(pattern)

            lessons = learn_result.get("improvement_suggestions", [])
            if lessons:
                st.subheader("💡 Lessons Learned")
                for lesson in lessons:
                    st.markdown(f"- {lesson}")

        optimize_result = st.session_state.get("optimize_result")
        if optimize_result:
            st.subheader("⚙️ Best Parameters Found")
            best_params = optimize_result.get("best_params", {})
            if best_params:
                st.dataframe(pd.DataFrame([
                    {"Parameter": key, "Value": value} for key, value in best_params.items()
                ]), use_container_width=True)
            else:
                st.info("No optimized parameters were returned.")

            top_3 = optimize_result.get("top_3", [])
            if top_3:
                st.subheader("🏆 Top 3 Parameter Sets")
                comparison_rows = []
                for idx, item in enumerate(top_3, start=1):
                    row = {"Rank": idx}
                    if isinstance(item, dict):
                        params = item.get("params", {})
                        if isinstance(params, dict):
                            row.update(params)
                        for key, value in item.items():
                            if key != "params":
                                row[key] = value
                    else:
                        row["Result"] = item
                    comparison_rows.append(row)
                st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

            overfit_warning = optimize_result.get("overfit_warning")
            if overfit_warning:
                st.warning(overfit_warning)

        st.divider()
        st.subheader("📚 Learning Journal")
        try:
            journal = LearningJournal("learning_journal.json")
            best_backtests = journal.get_best_performing(5)
            if best_backtests:
                journal_rows = []
                for item in best_backtests:
                    metrics = item.get("metrics", {})
                    journal_rows.append({
                        "Symbol": item.get("symbol", ""),
                        "Strategy": item.get("strategy_name", ""),
                        "Score": metrics.get("strategy_score", 0),
                        "Grade": metrics.get("grade", ""),
                        "Return %": metrics.get("total_return", 0),
                        "Sharpe": metrics.get("sharpe_ratio", 0),
                    })
                st.dataframe(pd.DataFrame(journal_rows), use_container_width=True)
            else:
                st.info("No journal entries yet. Run an analysis or optimization to start learning.")

            journal_lessons = journal.get_lessons_learned()
            if journal_lessons:
                st.markdown("**Latest lessons learned**")
                for lesson in journal_lessons[-10:]:
                    st.markdown(f"- {lesson}")

            with st.expander("📝 Full Journal Report"):
                st.text(journal.generate_report())
        except Exception as e:
            st.warning(f"Could not load learning journal: {e}")

# ── Tab 8: Portfolio Engine ──────────────────────────────────────────────────
with tab8:
    st.header("🤖 Portfolio Engine")
    st.caption("Run the live trading engine, scans your watchlist and manages positions")

    col1, col2 = st.columns(2)
    with col1:
        pe_symbols_input = st.text_input("Watchlist", value="AAPL, NVDA, TSLA, MSFT", key="pe_symbols")
        st.caption("Enter ticker symbols (e.g. AAPL, NVDA) or use the search above to find tickers")
        strategy_options = list(COMBINED_REGISTRY.keys())
        pe_default_index = strategy_options.index("CombinedSignal") if "CombinedSignal" in COMBINED_REGISTRY else 0
        pe_strategy = st.selectbox("Strategy", strategy_options, key="pe_strategy", index=pe_default_index)
    with col2:
        pe_capital = st.number_input("Account Capital ($)", value=1200, min_value=100, key="pe_capital")
        pe_dry_run = st.checkbox("Dry Run (no real orders)", value=True, help="Safe mode, logs what it would do but places no orders")

    col3, col4, col5 = st.columns(3)
    run_scan_btn = col3.button("🔍 Run Scan", type="primary", use_container_width=True)
    check_stops_btn = col4.button("🛑 Check Stops", use_container_width=True)
    get_status_btn = col5.button("📊 Get Status", use_container_width=True)

    try:
        from portfolio_engine import PortfolioEngine
        PORTFOLIO_ENGINE_AVAILABLE = True
    except ImportError:
        PORTFOLIO_ENGINE_AVAILABLE = False
        PortfolioEngine = None

    def _build_portfolio_engine(symbols, strategy_name, account_value):
        return PortfolioEngine(symbols=symbols, strategy_name=strategy_name, account_value=float(account_value), paper_mode=True)

    if not PORTFOLIO_ENGINE_AVAILABLE:
        st.error("Portfolio Engine is not available. Make sure portfolio_engine.py is in the same folder.")
    else:
        parsed_symbols = [s.strip().upper() for s in pe_symbols_input.split(",") if s.strip()]

        if run_scan_btn:
            with st.spinner(f"Scanning {len(parsed_symbols)} symbols..."):
                try:
                    engine = _build_portfolio_engine(parsed_symbols, pe_strategy, pe_capital)
                    execution_log = engine.execute_scan(dry_run=pe_dry_run)
                    st.session_state["pe_last_log"] = execution_log
                    st.session_state["pe_status"] = engine.get_status()
                except Exception as e:
                    st.error(f"Portfolio scan failed: {e}")

        if check_stops_btn:
            with st.spinner("Checking stops..."):
                try:
                    engine = _build_portfolio_engine(parsed_symbols, pe_strategy, pe_capital)
                    closed_positions = engine.check_stops(dry_run=pe_dry_run)
                    st.session_state["pe_closed_positions"] = closed_positions
                    st.session_state["pe_status"] = engine.get_status()
                except Exception as e:
                    st.error(f"Stop check failed: {e}")

        if get_status_btn:
            try:
                engine = _build_portfolio_engine(parsed_symbols, pe_strategy, pe_capital)
                st.session_state["pe_status"] = engine.get_status()
            except Exception as e:
                st.error(f"Could not fetch status: {e}")

        execution_log = st.session_state.get("pe_last_log")
        if execution_log is not None:
            st.subheader("🧾 Execution Log")
            if execution_log:
                st.dataframe(pd.DataFrame(execution_log), use_container_width=True)
            else:
                st.info("No actions were taken in the latest scan.")

        closed_positions = st.session_state.get("pe_closed_positions")
        if closed_positions is not None:
            st.subheader("🛑 Closed Positions")
            if closed_positions:
                st.dataframe(pd.DataFrame(closed_positions), use_container_width=True)
            else:
                st.info("No stop loss or take profit triggers were hit.")

        status = st.session_state.get("pe_status")
        if status:
            st.subheader("📊 Current Portfolio Status")
            trade_journal_summary = status.get("trade_journal", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Open Positions", status.get("open_count", 0))
            c2.metric("Total PnL", f"${trade_journal_summary.get('total_pnl', 0):,.2f}")
            c3.metric("Win Rate", f"{trade_journal_summary.get('win_rate', 0)}%")

            risk_status = status.get("risk_status", {})
            if risk_status:
                st.markdown("**Risk Status**")
                risk_rows = [{"Metric": key.replace("_", " ").title(), "Value": value} for key, value in risk_status.items()]
                st.dataframe(pd.DataFrame(risk_rows), use_container_width=True)

            open_positions = status.get("open_positions", {})
            if open_positions:
                st.markdown("**Open Positions**")
                open_rows = []
                for sym, pos in open_positions.items():
                    row = {"Symbol": sym}
                    if isinstance(pos, dict):
                        row.update(pos)
                    else:
                        row["Details"] = pos
                    open_rows.append(row)
                st.dataframe(pd.DataFrame(open_rows), use_container_width=True)

            if trade_journal_summary:
                st.markdown("**Trade Journal Summary**")
                summary_rows = [{"Metric": key.replace("_", " ").title(), "Value": value} for key, value in trade_journal_summary.items()]
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

