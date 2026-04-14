# 📈 Mini-Aladdin — AI Trading System

A personal, AI-driven trading and portfolio management system inspired by institutional platforms. Built for systematic backtesting, strategy discovery, and data-driven decision making — 100% free to run.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up environment (optional)
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY for AI analysis features
```

### 3. Launch the dashboard
```bash
streamlit run dashboard.py
```

Open your browser to `http://localhost:8501`

---

## 📁 Project Structure

```
mini-aladdin/
├── data/
│   └── cache/          # Cached Parquet files (auto-created)
├── strategies/
│   └── __init__.py
├── data_fetch.py       # Market data layer (yfinance + Polygon stub)
├── strategy.py         # Trading strategies (MA, RSI, MACD)
├── backtest.py         # VectorBT-powered backtesting engine
├── dashboard.py        # Streamlit trading board
├── agent.py            # AI strategy copilot (OpenAI)
├── .env.example        # Environment variable template
├── requirements.txt
└── README.md
```

---

## 🧩 Module Overview

### `data_fetch.py` — Data Layer
Fetches US stock market data via yfinance (free) with local Parquet caching.
- `YFinanceProvider` — pulls OHLCV bars and live quotes
- `PolygonProvider` — stub for future Polygon.io upgrade
- `get_provider("yfinance")` — factory function

```python
from data_fetch import get_provider
provider = get_provider("yfinance")
df = provider.get_bars("AAPL", "1d", "2022-01-01", "2024-01-01")
quote = provider.get_latest_quote("AAPL")
```

### `strategy.py` — Strategy Layer
Three plug-and-play strategies, each generating BUY/SELL/HOLD signals.

| Strategy | Description |
|---|---|
| `MovingAverageCrossover` | EMA/SMA fast/slow crossover |
| `RSIMeanReversion` | Buy oversold, sell overbought |
| `MACDStrategy` | MACD/signal line crossover |

```python
from strategy import build_strategy
strategy = build_strategy("MovingAverageCrossover", fast_period=10, slow_period=50)
signals_df = strategy.generate_signals(df)
```

### `backtest.py` — Backtesting Engine
VectorBT-powered engine with realistic commission + slippage modeling.

```python
from backtest import BacktestEngine
engine = BacktestEngine()
result = engine.run("AAPL", strategy, "2022-01-01", "2024-01-01", initial_capital=10000)
print(result.metrics)  # total_return, sharpe_ratio, max_drawdown, etc.

# Compare all strategies
results = engine.compare_strategies("SPY", "2022-01-01", "2024-01-01")
```

### `dashboard.py` — Trading Board
4-tab Streamlit dashboard:
- **Backtest** — Run and visualize single strategy backtests
- **Compare Strategies** — Side-by-side performance comparison
- **Live Quote** — Real-time price + candlestick chart
- **AI Builder** — Natural language → strategy config

### `agent.py` — AI Copilot
LLM-powered research assistant (requires `OPENAI_API_KEY`).
- `analyze_backtest(result)` — Plain English analysis with recommendations
- `suggest_strategy(text)` — Convert ideas to structured strategy configs

Falls back to rule-based analysis if no API key is set.

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Historical data | yfinance (free) |
| Real-time data | Polygon.io (free tier, optional) |
| Backtesting | VectorBT |
| Indicators | pandas-ta |
| Dashboard | Streamlit + Plotly |
| AI Layer | OpenAI gpt-4o-mini (optional) |
| Caching | DuckDB + Parquet |

---

## 📈 Adding New Strategies

1. Create a subclass of `Strategy` in `strategy.py`
2. Implement `name` property and `generate_signals(df)` method
3. Add to `STRATEGY_REGISTRY`

```python
class MyStrategy(Strategy):
    @property
    def name(self): return "My Strategy"
    
    def generate_signals(self, df):
        df = self._validate_df(df)
        # Your signal logic here
        df["signal"] = 0  # BUY=1, SELL=-1, HOLD=0
        return df
```

---

## ⚠️ Disclaimer

This system is for research and educational purposes only. Past backtest performance does not guarantee future results. Never risk money you cannot afford to lose.
