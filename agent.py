"""
agent.py — AI Strategy Copilot
LLM-powered layer for strategy suggestion, backtest analysis, and natural language input.
The agent generates configs and analysis — deterministic code does execution.
"""

import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv

from backtest import BacktestResult
from strategy import STRATEGY_REGISTRY

load_dotenv()
logger = logging.getLogger(__name__)

OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
    OPENAI_AVAILABLE = _client is not None
except ImportError:
    _client = None


SYSTEM_PROMPT = """You are a quantitative trading expert and system architect.
You help users analyze backtest results and discover trading strategies.
Be precise, data-driven, and practical. Avoid hype or vague statements.
Always ground your analysis in the actual numbers provided."""


class TradingAgent:
    """
    AI-powered trading research copilot.

    Responsibilities:
    - Analyze backtest results in plain English
    - Convert natural language ideas into structured strategy configs
    - Suggest next experiments based on results

    The agent NEVER places trades or modifies execution logic.
    It only generates ideas, configs, and analysis text.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.available = OPENAI_AVAILABLE

        if not self.available:
            logger.warning(
                "OpenAI API key not set. Agent will use rule-based fallback analysis. "
                "Add OPENAI_API_KEY to your .env file to enable AI analysis."
            )

    def analyze_backtest(self, result: BacktestResult) -> str:
        """
        Analyze a backtest result and produce a plain-English summary.

        Covers: what worked, what didn't, risk profile, and suggested next steps.

        Args:
            result: BacktestResult dataclass from backtest engine

        Returns:
            String with multi-paragraph analysis
        """
        metrics = result.metrics

        if not self.available:
            return self._rule_based_analysis(result)

        prompt = f"""Analyze this trading strategy backtest result and provide actionable insights.

Strategy: {result.strategy_name}
Symbol: {result.symbol}
Period: {result.start_date} to {result.end_date}
Initial Capital: ${metrics['initial_capital']:,.2f}
Final Value: ${metrics['final_value']:,.2f}

Performance Metrics:
- Total Return: {metrics['total_return']}%
- CAGR: {metrics['cagr']}%
- Sharpe Ratio: {metrics['sharpe_ratio']}
- Max Drawdown: {metrics['max_drawdown']}%
- Win Rate: {metrics['win_rate']}%
- Profit Factor: {metrics['profit_factor']}
- Total Trades: {metrics['total_trades']}

Provide:
1. Overall assessment (2-3 sentences)
2. Strengths of this result
3. Weaknesses or risk concerns
4. 2-3 specific next experiments to run (parameter tweaks, different symbols, or different strategies)
5. One key risk to watch in live trading

Be direct and specific. Use the actual numbers."""

        try:
            response = _client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=800,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
            return self._rule_based_analysis(result)

    def suggest_strategy(self, user_input: str) -> dict:
        """
        Convert a natural language strategy idea into a structured config.

        Args:
            user_input: Free-form description, e.g. "momentum strategy on large-cap tech"

        Returns:
            dict with keys: strategy, params, symbol, start_date, end_date
        """
        available_strategies = list(STRATEGY_REGISTRY.keys())
        strategy_descriptions = {
            "MovingAverageCrossover": "Buy when fast EMA crosses above slow EMA, sell on crossunder. Good for trending markets.",
            "RSIMeanReversion": "Buy when RSI is oversold, sell when overbought. Good for range-bound markets.",
            "MACDStrategy": "Buy on MACD/signal line bullish crossover, sell on bearish crossover. Momentum + trend.",
        }

        if not self.available:
            return self._rule_based_suggestion(user_input)

        prompt = f"""A user wants to test a trading strategy. Convert their idea into a structured config.

User's idea: "{user_input}"

Available strategies:
{json.dumps(strategy_descriptions, indent=2)}

Return ONLY a valid JSON object with exactly these fields:
{{
  "strategy": "<strategy name from available list>",
  "params": {{}},
  "symbol": "<US stock ticker>",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "reasoning": "<one sentence explaining your choices>"
}}

Rules:
- strategy must be one of: {available_strategies}
- params should override defaults only if the user implied specific values
- symbol should be a well-known US stock or ETF relevant to the user's idea
- default date range: last 3 years
- Return ONLY the JSON, no other text"""

        try:
            response = _client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown code blocks if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            config = json.loads(raw)
            # Validate required keys
            for key in ["strategy", "params", "symbol", "start_date", "end_date"]:
                if key not in config:
                    raise ValueError(f"Missing key: {key}")
            if config["strategy"] not in STRATEGY_REGISTRY:
                config["strategy"] = "MovingAverageCrossover"
            return config
        except Exception as e:
            logger.error(f"Strategy suggestion failed: {e}")
            return self._rule_based_suggestion(user_input)

    def _rule_based_analysis(self, result: BacktestResult) -> str:
        """Fallback analysis without LLM — uses heuristic rules on metrics."""
        m = result.metrics
        lines = [
            f"**Backtest Analysis: {result.strategy_name} on {result.symbol}**",
            f"Period: {result.start_date} → {result.end_date}",
            "",
        ]

        # Overall verdict
        if m["sharpe_ratio"] > 1.5 and m["total_return"] > 20:
            lines.append("✅ **Strong result.** Above-average risk-adjusted returns.")
        elif m["sharpe_ratio"] > 0.8:
            lines.append("🟡 **Decent result.** Acceptable risk-adjusted returns but room to improve.")
        else:
            lines.append("🔴 **Weak result.** Poor risk-adjusted returns — consider a different strategy or parameters.")

        lines.append("")
        lines.append(f"**Return:** {m['total_return']}% total | {m['cagr']}% CAGR")
        lines.append(f"**Risk:** {m['max_drawdown']}% max drawdown | Sharpe {m['sharpe_ratio']}")
        lines.append(f"**Trades:** {m['total_trades']} total | {m['win_rate']}% win rate | {m['profit_factor']}x profit factor")

        lines.append("")
        lines.append("**Suggested next steps:**")
        lines.append("- Try adjusting the strategy parameters (shorter/longer periods)")
        lines.append("- Run the same strategy on a different symbol or sector ETF")
        lines.append("- Compare against RSI or MACD strategy on the same period")

        if m["max_drawdown"] > 20:
            lines.append(f"- ⚠️ Drawdown of {m['max_drawdown']}% is high — add a stop-loss rule before going live")

        return "\n".join(lines)

    def _rule_based_suggestion(self, user_input: str) -> dict:
        """Fallback strategy suggestion without LLM."""
        user_lower = user_input.lower()

        if any(w in user_lower for w in ["rsi", "oversold", "mean reversion", "reversal"]):
            strategy = "RSIMeanReversion"
            params = {}
        elif any(w in user_lower for w in ["macd", "momentum", "crossover"]):
            strategy = "MACDStrategy"
            params = {}
        else:
            strategy = "MovingAverageCrossover"
            params = {}

        # Symbol heuristics
        if any(w in user_lower for w in ["tech", "technology", "nasdaq"]):
            symbol = "QQQ"
        elif any(w in user_lower for w in ["sp500", "s&p", "broad market"]):
            symbol = "SPY"
        elif any(w in user_lower for w in ["apple"]):
            symbol = "AAPL"
        elif any(w in user_lower for w in ["nvidia", "gpu", "ai chip"]):
            symbol = "NVDA"
        else:
            symbol = "SPY"

        return {
            "strategy": strategy,
            "params": params,
            "symbol": symbol,
            "start_date": "2021-01-01",
            "end_date": "2024-01-01",
            "reasoning": f"Rule-based suggestion (no LLM). Add OPENAI_API_KEY for smarter suggestions.",
        }


if __name__ == "__main__":
    agent = TradingAgent()
    print("Agent available:", agent.available)

    config = agent.suggest_strategy("momentum strategy on large-cap tech stocks")
    print("\nSuggested config:")
    print(json.dumps(config, indent=2))
