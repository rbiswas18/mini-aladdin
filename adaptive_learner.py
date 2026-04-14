"""
adaptive_learner.py — Adaptive learning and strategy optimization for Trading Alpha.
Python 3.9+ compatible, JSON-backed, beginner-friendly plain English outputs.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from backtest_simple import run_backtest
from strategy_simple import STRATEGY_REGISTRY, build_strategy

try:
    from strategy_pro import PRO_STRATEGY_REGISTRY, build_pro_strategy
except ImportError:
    PRO_STRATEGY_REGISTRY = {}
    build_pro_strategy = None


COMBINED_REGISTRY = {**STRATEGY_REGISTRY, **PRO_STRATEGY_REGISTRY}


class TradeAnalyzer:
    """Analyze completed trades and convert them into plain-English lessons."""

    def analyze_trades(self, trades_df: pd.DataFrame, strategy_name: str, symbol: str) -> dict:
        if trades_df is None or trades_df.empty:
            return {
                "strategy_name": strategy_name,
                "symbol": symbol,
                "total_trades": 0,
                "avg_win_size": 0.0,
                "avg_loss_size": 0.0,
                "win_count": 0,
                "loss_count": 0,
                "best_entry_conditions": ["No completed trades yet, so there is nothing reliable to learn."],
                "worst_entry_conditions": ["No losing trades yet, or no trades closed during this period."],
                "time_of_month_pattern": "No time-of-month pattern found because there were no completed trades.",
                "holding_period_pattern": "No holding period pattern found because there were no completed trades.",
                "summary": f"{strategy_name} on {symbol} did not produce any completed trades in this test period.",
            }

        df = trades_df.copy()
        df = self._normalize_trade_dates(df)
        winners = df[df["PnL"] > 0].copy()
        losers = df[df["PnL"] <= 0].copy()

        avg_win = float(winners["PnL"].mean()) if not winners.empty else 0.0
        avg_loss = float(losers["PnL"].mean()) if not losers.empty else 0.0

        best_entry_conditions = self._build_best_conditions(df, winners)
        worst_entry_conditions = self._build_worst_conditions(df, losers)
        time_pattern = self._time_of_month_pattern(df)
        holding_pattern = self._holding_period_pattern(df)

        summary = (
            f"{strategy_name} on {symbol} closed {len(df)} trades. "
            f"Average winner: ${avg_win:,.2f}. Average loser: ${avg_loss:,.2f}. "
            f"{time_pattern} {holding_pattern}"
        )

        return {
            "strategy_name": strategy_name,
            "symbol": symbol,
            "total_trades": int(len(df)),
            "avg_win_size": round(avg_win, 2),
            "avg_loss_size": round(avg_loss, 2),
            "win_count": int(len(winners)),
            "loss_count": int(len(losers)),
            "best_entry_conditions": best_entry_conditions,
            "worst_entry_conditions": worst_entry_conditions,
            "time_of_month_pattern": time_pattern,
            "holding_period_pattern": holding_pattern,
            "summary": summary,
        }

    def find_failure_patterns(self, trades_df: pd.DataFrame) -> List[str]:
        warnings: List[str] = []
        if trades_df is None or trades_df.empty:
            return ["No completed trades yet, so there are no failure patterns to review."]

        df = trades_df.copy()
        df = self._normalize_trade_dates(df)
        losers = df[df["PnL"] <= 0].copy()

        if "entry_near_52w_high" in df.columns and bool(df["entry_near_52w_high"].fillna(False).any()):
            count = int(df["entry_near_52w_high"].fillna(False).sum())
            warnings.append(f"Bought near the 52-week high on {count} trade(s), which suggests the strategy may be chasing extended moves.")

        if not losers.empty:
            avg_loss_abs = float(losers["PnL"].abs().mean())
            outsized = losers[losers["PnL"].abs() > (2 * avg_loss_abs)] if avg_loss_abs > 0 else pd.DataFrame()
            if not outsized.empty:
                warnings.append(
                    f"Loss exceeded 2x the average loss on {len(outsized)} trade(s). That is a sign that risk controls may be too loose."
                )

            if "Holding Days" in losers.columns:
                long_losers = losers[losers["Holding Days"] > 20]
                if not long_losers.empty:
                    warnings.append(
                        f"Held losing trade longer than 20 days on {len(long_losers)} trade(s). This suggests holding losers too long."
                    )

        recent = df.tail(5)
        if len(recent) == 5:
            recent_win_rate = (recent["PnL"] > 0).mean() * 100
            if recent_win_rate < 30:
                warnings.append(
                    f"Win rate was only {recent_win_rate:.0f}% in the last 5 trades. The strategy may be degrading in the current regime."
                )

        if not warnings:
            warnings.append("No major failure pattern stood out in this sample. Losses looked contained and recent trade quality was stable.")

        return warnings

    def calculate_strategy_score(self, metrics: dict) -> dict:
        total_return = float(metrics.get("total_return", 0.0))
        sharpe = float(metrics.get("sharpe_ratio", 0.0))
        win_rate = float(metrics.get("win_rate", 0.0))
        profit_factor = float(metrics.get("profit_factor", 0.0))
        max_drawdown = abs(float(metrics.get("max_drawdown", 0.0)))
        benchmark_return = float(metrics.get("benchmark_return", 0.0))

        return_edge = total_return - benchmark_return
        if return_edge >= 25:
            return_score = 25
        elif return_edge >= 10:
            return_score = 20
        elif return_edge >= 0:
            return_score = 16
        elif total_return > 0:
            return_score = 12
        elif total_return > -10:
            return_score = 8
        else:
            return_score = 3

        if sharpe >= 1.5:
            risk_score = 25
        elif sharpe >= 1.0:
            risk_score = 20
        elif sharpe >= 0.5:
            risk_score = 15
        elif sharpe > 0:
            risk_score = 10
        else:
            risk_score = 4

        consistency_raw = (win_rate * 0.5) + (min(profit_factor, 3.0) / 3.0 * 50)
        if consistency_raw >= 75:
            consistency_score = 25
        elif consistency_raw >= 60:
            consistency_score = 20
        elif consistency_raw >= 45:
            consistency_score = 15
        elif consistency_raw >= 30:
            consistency_score = 10
        else:
            consistency_score = 5

        if max_drawdown <= 10:
            drawdown_score = 25
        elif max_drawdown <= 20:
            drawdown_score = 20
        elif max_drawdown <= 30:
            drawdown_score = 15
        elif max_drawdown <= 40:
            drawdown_score = 10
        else:
            drawdown_score = 4

        total_score = int(round(return_score + risk_score + consistency_score + drawdown_score))

        if total_score >= 90:
            grade = "A"
            recommendation = "Strong candidate. Keep risk controls in place, but this strategy looks robust." 
        elif total_score >= 75:
            grade = "B"
            recommendation = "Promising. Worth paper trading and monitoring for regime changes."
        elif total_score >= 60:
            grade = "C"
            recommendation = "Mixed quality. Improve entries or exits before trusting it with real money."
        elif total_score >= 45:
            grade = "D"
            recommendation = "Weak. Only use for research, not live trading."
        else:
            grade = "F"
            recommendation = "Poor fit. Avoid this setup and look for a different strategy or parameter set."

        return {
            "total_score": total_score,
            "breakdown": {
                "return_score": int(return_score),
                "risk_score": int(risk_score),
                "consistency_score": int(consistency_score),
                "drawdown_score": int(drawdown_score),
            },
            "grade": grade,
            "recommendation": recommendation,
        }

    def _normalize_trade_dates(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        df = trades_df.copy()
        if "Entry Date" in df.columns:
            df["Entry Date"] = pd.to_datetime(df["Entry Date"], errors="coerce")
        if "Exit Date" in df.columns:
            df["Exit Date"] = pd.to_datetime(df["Exit Date"], errors="coerce")
        if "Holding Days" not in df.columns and "Entry Date" in df.columns and "Exit Date" in df.columns:
            df["Holding Days"] = (df["Exit Date"] - df["Entry Date"]).dt.days
        if "Entry Day" not in df.columns and "Entry Date" in df.columns:
            df["Entry Day"] = df["Entry Date"].dt.day
        if "Entry Bucket" not in df.columns and "Entry Day" in df.columns:
            df["Entry Bucket"] = df["Entry Day"].apply(self._bucket_month_day)
        return df

    def _bucket_month_day(self, day_number: Any) -> str:
        try:
            day = int(day_number)
        except Exception:
            return "unknown"
        if day <= 10:
            return "early month"
        if day <= 20:
            return "mid month"
        return "late month"

    def _time_of_month_pattern(self, df: pd.DataFrame) -> str:
        if "Entry Bucket" not in df.columns:
            return "No time-of-month pattern could be measured."
        bucket_perf = df.groupby("Entry Bucket")["PnL"].mean().sort_values(ascending=False)
        if bucket_perf.empty:
            return "No time-of-month pattern could be measured."
        best_bucket = str(bucket_perf.index[0])
        worst_bucket = str(bucket_perf.index[-1])
        return f"Best entries tended to happen in the {best_bucket}, while the weakest entries tended to happen in the {worst_bucket}."

    def _holding_period_pattern(self, df: pd.DataFrame) -> str:
        if "Holding Days" not in df.columns:
            return "No holding period pattern could be measured."
        winners = df[df["PnL"] > 0]
        losers = df[df["PnL"] <= 0]
        if winners.empty or losers.empty:
            return "There were not enough winners and losers to compare holding periods." 
        winner_days = float(winners["Holding Days"].mean())
        loser_days = float(losers["Holding Days"].mean())
        if winner_days > loser_days:
            return f"Winning trades were held about {winner_days:.1f} days on average, versus {loser_days:.1f} days for losers, so patience helped the winners more than the losers."
        return f"Losing trades were held about {loser_days:.1f} days on average, versus {winner_days:.1f} days for winners, which suggests exits may be too slow when trades go bad."

    def _build_best_conditions(self, df: pd.DataFrame, winners: pd.DataFrame) -> List[str]:
        if winners.empty:
            return ["No winners in this sample, so there are no strong entry conditions to copy yet."]
        notes: List[str] = []
        if "Entry Bucket" in winners.columns:
            best_bucket = winners.groupby("Entry Bucket")["PnL"].mean().sort_values(ascending=False).index[0]
            notes.append(f"Best winning trades entered in the {best_bucket}.")
        if "Holding Days" in winners.columns:
            notes.append(f"Winning trades were held about {winners['Holding Days'].mean():.1f} days on average.")
        if "entry_near_52w_high" in winners.columns:
            ratio = winners["entry_near_52w_high"].fillna(False).mean() * 100
            if ratio < 30:
                notes.append("Most winners were not bought right at the 52-week high, so waiting for better value may help.")
            else:
                notes.append("A meaningful share of winners came from strength near highs, so momentum entries can work when the trend is clean.")
        if not notes:
            notes.append("Winners tended to come from cleaner entries with smaller early drawdowns.")
        return notes

    def _build_worst_conditions(self, df: pd.DataFrame, losers: pd.DataFrame) -> List[str]:
        if losers.empty:
            return ["No losing trades in this sample."]
        notes: List[str] = []
        if "Entry Bucket" in losers.columns:
            worst_bucket = losers.groupby("Entry Bucket")["PnL"].mean().sort_values().index[0]
            notes.append(f"Worst trades tended to start in the {worst_bucket}.")
        if "Holding Days" in losers.columns:
            long_loss_days = float(losers["Holding Days"].mean())
            notes.append(f"Losing trades were held about {long_loss_days:.1f} days on average.")
        if "entry_near_52w_high" in losers.columns and bool(losers["entry_near_52w_high"].fillna(False).any()):
            notes.append("Some losers were opened near a 52-week high, which can mean buying after the move is already stretched.")
        if not notes:
            notes.append("Losing trades often started when the setup was weaker and the move had less follow-through.")
        return notes


class StrategyOptimizer:
    """Brute-force a small parameter grid and rank results by Sharpe ratio."""

    def get_parameter_grid(self, strategy_name: str) -> List[dict]:
        if strategy_name == "MovingAverageCrossover":
            grid = []
            for fast_period, slow_period in itertools.product([5, 10, 20], [30, 50, 100]):
                if fast_period < slow_period:
                    grid.append({"fast_period": fast_period, "slow_period": slow_period})
            return grid[:50]

        if strategy_name == "RSIMeanReversion":
            grid = []
            for rsi_period, oversold, overbought in itertools.product([10, 14, 21], [25, 30, 35], [65, 70, 75]):
                if oversold < overbought:
                    grid.append({"rsi_period": rsi_period, "oversold": oversold, "overbought": overbought})
            return grid[:50]

        if strategy_name == "BollingerBands":
            return [
                {"window": w, "std_dev": s}
                for w, s in itertools.product([10, 15, 20, 25], [1.5, 2.0, 2.5])
            ][:50]

        if strategy_name == "Momentum":
            return [{"window": v} for v in [10, 20, 30, 50]][:50]

        if strategy_name == "MACDStrategy":
            grid = []
            for fast, slow, signal in itertools.product([8, 12, 16], [21, 26, 35], [6, 9, 12]):
                if fast < slow:
                    grid.append({"fast": fast, "slow": slow, "signal": signal})
            return grid[:50]

        if strategy_name == "CombinedSignal":
            return [{"required_votes": v} for v in [1, 2, 3]][:50]

        if strategy_name == "TrendVolumeConfirmation":
            return [
                {"ema_period": 30, "volume_multiplier": 1.2, "adx_period": 14, "adx_threshold": 20.0, "volume_window": 20},
                {"ema_period": 50, "volume_multiplier": 1.5, "adx_period": 14, "adx_threshold": 25.0, "volume_window": 20},
                {"ema_period": 75, "volume_multiplier": 1.5, "adx_period": 20, "adx_threshold": 25.0, "volume_window": 30},
                {"ema_period": 100, "volume_multiplier": 2.0, "adx_period": 14, "adx_threshold": 30.0, "volume_window": 20},
            ]

        if strategy_name == "MeanReversionRegimeFilter":
            return [
                {"rsi_period": 14, "bb_period": 20, "bb_std_dev": 2.0, "bb_threshold": 0.05, "bb_trend_threshold": 0.08, "rsi_oversold": 35.0, "rsi_overbought": 65.0, "sma_period": 20, "price_tolerance": 0.02},
                {"rsi_period": 10, "bb_period": 15, "bb_std_dev": 2.0, "bb_threshold": 0.04, "bb_trend_threshold": 0.07, "rsi_oversold": 30.0, "rsi_overbought": 70.0, "sma_period": 15, "price_tolerance": 0.015},
                {"rsi_period": 21, "bb_period": 25, "bb_std_dev": 2.5, "bb_threshold": 0.06, "bb_trend_threshold": 0.09, "rsi_oversold": 35.0, "rsi_overbought": 68.0, "sma_period": 25, "price_tolerance": 0.03},
                {"rsi_period": 14, "bb_period": 20, "bb_std_dev": 1.5, "bb_threshold": 0.05, "bb_trend_threshold": 0.08, "rsi_oversold": 32.0, "rsi_overbought": 68.0, "sma_period": 20, "price_tolerance": 0.02},
            ]

        if strategy_name == "MultiTimeframeMomentum":
            return [
                {"short_roc": 3, "medium_roc": 15, "ema_period": 50},
                {"short_roc": 5, "medium_roc": 20, "ema_period": 100},
                {"short_roc": 7, "medium_roc": 30, "ema_period": 120},
                {"short_roc": 10, "medium_roc": 40, "ema_period": 150},
            ]

        registry_entry = COMBINED_REGISTRY.get(strategy_name, {})
        defaults = registry_entry.get("default_params", {})
        return [defaults] if defaults else [{}]

    def optimize(
        self,
        symbol: str,
        strategy_name: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 1200.0,
    ) -> dict:
        grid = self.get_parameter_grid(strategy_name)[:50]
        if not grid:
            raise ValueError(f"No parameter grid found for {strategy_name}")

        results: List[dict] = []
        errors: List[str] = []

        for params in grid:
            try:
                strategy = _build_any_strategy(strategy_name, params)
                result = run_backtest(
                    symbol=symbol,
                    strategy=strategy,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                )
                metrics = dict(result.metrics)
                metrics["trade_count"] = int(len(result.trades))
                results.append({
                    "params": params,
                    "metrics": metrics,
                    "strategy_display_name": result.strategy_name,
                })
            except Exception as exc:
                errors.append(f"{params}: {exc}")

        if not results:
            raise ValueError(f"All optimization runs failed for {strategy_name}. First error: {errors[0] if errors else 'Unknown error'}")

        ranked = sorted(
            results,
            key=lambda item: (
                float(item["metrics"].get("sharpe_ratio", -999)),
                float(item["metrics"].get("total_return", -999)),
            ),
            reverse=True,
        )
        best = ranked[0]
        trade_count = int(best["metrics"].get("trade_count", best["metrics"].get("total_trades", 0)))
        overfit_warning = None
        if trade_count <= 2:
            overfit_warning = "Best parameter set only produced 1-2 trades. This may be overfit and not reliable."

        return {
            "symbol": symbol,
            "strategy_name": strategy_name,
            "tested_combinations": len(grid),
            "successful_runs": len(results),
            "best_params": best["params"],
            "best_metrics": best["metrics"],
            "top_3": ranked[:3],
            "overfit_warning": overfit_warning,
            "errors": errors[:10],
        }


class LearningJournal:
    """Simple JSON persistence layer for lessons, backtests, and summaries."""

    def __init__(self, filepath: str = "learning_journal.json"):
        self.filepath = filepath
        self.data = {
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "backtests": [],
            "insights": [],
        }
        self.load(filepath)

    def log_backtest(self, symbol, strategy_name, params, metrics, analysis):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "strategy_name": strategy_name,
            "params": params or {},
            "metrics": metrics or {},
            "analysis": analysis or {},
        }
        self.data.setdefault("backtests", []).append(entry)
        self.data["updated_at"] = datetime.utcnow().isoformat()
        self.save(self.filepath)

    def log_insight(self, insight: str, category: str):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "category": category,
            "insight": insight,
        }
        self.data.setdefault("insights", []).append(entry)
        self.data["updated_at"] = datetime.utcnow().isoformat()
        self.save(self.filepath)

    def get_lessons_learned(self, strategy_name: str = None) -> List[str]:
        lessons: List[str] = []
        for insight in self.data.get("insights", []):
            if strategy_name and strategy_name.lower() not in insight.get("insight", "").lower() and strategy_name.lower() not in insight.get("category", "").lower():
                continue
            lessons.append(insight.get("insight", ""))
        return lessons[-20:]

    def get_best_performing(self, top_n=5) -> List[dict]:
        backtests = list(self.data.get("backtests", []))
        ranked = sorted(backtests, key=lambda x: float(x.get("metrics", {}).get("strategy_score", 0)), reverse=True)
        return ranked[:top_n]

    def get_worst_performing(self, top_n=5) -> List[dict]:
        backtests = list(self.data.get("backtests", []))
        ranked = sorted(backtests, key=lambda x: float(x.get("metrics", {}).get("strategy_score", 0)))
        return ranked[:top_n]

    def generate_report(self) -> str:
        backtests = self.data.get("backtests", [])
        insights = self.data.get("insights", [])
        if not backtests:
            return "The learning journal is still empty. Run an analysis or optimization to start building lessons."

        best = self.get_best_performing(3)
        worst = self.get_worst_performing(3)
        lines = ["Trading Alpha Learning Journal", ""]
        lines.append(f"Total backtests logged: {len(backtests)}")
        lines.append(f"Insights captured: {len(insights)}")
        lines.append("")
        lines.append("Best recent performers:")
        for item in best:
            m = item.get("metrics", {})
            lines.append(
                f"- {item.get('symbol')} | {item.get('strategy_name')} | score {m.get('strategy_score', 0)} | return {m.get('total_return', 0)}% | Sharpe {m.get('sharpe_ratio', 0)}"
            )
        lines.append("")
        lines.append("Worst recent performers:")
        for item in worst:
            m = item.get("metrics", {})
            lines.append(
                f"- {item.get('symbol')} | {item.get('strategy_name')} | score {m.get('strategy_score', 0)} | return {m.get('total_return', 0)}% | Sharpe {m.get('sharpe_ratio', 0)}"
            )
        if insights:
            lines.append("")
            lines.append("Latest lessons:")
            for insight in insights[-5:]:
                lines.append(f"- [{insight.get('category', 'general')}] {insight.get('insight', '')}")
        return "\n".join(lines)

    def save(self, filepath="learning_journal.json"):
        self.filepath = filepath
        Path(filepath).write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def load(self, filepath):
        self.filepath = filepath
        path = Path(filepath)
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                    "backtests": [],
                    "insights": [],
                }
        return self


class AdaptiveLearner:
    """Main orchestrator for analysis, optimization, and journal updates."""

    def __init__(self, symbols=None, state_file="learning_journal.json"):
        self.symbols = symbols or []
        self.state_file = state_file
        self.analyzer = TradeAnalyzer()
        self.optimizer = StrategyOptimizer()
        self.journal = LearningJournal(state_file)

    def run_full_analysis(self, symbol: str, strategy_name: str, years_back: int = 10) -> dict:
        end_dt = date.today() - timedelta(days=1)
        start_dt = end_dt - timedelta(days=int(years_back * 365.25))
        params = self._default_params(strategy_name)
        strategy = _build_any_strategy(strategy_name, params)

        result = run_backtest(
            symbol=symbol,
            strategy=strategy,
            start_date=str(start_dt),
            end_date=str(end_dt),
            initial_capital=1200.0,
        )

        benchmark_return = self._benchmark_return(symbol, str(start_dt), str(end_dt))
        trades = self._enrich_trades(result.trades, symbol, str(start_dt), str(end_dt))
        analysis = self.analyzer.analyze_trades(trades, strategy_name, symbol)
        failure_patterns = self.analyzer.find_failure_patterns(trades)

        metrics = dict(result.metrics)
        metrics["benchmark_return"] = benchmark_return
        score = self.analyzer.calculate_strategy_score(metrics)
        metrics["strategy_score"] = score["total_score"]
        metrics["grade"] = score["grade"]

        suggestions = self._build_improvement_suggestions(metrics, failure_patterns, analysis)

        report = {
            "symbol": symbol,
            "strategy_name": strategy_name,
            "strategy_display_name": result.strategy_name,
            "period": {"start_date": str(start_dt), "end_date": str(end_dt), "years_back": years_back},
            "params": params,
            "metrics": metrics,
            "analysis": analysis,
            "strategy_score": score,
            "failure_patterns": failure_patterns,
            "improvement_suggestions": suggestions,
            "trade_count": int(len(trades)),
            "trades": trades.to_dict(orient="records") if not trades.empty else [],
        }

        self.journal.log_backtest(symbol, strategy_name, params, metrics, {
            "analysis": analysis,
            "failure_patterns": failure_patterns,
            "improvement_suggestions": suggestions,
        })

        for line in suggestions[:3]:
            self.journal.log_insight(f"{symbol} | {strategy_name}: {line}", "improvement")
        for line in failure_patterns[:3]:
            self.journal.log_insight(f"{symbol} | {strategy_name}: {line}", "risk")

        return report

    def optimize_and_learn(self, symbol: str, strategy_name: str, years_back: int = 10) -> dict:
        end_dt = date.today() - timedelta(days=1)
        start_dt = end_dt - timedelta(days=int(years_back * 365.25))

        optimization = self.optimizer.optimize(
            symbol=symbol,
            strategy_name=strategy_name,
            start_date=str(start_dt),
            end_date=str(end_dt),
            initial_capital=1200.0,
        )

        best_params = optimization["best_params"]
        strategy = _build_any_strategy(strategy_name, best_params)
        result = run_backtest(
            symbol=symbol,
            strategy=strategy,
            start_date=str(start_dt),
            end_date=str(end_dt),
            initial_capital=1200.0,
        )

        benchmark_return = self._benchmark_return(symbol, str(start_dt), str(end_dt))
        trades = self._enrich_trades(result.trades, symbol, str(start_dt), str(end_dt))
        analysis = self.analyzer.analyze_trades(trades, strategy_name, symbol)
        failure_patterns = self.analyzer.find_failure_patterns(trades)

        metrics = dict(result.metrics)
        metrics["benchmark_return"] = benchmark_return
        score = self.analyzer.calculate_strategy_score(metrics)
        metrics["strategy_score"] = score["total_score"]
        metrics["grade"] = score["grade"]

        suggestions = self._build_improvement_suggestions(metrics, failure_patterns, analysis)
        if optimization.get("overfit_warning"):
            suggestions.append(optimization["overfit_warning"])

        summary = {
            "symbol": symbol,
            "strategy_name": strategy_name,
            "best_params": best_params,
            "best_metrics": metrics,
            "top_3": optimization.get("top_3", []),
            "overfit_warning": optimization.get("overfit_warning"),
            "analysis": analysis,
            "failure_patterns": failure_patterns,
            "improvement_suggestions": suggestions,
            "strategy_score": score,
        }

        self.journal.log_backtest(symbol, strategy_name, best_params, metrics, {
            "analysis": analysis,
            "failure_patterns": failure_patterns,
            "improvement_suggestions": suggestions,
            "optimization": optimization,
        })
        self.journal.log_insight(
            f"{symbol} | {strategy_name}: best params {best_params} produced score {score['total_score']} and grade {score['grade']}.",
            "optimization",
        )
        return summary

    def generate_daily_briefing(self) -> str:
        backtests = self.journal.data.get("backtests", [])
        if not backtests:
            return "Here is what the system learned today: nothing has been analyzed yet. Run a full analysis or optimization first."

        best_by_symbol: Dict[str, dict] = {}
        failure_counter: Counter = Counter()
        suggestion_counter: Counter = Counter()

        for entry in backtests:
            symbol = entry.get("symbol", "UNKNOWN")
            score = float(entry.get("metrics", {}).get("strategy_score", 0))
            current_best = best_by_symbol.get(symbol)
            if current_best is None or score > float(current_best.get("metrics", {}).get("strategy_score", 0)):
                best_by_symbol[symbol] = entry

            analysis_block = entry.get("analysis", {})
            if isinstance(analysis_block, dict):
                for item in analysis_block.get("failure_patterns", []) or []:
                    failure_counter[item] += 1
                for item in analysis_block.get("improvement_suggestions", []) or []:
                    suggestion_counter[item] += 1

        lines = ["🧠 Here is what the system learned today:", ""]
        lines.append("Best strategy by symbol:")
        for symbol, entry in sorted(best_by_symbol.items()):
            metrics = entry.get("metrics", {})
            lines.append(
                f"- {symbol}: {entry.get('strategy_name')} scored {metrics.get('strategy_score', 0)}/100 ({metrics.get('grade', 'N/A')}) with return {metrics.get('total_return', 0)}% and Sharpe {metrics.get('sharpe_ratio', 0)}."
            )

        lines.append("")
        lines.append("Most common failure patterns:")
        if failure_counter:
            for item, _count in failure_counter.most_common(3):
                lines.append(f"- {item}")
        else:
            lines.append("- No repeated failure pattern stood out yet.")

        lines.append("")
        lines.append("Improvement suggestions:")
        if suggestion_counter:
            for item, _count in suggestion_counter.most_common(3):
                lines.append(f"- {item}")
        else:
            lines.append("- Keep collecting more backtests so the learner has a stronger sample.")

        return "\n".join(lines)

    def _default_params(self, strategy_name: str) -> dict:
        return dict(COMBINED_REGISTRY.get(strategy_name, {}).get("default_params", {}))

    def _benchmark_return(self, symbol: str, start_date: str, end_date: str) -> float:
        try:
            df = yf.Ticker(symbol.upper()).history(start=start_date, end=end_date, auto_adjust=True)
            if df.empty:
                return 0.0
            return round((float(df["Close"].iloc[-1]) / float(df["Close"].iloc[0]) - 1.0) * 100.0, 2)
        except Exception:
            return 0.0

    def _enrich_trades(self, trades_df: pd.DataFrame, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        if trades_df is None or trades_df.empty:
            return pd.DataFrame()

        df = trades_df.copy()
        df["Entry Date"] = pd.to_datetime(df["Entry Date"], errors="coerce")
        df["Exit Date"] = pd.to_datetime(df["Exit Date"], errors="coerce")
        df["Holding Days"] = (df["Exit Date"] - df["Entry Date"]).dt.days
        df["Entry Day"] = df["Entry Date"].dt.day
        df["Entry Bucket"] = df["Entry Day"].apply(self.analyzer._bucket_month_day)
        df["entry_near_52w_high"] = False

        try:
            raw = yf.Ticker(symbol.upper()).history(start=start_date, end=end_date, auto_adjust=True)
            if not raw.empty:
                raw = raw[["Close"]].copy()
                raw["rolling_252_high"] = raw["Close"].rolling(252, min_periods=20).max()
                raw["pct_from_52w_high"] = raw["Close"] / raw["rolling_252_high"] - 1.0
                raw.index = pd.to_datetime(raw.index).tz_localize(None)
                high_lookup = raw[["pct_from_52w_high"]].copy()
                high_lookup["entry_near_52w_high"] = high_lookup["pct_from_52w_high"] >= -0.02
                mapping = high_lookup["entry_near_52w_high"].to_dict()
                df["entry_near_52w_high"] = df["Entry Date"].map(mapping).fillna(False)
        except Exception:
            pass

        return df

    def _build_improvement_suggestions(self, metrics: dict, failure_patterns: List[str], analysis: dict) -> List[str]:
        suggestions: List[str] = []
        if float(metrics.get("max_drawdown", 0)) < -25:
            suggestions.append("Max drawdown was deep. Tighten stop losses or reduce position size so losses stay survivable.")
        if float(metrics.get("profit_factor", 0)) < 1.2:
            suggestions.append("Profit factor is weak. Focus on fewer, higher-quality entries instead of forcing trades.")
        if float(metrics.get("win_rate", 0)) < 35:
            suggestions.append("Win rate is low. Consider more selective entry rules or adding a regime filter.")
        if any("52-week high" in item for item in failure_patterns):
            suggestions.append("Several losses came from buying near highs. Wait for pullbacks or stronger confirmation before entering.")
        if any("20 days" in item for item in failure_patterns):
            suggestions.append("Some losers were held too long. Define a time stop so weak trades get closed earlier.")

        for item in analysis.get("worst_entry_conditions", [])[:2]:
            suggestions.append(item)

        deduped: List[str] = []
        seen = set()
        for item in suggestions:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        return deduped[:6]


LEARN_TAB_CODE = '''
with tab6:
    st.header("🧠 Learn & Optimize")
    st.caption("Let the system study past trades, score strategies, and keep a journal of what it learns.")

    try:
        from adaptive_learner import AdaptiveLearner
    except Exception as e:
        st.error(f"Could not load adaptive learner: {e}")
    else:
        learner = AdaptiveLearner(state_file="learning_journal.json")

        learn_col1, learn_col2 = st.columns(2)
        with learn_col1:
            learn_symbol = st.text_input("Symbol to learn from", value=symbol, key="learn_symbol").upper().strip()
        with learn_col2:
            learn_strategy = st.selectbox("Strategy to analyze", list(COMBINED_REGISTRY.keys()), key="learn_strategy")

        years_back = st.slider("Years of history", 3, 20, 10, key="learn_years")

        btn_col1, btn_col2, btn_col3 = st.columns(3)
        if btn_col1.button("🧠 Run Full Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing completed trades and learning from wins and losses..."):
                try:
                    st.session_state["learn_full_report"] = learner.run_full_analysis(learn_symbol, learn_strategy, years_back)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

        if btn_col2.button("⚙️ Optimize Parameters", use_container_width=True):
            with st.spinner("Sweeping parameter combinations to find better settings..."):
                try:
                    st.session_state["learn_opt_report"] = learner.optimize_and_learn(learn_symbol, learn_strategy, years_back)
                except Exception as e:
                    st.error(f"Optimization failed: {e}")

        if btn_col3.button("📰 Daily Briefing", use_container_width=True):
            try:
                st.session_state["learn_daily_briefing"] = learner.generate_daily_briefing()
            except Exception as e:
                st.error(f"Could not generate briefing: {e}")

        full_report = st.session_state.get("learn_full_report")
        if full_report:
            st.subheader("Strategy Score")
            score = full_report.get("strategy_score", {})
            score_cols = st.columns(4)
            score_cols[0].metric("Score", f"{score.get('total_score', 0)}/100")
            score_cols[1].metric("Grade", score.get("grade", "N/A"))
            score_cols[2].metric("Return", f"{full_report.get('metrics', {}).get('total_return', 0)}%")
            score_cols[3].metric("Sharpe", full_report.get('metrics', {}).get('sharpe_ratio', 0))

            st.info(score.get("recommendation", ""))

            st.subheader("Failure Patterns")
            for item in full_report.get("failure_patterns", []):
                st.write(f"- {item}")

            st.subheader("What the learner noticed")
            analysis = full_report.get("analysis", {})
            st.write(f"- {analysis.get('summary', '')}")
            for item in analysis.get("best_entry_conditions", []):
                st.write(f"- {item}")
            for item in full_report.get("improvement_suggestions", []):
                st.write(f"- {item}")

        opt_report = st.session_state.get("learn_opt_report")
        if opt_report:
            st.subheader("Best Parameters Found")
            st.json(opt_report.get("best_params", {}))
            if opt_report.get("overfit_warning"):
                st.warning(opt_report["overfit_warning"])

            top3 = opt_report.get("top_3", [])
            if top3:
                rows = []
                for idx, item in enumerate(top3, start=1):
                    metrics = item.get("metrics", {})
                    rows.append({
                        "Rank": idx,
                        "Params": str(item.get("params", {})),
                        "Sharpe": metrics.get("sharpe_ratio", 0),
                        "Return %": metrics.get("total_return", 0),
                        "Win Rate %": metrics.get("win_rate", 0),
                        "Drawdown %": metrics.get("max_drawdown", 0),
                        "Trades": metrics.get("trade_count", metrics.get("total_trades", 0)),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.subheader("View Journal")
        lessons = learner.journal.get_lessons_learned(learn_strategy)
        if lessons:
            st.markdown("**Lessons learned**")
            for item in lessons[-10:]:
                st.write(f"- {item}")
        else:
            st.caption("No lessons stored yet.")

        best_runs = learner.journal.get_best_performing(5)
        worst_runs = learner.journal.get_worst_performing(5)
        journal_col1, journal_col2 = st.columns(2)
        with journal_col1:
            st.markdown("**Best performers**")
            if best_runs:
                st.dataframe(pd.DataFrame([
                    {
                        "Symbol": item.get("symbol"),
                        "Strategy": item.get("strategy_name"),
                        "Score": item.get("metrics", {}).get("strategy_score", 0),
                        "Return %": item.get("metrics", {}).get("total_return", 0),
                        "Sharpe": item.get("metrics", {}).get("sharpe_ratio", 0),
                    }
                    for item in best_runs
                ]), use_container_width=True)
            else:
                st.caption("Nothing logged yet.")
        with journal_col2:
            st.markdown("**Worst performers**")
            if worst_runs:
                st.dataframe(pd.DataFrame([
                    {
                        "Symbol": item.get("symbol"),
                        "Strategy": item.get("strategy_name"),
                        "Score": item.get("metrics", {}).get("strategy_score", 0),
                        "Return %": item.get("metrics", {}).get("total_return", 0),
                        "Sharpe": item.get("metrics", {}).get("sharpe_ratio", 0),
                    }
                    for item in worst_runs
                ]), use_container_width=True)
            else:
                st.caption("Nothing logged yet.")

        briefing = st.session_state.get("learn_daily_briefing")
        if briefing:
            st.subheader("Daily Briefing")
            st.text(briefing)
'''


def _build_any_strategy(strategy_name: str, params: Optional[dict] = None):
    params = params or {}
    if strategy_name in STRATEGY_REGISTRY:
        return build_strategy(strategy_name, **params)
    if strategy_name in PRO_STRATEGY_REGISTRY and build_pro_strategy:
        return build_pro_strategy(strategy_name, **params)
    raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {list(COMBINED_REGISTRY.keys())}")
