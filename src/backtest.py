from __future__ import annotations

import pandas as pd

from .config import AppConfig, get_config
from .strategy import SignalRecommendation, load_predictions, pick_signal


def settle_bet(total_goals: float, line: float, selection: str, stake: float, odds: float) -> tuple[bool, float]:
    if pd.isna(total_goals):
        return False, 0.0
    if selection == "Over":
        won = total_goals > line
    else:
        won = total_goals < line
    profit = stake * (odds - 1) if won else -stake
    return won, profit


def run_backtest(cfg: AppConfig) -> pd.DataFrame:
    predictions = load_predictions(cfg)
    bankroll = cfg.backtest.starting_bankroll
    records = []
    for _, row in predictions.sort_values("Date").iterrows():
        signal = pick_signal(row, cfg)
        if not signal or pd.isna(row.get("total_goals")):
            continue
        recommended_stake = signal.stake
        if recommended_stake <= 0:
            continue
        stake_amount = min(
            recommended_stake,
            bankroll * cfg.backtest.max_bet_fraction,
        )
        if stake_amount <= 0:
            continue
        won, profit = settle_bet(
            total_goals=row["total_goals"],
            line=signal.line,
            selection=signal.selection,
            stake=stake_amount,
            odds=signal.odds,
        )
        bankroll += profit
        record = {
            "match_id": signal.match_id,
            "date": signal.date,
            "league_code": signal.league_code,
            "selection": signal.selection,
            "line": signal.line,
            "odds": signal.odds,
            "stake": stake_amount,
            "probability": signal.probability,
            "edge": signal.edge,
            "actual_total": row["total_goals"],
            "won": won,
            "profit": profit,
            "bankroll_after": bankroll,
        }
        records.append(record)
    if not records:
        print("No qualifying bets for backtest.")
        return pd.DataFrame()
    results = pd.DataFrame(records)
    results.to_csv(cfg.processed_dir / "backtest_trades.csv", index=False)
    summarize_backtest(results, cfg.backtest.starting_bankroll)
    return results


def summarize_backtest(trades: pd.DataFrame, starting_bankroll: float) -> None:
    total_profit = trades["profit"].sum()
    roi = total_profit / starting_bankroll
    hit_rate = trades["won"].mean()
    max_bankroll = trades["bankroll_after"].cummax()
    drawdowns = trades["bankroll_after"] - max_bankroll
    pct_drawdown = (drawdowns / max_bankroll.replace(0, pd.NA)).min()
    print(
        "Backtest summary",
        {
            "bets": int(len(trades)),
            "roi": round(roi, 3),
            "hit_rate": round(hit_rate, 3),
            "max_drawdown": float(pct_drawdown or 0),
            "ending_bankroll": float(trades["bankroll_after"].iloc[-1]),
        },
    )


def main() -> None:
    cfg = get_config()
    run_backtest(cfg)


if __name__ == "__main__":
    main()
