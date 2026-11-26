"""
analytics.py
============
Turns the raw training history (one record per real training episode) into
statistics, summaries and CSV exports.  Every number is computed from recorded
episodes — nothing is estimated or hard-coded.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

BASE_COLUMNS = ["episode", "reward", "steps", "success", "wall_hits",
                "epsilon", "explore_actions", "exploit_actions"]


def history_to_dataframe(history: List[dict], window: int = 50) -> pd.DataFrame:
    """Build the training-statistics table with rolling (moving-window) metrics."""
    if not history:
        return pd.DataFrame(columns=BASE_COLUMNS + [
            "cumulative_reward", "avg_reward", "success_rate", "cumulative_success_rate",
            "steps_if_success", "avg_steps_success", "explore_ratio"])

    window = max(1, int(window))
    df = pd.DataFrame(history)[BASE_COLUMNS].copy()
    df["success"] = df["success"].astype(bool)
    success_num = df["success"].astype(float)

    df["cumulative_reward"] = df["reward"].cumsum()
    df["avg_reward"] = df["reward"].rolling(window, min_periods=1).mean()
    df["success_rate"] = success_num.rolling(window, min_periods=1).mean() * 100.0
    df["cumulative_success_rate"] = success_num.expanding().mean() * 100.0
    df["steps_if_success"] = df["steps"].where(df["success"])
    df["avg_steps_success"] = df["steps_if_success"].rolling(window, min_periods=1).mean()
    total_actions = (df["explore_actions"] + df["exploit_actions"]).replace(0, np.nan)
    df["explore_ratio"] = (df["explore_actions"] / total_actions * 100.0).fillna(0.0)
    return df


def convergence_episode(df: pd.DataFrame, optimal_steps: Optional[int],
                        window: int = 20, tolerance: float = 0.10) -> Optional[int]:
    """
    First episode after which the agent reached the goal in ``window`` consecutive
    episodes using at most (1 + tolerance) × optimal steps on average.
    Returns None if that has not happened yet.
    """
    if df.empty or optimal_steps is None or len(df) < window:
        return None
    success = df["success"].to_numpy()
    steps = df["steps"].to_numpy(dtype=float)
    limit = optimal_steps * (1.0 + tolerance)
    for i in range(0, len(df) - window + 1):
        chunk_ok = success[i:i + window].all()
        if chunk_ok and steps[i:i + window].mean() <= limit:
            return int(df["episode"].iloc[i])
    return None


def training_summary(df: pd.DataFrame, optimal_steps: Optional[int] = None,
                     recent: int = 100) -> Dict[str, Optional[float]]:
    """Headline numbers for the dashboard and analytics page."""
    if df.empty:
        return {
            "episodes": 0, "total_reward": 0.0, "success_rate": 0.0,
            "recent_success_rate": 0.0, "avg_steps_success": None,
            "recent_avg_steps_success": None, "best_reward": None,
            "avg_reward": None, "recent_avg_reward": None, "first_success": None,
            "convergence_episode": None, "total_wall_hits": 0, "total_steps": 0,
        }
    tail = df.tail(recent)
    wins = df[df["success"]]
    tail_wins = tail[tail["success"]]
    return {
        "episodes": int(len(df)),
        "total_reward": float(df["reward"].sum()),
        "success_rate": float(df["success"].mean() * 100.0),
        "recent_success_rate": float(tail["success"].mean() * 100.0),
        "avg_steps_success": float(wins["steps"].mean()) if not wins.empty else None,
        "recent_avg_steps_success": float(tail_wins["steps"].mean()) if not tail_wins.empty else None,
        "best_reward": float(df["reward"].max()),
        "avg_reward": float(df["reward"].mean()),
        "recent_avg_reward": float(tail["reward"].mean()),
        "first_success": int(wins["episode"].iloc[0]) if not wins.empty else None,
        "convergence_episode": convergence_episode(df, optimal_steps),
        "total_wall_hits": int(df["wall_hits"].sum()),
        "total_steps": int(df["steps"].sum()),
    }


def phase_comparison(df: pd.DataFrame, fraction: float = 0.10,
                     min_n: int = 5, max_n: int = 100) -> pd.DataFrame:
    """Compare the first X% of training episodes with the last X% (actual records)."""
    if len(df) < 2 * min_n:
        return pd.DataFrame()
    n = int(np.clip(round(len(df) * fraction), min_n, max_n))
    n = min(n, len(df) // 2)
    early, late = df.head(n), df.tail(n)

    def stats(part: pd.DataFrame) -> dict:
        wins = part[part["success"]]
        return {
            "Success rate (%)": part["success"].mean() * 100.0,
            "Average reward": part["reward"].mean(),
            "Average steps": part["steps"].mean(),
            "Average steps (successful)": wins["steps"].mean() if not wins.empty else np.nan,
            "Wall hits per episode": part["wall_hits"].mean(),
            "Exploration rate ε": part["epsilon"].mean(),
        }

    e, l = stats(early), stats(late)
    table = pd.DataFrame({
        f"First {n} episodes": e,
        f"Last {n} episodes": l,
    })
    table["Change"] = table.iloc[:, 1] - table.iloc[:, 0]
    return table.round(2)


def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    export = df.copy()
    return export.to_csv(index=False).encode("utf-8")


def downsample(df: pd.DataFrame, max_points: int = 3000) -> pd.DataFrame:
    """Keep charts responsive for very long training runs (display only)."""
    if len(df) <= max_points:
        return df
    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step]


def fmt_num(value, digits: int = 1, suffix: str = "", none: str = "—") -> str:
    """Pretty-print a number for stat cards."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return none
    if isinstance(value, (int, np.integer)) or float(value).is_integer() and digits == 0:
        return f"{int(value):,}{suffix}"
    return f"{float(value):,.{digits}f}{suffix}"
