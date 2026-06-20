from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def generate_traffic_series(
    periods: int = 24 * 45,
    freq: str = "h",
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate synthetic hourly 5G traffic for eMBB, URLLC and mMTC.

    The generated data includes daily seasonality, weekly modulation, noise and
    bursts. It is useful when real operator traces are not available.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-01", periods=periods, freq=freq)
    hour = np.arange(periods) % 24
    day = np.arange(periods) // 24

    daily_evening = np.sin(2 * np.pi * (hour - 18) / 24)
    daily_business = np.sin(2 * np.pi * (hour - 10) / 24)
    weekly = 1 + 0.12 * np.sin(2 * np.pi * day / 7)

    embb = (420 + 150 * daily_evening.clip(min=-0.3) + rng.normal(0, 28, periods)) * weekly
    urllc = 95 + 30 * daily_business.clip(min=-0.2) + rng.normal(0, 10, periods)
    mmtc = 160 + 22 * np.sin(2 * np.pi * hour / 24 + 1.1) + rng.normal(0, 12, periods)

    burst_mask = rng.random(periods) < 0.04
    embb[burst_mask] += rng.uniform(120, 280, burst_mask.sum())
    mmtc_burst_mask = rng.random(periods) < 0.03
    mmtc[mmtc_burst_mask] += rng.uniform(40, 90, mmtc_burst_mask.sum())

    data = pd.DataFrame(
        {
            "timestamp": index,
            "embb_mbps": np.maximum(embb, 1).round(3),
            "urllc_mbps": np.maximum(urllc, 1).round(3),
            "mmtc_mbps": np.maximum(mmtc, 1).round(3),
        }
    )
    data["total_mbps"] = data[["embb_mbps", "urllc_mbps", "mmtc_mbps"]].sum(axis=1).round(3)
    return data
