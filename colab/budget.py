"""Compute-unit budget tracker and append-only ledger for Colab A100 campaigns."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER_PATH = REPO_ROOT / "colab" / "budget_ledger.jsonl"

A100_UNITS_PER_HOUR = 15.0
HARD_CAMPAIGN_CAP_UNITS = 25.0

HARDWARE_RATES: dict[str, float] = {
    "A100": 15.0,
    "H100": 25.0,
    "L4": 5.0,
    "T4": 2.0,
    "CPU": 0.1,
}


class BudgetExceededError(RuntimeError):
    """Raised when launching or continuing a session would exceed the campaign cap."""


@dataclass
class LedgerEntry:
    session_name: str
    hardware: str
    started_at: str
    ended_at: str
    elapsed_seconds: float
    rate_units_per_hr: float
    estimated_units: float
    cumulative_units: float
    active_sessions_after: int
    status: str
    notes: str = ""


def read_ledger(ledger_path: Path = LEDGER_PATH) -> list[dict[str, Any]]:
    if not ledger_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entries.append(json.loads(line))
    return entries


def get_cumulative_units(ledger_path: Path = LEDGER_PATH) -> float:
    entries = read_ledger(ledger_path)
    return round(sum(float(e.get("estimated_units", 0.0)) for e in entries), 4)


def check_budget_available(
    reserve_units: float = 0.25,
    hard_cap: float = HARD_CAMPAIGN_CAP_UNITS,
    ledger_path: Path = LEDGER_PATH,
) -> float:
    spent = get_cumulative_units(ledger_path)
    remaining = hard_cap - spent
    if remaining < reserve_units:
        raise BudgetExceededError(
            f"Colab compute-unit cap reached: spent={spent:.3f}u, cap={hard_cap:.3f}u, "
            f"remaining={remaining:.3f}u < reserve={reserve_units:.3f}u"
        )
    return remaining


def estimate_units(elapsed_seconds: float, hardware: str = "A100") -> float:
    rate = HARDWARE_RATES.get(hardware.upper(), A100_UNITS_PER_HOUR)
    return round((max(0.0, elapsed_seconds) / 3600.0) * rate, 4)


def record_session_usage(
    *,
    session_name: str,
    hardware: str,
    started_epoch: float,
    ended_epoch: float | None = None,
    active_sessions_after: int = 0,
    status: str = "stopped",
    notes: str = "",
    ledger_path: Path = LEDGER_PATH,
) -> LedgerEntry:
    end_ts = ended_epoch if ended_epoch is not None else time.time()
    elapsed = max(0.0, end_ts - started_epoch)
    rate = HARDWARE_RATES.get(hardware.upper(), A100_UNITS_PER_HOUR)
    units = estimate_units(elapsed, hardware)
    prev_total = get_cumulative_units(ledger_path)
    cum_total = round(prev_total + units, 4)

    entry = LedgerEntry(
        session_name=session_name,
        hardware=hardware.upper(),
        started_at=datetime.fromtimestamp(started_epoch, tz=timezone.utc).isoformat(),
        ended_at=datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat(),
        elapsed_seconds=round(elapsed, 2),
        rate_units_per_hr=rate,
        estimated_units=units,
        cumulative_units=cum_total,
        active_sessions_after=active_sessions_after,
        status=status,
        notes=notes,
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry)) + "\n")
    return entry
