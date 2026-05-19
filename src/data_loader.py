from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_portfolio(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_news(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_market_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
