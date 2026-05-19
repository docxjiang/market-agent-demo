from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents import run_all_agents
from src.data_loader import load_market_data, load_news, load_portfolio
from src.report import build_report


def main() -> None:
    portfolio = load_portfolio(PROJECT_ROOT / "data" / "sample_portfolio.json")
    news_items = load_news(PROJECT_ROOT / "data" / "sample_news.json")
    market_df = load_market_data(PROJECT_ROOT / "data" / "sample_market.csv")
    agent_results = run_all_agents(portfolio, news_items, market_df)
    report = build_report(portfolio, market_df, agent_results)

    output_path = PROJECT_ROOT / "outputs" / "example_report.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    main()
