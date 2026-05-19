from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from xml.etree import ElementTree

import requests


GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
USER_AGENT = "Mozilla/5.0 (compatible; MarketAgentDemo/1.0)"
MIN_NEWS_COUNT = 50


def clean_text(value: str) -> str:
    text = repair_mojibake(html.unescape(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def repair_mojibake(value: str) -> str:
    if not value:
        return value
    markers = ("Ã", "Â", "Ö", "Ð", "Î", "Ä", "Ç", "¼", "£", "Ô", "Ê", "µ", "¾")
    if not any(marker in value for marker in markers):
        return value
    for encoding in ("gbk", "utf-8"):
        try:
            repaired = value.encode("latin1").decode(encoding)
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if repaired and repaired != value:
            return repaired
    return value


def build_queries(portfolio: dict[str, Any]) -> list[str]:
    queries: list[str] = []
    for item in portfolio.get("positions", []):
        symbol = item.get("symbol", "")
        name = item.get("name", "")
        sector = item.get("sector", "")
        queries.extend(
            [
                f'"{symbol}" stock market news when:30d',
                f'"{name}" finance news when:30d',
                f'"{sector}" market risk when:30d',
            ]
        )
    queries.extend(
        [
            "global stock market risk interest rates when:30d",
            "AI capital expenditure technology stocks when:30d",
            "electric vehicle price competition market news when:30d",
            "Hong Kong technology stocks market news when:30d",
        ]
    )
    return queries


def build_position_queries(position: dict[str, Any]) -> list[str]:
    symbol = position.get("symbol", "")
    name = position.get("name", "")
    sector = position.get("sector", "")
    return [
        f'"{symbol}" stock market news when:30d',
        f'"{name}" finance news when:30d',
        f'"{symbol}" "{name}" earnings market when:30d',
        f'"{sector}" market news "{symbol}" when:30d',
    ]


def build_market_queries() -> list[str]:
    return [
        "global stock market risk interest rates when:30d",
        "AI capital expenditure technology stocks when:30d",
        "electric vehicle price competition market news when:30d",
        "Hong Kong technology stocks market news when:30d",
    ]


def infer_related_symbols(title: str, summary: str, portfolio: dict[str, Any]) -> list[str]:
    text = f"{title} {summary}".lower()
    symbols: list[str] = []
    for item in portfolio.get("positions", []):
        symbol = item.get("symbol", "")
        name = item.get("name", "")
        name_key = name.lower().split(" ")[0] if name else ""
        if symbol.lower() in text or (name_key and name_key in text):
            symbols.append(symbol)
    return symbols


def infer_sentiment_hint(title: str, summary: str) -> str:
    text = f"{title} {summary}".lower()
    negative_words = [
        "risk",
        "falls",
        "drop",
        "cuts",
        "slump",
        "concern",
        "pressure",
        "probe",
        "warning",
        "下滑",
        "下降",
        "承压",
        "风险",
        "被查",
        "净卖出",
        "未完成",
        "分歧",
        "减少",
        "低于预期",
    ]
    positive_words = [
        "rises",
        "gain",
        "growth",
        "beats",
        "surge",
        "record",
        "upgrade",
        "strong",
        "增长",
        "增",
        "提价",
        "分红",
        "回购",
        "改善",
        "高于预期",
    ]
    if any(word in text for word in negative_words):
        return "risk"
    if any(word in text for word in positive_words):
        return "positive"
    return "neutral"


def parse_google_news_rss(xml_text: str, portfolio: dict[str, Any], query: str) -> list[dict[str, Any]]:
    root = ElementTree.fromstring(xml_text)
    items: list[dict[str, Any]] = []
    for node in root.findall("./channel/item"):
        title = clean_text(node.findtext("title", default=""))
        link = html.unescape(node.findtext("link", default="")).strip()
        pub_date = clean_text(node.findtext("pubDate", default=""))
        source_node = node.find("source")
        source = clean_text(source_node.text if source_node is not None and source_node.text else "Google News")
        description = html.unescape(node.findtext("description", default="")).strip()
        summary = clean_text(description)
        if not title:
            continue
        item_id = hashlib.sha1(f"{title}|{link}".encode("utf-8")).hexdigest()[:16]
        items.append(
            {
                "id": item_id,
                "date": pub_date,
                "source": source,
                "title": title,
                "summary": summary[:500],
                "url": link,
                "query": query,
                "related_symbols": infer_related_symbols(title, summary, portfolio),
                "sentiment_hint": infer_sentiment_hint(title, summary),
                "fetched_via": "google_news_rss",
            }
        )
    return items


def fetch_google_news(query: str, portfolio: dict[str, Any], timeout: int = 12) -> list[dict[str, Any]]:
    url = GOOGLE_NEWS_RSS.format(query=quote_plus(query))
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return parse_google_news_rss(response.text, portfolio, query)


def dedupe_news(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        key = item.get("url") or item.get("title", "").lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def build_fallback_news(portfolio: dict[str, Any], needed: int) -> list[dict[str, Any]]:
    topics = [
        ("利率预期变化影响成长股波动", "市场重新评估利率路径，成长类资产估值波动可能放大。", "risk"),
        ("AI 资本开支保持高位", "大型科技公司持续投入 AI 基础设施，市场关注成本和供应链变化。", "mixed"),
        ("电动车行业价格竞争延续", "价格竞争可能影响行业利润率和库存节奏。", "risk"),
        ("港股科技板块成交活跃", "资金面和政策预期变化可能带来板块轮动。", "neutral"),
        ("美元走势影响跨市场风险偏好", "汇率和利率变量可能影响全球权益市场定价。", "risk"),
    ]
    symbols = [item.get("symbol", "MARKET") for item in portfolio.get("positions", [])] or ["MARKET"]
    items: list[dict[str, Any]] = []
    for index in range(needed):
        topic, summary, sentiment = topics[index % len(topics)]
        symbol = symbols[index % len(symbols)]
        items.append(
            {
                "id": f"fallback_{index + 1:03d}",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "source": "Local Demo Fallback",
                "title": f"{topic}：{symbol} 相关演示新闻 {index + 1}",
                "summary": summary,
                "url": "",
                "query": "local_demo_fallback",
                "related_symbols": [symbol],
                "sentiment_hint": sentiment,
                "fetched_via": "local_fallback",
            }
        )
    return items


def fetch_portfolio_news(
    portfolio: dict[str, Any],
    min_count: int = MIN_NEWS_COUNT,
    max_per_query: int = 20,
) -> list[dict[str, Any]]:
    all_items: list[dict[str, Any]] = []
    positions = portfolio.get("positions", [])
    per_position_target = max(8, min_count // max(len(positions), 1) // 2)

    for position in positions:
        position_items: list[dict[str, Any]] = []
        for query in build_position_queries(position):
            try:
                position_items.extend(fetch_google_news(query, portfolio)[:max_per_query])
            except (requests.RequestException, ElementTree.ParseError):
                continue
            position_items = dedupe_news(position_items)
            if len(position_items) >= per_position_target:
                break
        all_items.extend(position_items[:per_position_target])
        all_items = dedupe_news(all_items)

    for query in build_market_queries():
        if len(all_items) >= min_count:
            break
        try:
            all_items.extend(fetch_google_news(query, portfolio)[:max_per_query])
        except (requests.RequestException, ElementTree.ParseError):
            continue
        all_items = dedupe_news(all_items)

    if len(all_items) < min_count:
        all_items.extend(build_fallback_news(portfolio, min_count - len(all_items)))

    return dedupe_news(all_items)[: max(min_count, len(all_items))]


def save_news(items: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
