from __future__ import annotations

from typing import Any


NEGATIVE_WORDS = (
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
)

POSITIVE_WORDS = (
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
    "预增",
    "量价齐升",
)


def normalized_sentiment_label(label: Any) -> str:
    normalized = str(label or "neutral").strip().lower()
    if normalized in {"risk", "negative", "bearish", "负面", "风险"}:
        return "risk"
    if normalized in {"positive", "bullish", "正面", "积极"}:
        return "positive"
    if normalized in {"neutral", "mixed", "medium", "moderate", "中性", "混合"}:
        return "neutral"
    return normalized or "neutral"


def classify_news_item_sentiment(item: dict[str, Any]) -> str:
    normalized = normalized_sentiment_label(item.get("sentiment_hint"))
    if normalized in {"risk", "positive"}:
        return normalized

    text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
    if any(word in text for word in POSITIVE_WORDS):
        return "positive"
    if any(word in text for word in NEGATIVE_WORDS):
        return "risk"
    return "neutral"


def select_representative_news_evidence(
    news_items: list[dict[str, Any]],
    per_sentiment: int = 3,
    max_total: int = 10,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    sentiment_order = ["neutral", "risk", "positive"]

    for sentiment in sentiment_order:
        bucket = [
            item
            for item in news_items
            if classify_news_item_sentiment(item) == sentiment
        ]
        for item in bucket[:per_sentiment]:
            selected.append(item)
            if len(selected) >= max_total:
                return selected

    if not selected:
        return news_items[:max_total]
    return selected
