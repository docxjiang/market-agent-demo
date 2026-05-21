from __future__ import annotations

import html
import re
from typing import Any

from src.news_evidence import normalized_sentiment_label, select_representative_news_evidence


UI_STYLES = """
<style>
.ui-stack {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.agent-block {
  color: #1f2937;
  font-size: 15px;
  line-height: 1.65;
}
.agent-block h2,
.agent-block h3 {
  margin: 0 0 10px 0;
  color: #111827;
  letter-spacing: 0;
}
.agent-block p {
  margin: 0 0 8px 0;
}
.agent-list {
  margin: 0 0 10px 0;
  padding-left: 20px;
}
.agent-list li {
  margin: 4px 0;
}
.agent-table {
  width: 100%;
  border-collapse: collapse;
  margin: 10px 0 14px 0;
  overflow: hidden;
  font-size: 14px;
}
.agent-table th {
  color: #374151;
  font-weight: 650;
  background: #f9fafb;
  border-bottom: 1px solid #e5e7eb;
  padding: 8px 10px;
}
.agent-table td {
  border-bottom: 1px solid #eef2f7;
  padding: 8px 10px;
  vertical-align: top;
}
.agent-table tr:last-child td {
  border-bottom: 0;
}
.agent-table .numeric-cell {
  text-align: right;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.agent-table .text-cell {
  text-align: left;
}
.news-card,
.announcement-card {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 12px 14px;
  background: #ffffff;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.card-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.card-title {
  margin-top: 8px;
  color: #111827;
  font-weight: 650;
  line-height: 1.4;
}
.card-summary {
  margin-top: 6px;
  color: #4b5563;
  line-height: 1.55;
}
.card-meta {
  color: #6b7280;
  font-size: 13px;
}
.sentiment-badge {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 2px 8px;
  font-size: 12px;
  font-weight: 650;
  line-height: 1.6;
  white-space: nowrap;
}
.sentiment-risk {
  color: #991b1b;
  background: #fee2e2;
  border: 1px solid #fecaca;
}
.sentiment-positive {
  color: #166534;
  background: #dcfce7;
  border: 1px solid #bbf7d0;
}
.sentiment-mixed {
  color: #92400e;
  background: #fef3c7;
  border: 1px solid #fde68a;
}
.sentiment-neutral {
  color: #374151;
  background: #f3f4f6;
  border: 1px solid #e5e7eb;
}
.semantic-badge {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 1px 8px;
  margin-left: 4px;
  font-size: 12px;
  font-weight: 700;
  line-height: 1.6;
  white-space: nowrap;
  border: 1px solid #e5e7eb;
}
.semantic-positive,
.semantic-strong,
.semantic-high {
  color: #166534;
  background: #dcfce7;
  border-color: #bbf7d0;
}
.semantic-negative,
.semantic-low {
  color: #991b1b;
  background: #fee2e2;
  border-color: #fecaca;
}
.semantic-neutral,
.semantic-divergent,
.semantic-medium,
.semantic-reflected {
  color: #374151;
  background: #f3f4f6;
  border-color: #e5e7eb;
}
.semantic-short {
  color: #92400e;
  background: #fef3c7;
  border-color: #fde68a;
}
.semantic-long {
  color: #1d4ed8;
  background: #dbeafe;
  border-color: #bfdbfe;
}
.source-link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  color: #1d4ed8;
  text-decoration: none;
  font-size: 13px;
  font-weight: 600;
  white-space: nowrap;
}
.source-link:hover {
  text-decoration: underline;
}
.decision-summary {
  border: 1px solid #dbe4ef;
  border-radius: 8px;
  background: #ffffff;
  padding: 16px;
  margin: 8px 0 18px 0;
  color: #111827;
}
.decision-summary-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  border-bottom: 1px solid #eef2f7;
  padding-bottom: 12px;
  margin-bottom: 12px;
}
.decision-summary-title {
  font-size: 15px;
  font-weight: 700;
  color: #374151;
}
.decision-summary-state {
  margin-top: 6px;
  font-size: 24px;
  font-weight: 700;
  line-height: 1.2;
}
.decision-summary-line {
  margin-top: 8px;
  color: #4b5563;
  line-height: 1.55;
}
.decision-summary-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}
.decision-summary-panel h4 {
  margin: 0 0 8px 0;
  color: #111827;
  font-size: 14px;
}
.decision-summary-panel ul {
  margin: 0;
  padding-left: 18px;
}
.decision-summary-panel li {
  margin: 5px 0;
  color: #374151;
  line-height: 1.55;
}
@media (max-width: 760px) {
  .decision-summary-grid {
    grid-template-columns: 1fr;
  }
}
</style>
"""


SENTIMENT_LABELS = {
    "risk": "risk",
    "positive": "positive",
    "neutral": "neutral",
}


def sentiment_badge_html(label: str | None) -> str:
    normalized = normalized_sentiment_label(label)
    css_label = normalized if normalized in SENTIMENT_LABELS else "neutral"
    text = SENTIMENT_LABELS.get(normalized, normalized)
    return f'<span class="sentiment-badge sentiment-{css_label}">{_escape(text)}</span>'


def build_news_card_html(item: dict[str, Any]) -> str:
    title = _escape(str(item.get("title") or "未命名新闻"))
    summary = _escape(str(item.get("summary") or ""))
    source = _escape(str(item.get("source") or "未知来源"))
    date = _escape(str(item.get("date") or "未知日期"))
    link = _external_link_html(str(item.get("url") or ""), "查看来源")
    summary_html = f'<div class="card-summary">{summary}</div>' if summary else ""
    return f"""
<article class="news-card">
  <div class="card-row">
    <div>{sentiment_badge_html(str(item.get("sentiment_hint") or "neutral"))}</div>
    <div>{link}</div>
  </div>
  <div class="card-title">{title}</div>
  <div class="card-meta">{source} · {date}</div>
  {summary_html}
</article>
"""


def build_news_cards_html(items: list[dict[str, Any]], limit: int | None = None) -> str:
    selected = items[:limit] if limit is not None else items
    if not selected:
        return f"{UI_STYLES}<p class=\"card-meta\">暂无可展示新闻。</p>"
    cards = "\n".join(build_news_card_html(item) for item in selected)
    return f'{UI_STYLES}<div class="ui-stack">{cards}</div>'


def build_announcement_card_html(item: dict[str, Any]) -> str:
    title = _escape(str(item.get("title") or "未命名公告"))
    source = _escape(str(item.get("source") or "未知来源"))
    date = _escape(str(item.get("date") or "未知日期"))
    source_link = _external_link_html(str(item.get("url") or ""), "公告页面")
    pdf_link = _external_link_html(str(item.get("pdf_url") or ""), "PDF")
    links = " ".join(link for link in [source_link, pdf_link] if link)
    return f"""
<article class="announcement-card">
  <div class="card-row">
    <div class="card-meta">{source} · {date}</div>
    <div>{links}</div>
  </div>
  <div class="card-title">{title}</div>
</article>
"""


def build_announcement_cards_html(items: list[dict[str, Any]], limit: int | None = None) -> str:
    selected = items[:limit] if limit is not None else items
    if not selected:
        return f"{UI_STYLES}<p class=\"card-meta\">暂无可展示公告。</p>"
    cards = "\n".join(build_announcement_card_html(item) for item in selected)
    return f'{UI_STYLES}<div class="ui-stack">{cards}</div>'


def build_agent_analysis_html(
    content: str,
    news_items: list[dict[str, Any]] | None = None,
    financial_reports: list[dict[str, Any]] | None = None,
) -> str:
    if news_items:
        prefix = content.split("代表性证据链：", 1)[0].strip()
        body = _simple_markdown_html(prefix)
        evidence_items = select_representative_news_evidence(news_items)
        evidence = f"<h3>代表性证据链</h3>{build_news_cards_html(evidence_items)}"
        return f'{UI_STYLES}<section class="agent-block">{body}{evidence}</section>'
    if financial_reports:
        body = _simple_markdown_html(content)
        reports = f"<h3>公告来源</h3>{build_announcement_cards_html(financial_reports, limit=8)}"
        return f'{UI_STYLES}<section class="agent-block">{body}{reports}</section>'
    return f'{UI_STYLES}<section class="agent-block">{_simple_markdown_html(content)}</section>'


def build_decision_summary_html(summary: dict[str, Any]) -> str:
    panels = [
        ("最重要的3个判断", summary.get("top_judgements", [])),
        ("关键利多", summary.get("positive_factors", [])),
        ("关键利空", summary.get("negative_factors", [])),
        (
            "后续触发条件",
            [
                *(f"偏积极：{item}" for item in summary.get("bullish_triggers", [])[:2]),
                *(f"偏消极：{item}" for item in summary.get("bearish_triggers", [])[:2]),
            ],
        ),
        ("已持有者关注", summary.get("holder_focus", [])),
        ("准备买入/卖出关注", [*summary.get("buyer_focus", [])[:2], *summary.get("seller_focus", [])[:2]]),
    ]
    panel_html = "\n".join(_decision_panel_html(title, items) for title, items in panels)
    return f"""
{UI_STYLES}
<section class="decision-summary">
  <div class="decision-summary-header">
    <div>
      <div class="decision-summary-title">AI决策摘要</div>
      <div class="decision-summary-state">{_escape(str(summary.get("state", "信息中性")))}</div>
      <div class="decision-summary-line">{_escape(str(summary.get("one_line", "")))}</div>
    </div>
  </div>
  <div class="decision-summary-grid">
    {panel_html}
  </div>
</section>
"""


def _simple_markdown_html(content: str) -> str:
    blocks: list[str] = []
    list_items: list[str] = []
    lines = content.splitlines()
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        line = raw_line.rstrip()
        if not line.strip():
            if list_items:
                blocks.append(_list_html(list_items))
                list_items = []
            index += 1
            continue
        stripped = line.strip()
        if _is_markdown_table_start(lines, index):
            if list_items:
                blocks.append(_list_html(list_items))
                list_items = []
            table_lines, index = _collect_markdown_table(lines, index)
            blocks.append(_table_html(table_lines))
        elif stripped.startswith("### "):
            if list_items:
                blocks.append(_list_html(list_items))
                list_items = []
            blocks.append(f"<h3>{_inline_markdown_html(stripped[4:])}</h3>")
            index += 1
        elif stripped.startswith("## "):
            if list_items:
                blocks.append(_list_html(list_items))
                list_items = []
            blocks.append(f"<h2>{_inline_markdown_html(stripped[3:])}</h2>")
            index += 1
        elif stripped.startswith("- "):
            list_items.append(stripped[2:])
            index += 1
        else:
            if list_items:
                blocks.append(_list_html(list_items))
                list_items = []
            blocks.append(f"<p>{_inline_markdown_html(stripped)}</p>")
            index += 1
    if list_items:
        blocks.append(_list_html(list_items))
    return "\n".join(blocks)


def _list_html(items: list[str]) -> str:
    lines = "\n".join(f"<li>{_inline_semantic_html(item)}</li>" for item in items)
    return f'<ul class="agent-list">{lines}</ul>'


def _decision_panel_html(title: str, items: list[str]) -> str:
    visible_items = [str(item) for item in items[:4]] or ["暂无明确线索。"]
    list_html = "".join(f"<li>{_inline_semantic_html(item)}</li>" for item in visible_items)
    return f'<div class="decision-summary-panel"><h4>{_escape(title)}</h4><ul>{list_html}</ul></div>'


def _inline_semantic_html(text: str) -> str:
    stripped = str(text).strip()
    match = _semantic_field_match(stripped)
    if not match:
        return _inline_markdown_html(stripped)
    field, value = match
    css_class = _semantic_class(value)
    return f'{_escape(field)}：<span class="semantic-badge {css_class}">{_escape(value)}</span>'


def _inline_markdown_html(text: str) -> str:
    value = str(text)
    pieces: list[str] = []
    last_index = 0
    for match in re.finditer(r"\*\*(.+?)\*\*", value):
        pieces.append(_escape(value[last_index : match.start()]))
        pieces.append(f"<strong>{_escape(match.group(1))}</strong>")
        last_index = match.end()
    pieces.append(_escape(value[last_index:]))
    return "".join(pieces)


def _semantic_field_match(text: str) -> tuple[str, str] | None:
    for separator in ("：", ":"):
        if separator not in text:
            continue
        field, value = text.split(separator, 1)
        field = field.strip()
        value = value.strip(" ；。")
        if field in {
            "影响方向",
            "催化强度",
            "影响周期",
            "可信度",
            "是否已被股价反映",
            "严重度",
            "当前风险等级",
        }:
            return field, value
    return None


def _semantic_class(value: str) -> str:
    normalized = value.strip()
    if normalized in {"利多", "偏积极", "强", "高"}:
        return "semantic-positive" if normalized in {"利多", "偏积极"} else f"semantic-{_strength_class(normalized)}"
    if normalized in {"利空", "偏消极", "低"}:
        return "semantic-negative" if normalized in {"利空", "偏消极"} else "semantic-low"
    if normalized in {"中性", "分歧", "中", "中高", "部分反映", "大概率已反映", "尚未充分反映", "无法判断"}:
        if normalized == "分歧":
            return "semantic-divergent"
        if normalized in {"部分反映", "大概率已反映", "尚未充分反映", "无法判断"}:
            return "semantic-reflected"
        return "semantic-medium" if normalized in {"中", "中高"} else "semantic-neutral"
    if normalized == "短线":
        return "semantic-short"
    if normalized == "长期":
        return "semantic-long"
    if normalized == "中期":
        return "semantic-medium"
    return "semantic-neutral"


def _strength_class(value: str) -> str:
    if value == "强":
        return "strong"
    if value == "高":
        return "high"
    return "medium"


def _is_markdown_table_start(lines: list[str], index: int) -> bool:
    if not _is_table_row(lines[index].strip()):
        return False
    next_index = index + 1
    while next_index < len(lines) and not lines[next_index].strip():
        next_index += 1
    return next_index < len(lines) and _is_table_separator(lines[next_index].strip())


def _collect_markdown_table(lines: list[str], index: int) -> tuple[list[str], int]:
    table_lines: list[str] = []
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            lookahead = index + 1
            while lookahead < len(lines) and not lines[lookahead].strip():
                lookahead += 1
            if lookahead < len(lines) and _is_table_row(lines[lookahead].strip()):
                index += 1
                continue
            break
        if not _is_table_row(stripped):
            break
        table_lines.append(stripped)
        index += 1
    return table_lines, index


def _table_html(table_lines: list[str]) -> str:
    if len(table_lines) < 2:
        return "\n".join(f"<p>{_escape(line)}</p>" for line in table_lines)

    headers = _split_table_row(table_lines[0])
    alignment = _split_table_row(table_lines[1])
    numeric_columns = {
        index
        for index, marker in enumerate(alignment)
        if marker.strip().endswith(":")
    }
    body_rows = [_split_table_row(line) for line in table_lines[2:] if not _is_table_separator(line)]

    header_html = "".join(
        f'<th class="{"numeric-cell" if index in numeric_columns else "text-cell"}">{_inline_markdown_html(cell)}</th>'
        for index, cell in enumerate(headers)
    )
    body_html = "\n".join(
        "<tr>"
        + "".join(
            f'<td class="numeric-cell">{_inline_markdown_html(cell)}</td>'
            if column_index in numeric_columns
            else f'<td class="text-cell">{_inline_markdown_html(cell)}</td>'
            for column_index, cell in enumerate(row)
        )
        + "</tr>"
        for row in body_rows
    )
    return f'<table class="agent-table"><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>'


def _is_table_row(line: str) -> bool:
    return line.startswith("|") and line.endswith("|") and line.count("|") >= 2


def _is_table_separator(line: str) -> bool:
    if not _is_table_row(line):
        return False
    cells = _split_table_row(line)
    return bool(cells) and all(cell.replace(":", "").replace("-", "").strip() == "" and "---" in cell for cell in cells)


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _external_link_html(url: str, label: str) -> str:
    if not url:
        return ""
    href = _escape(url, quote=True)
    return f'<a class="source-link" href="{href}" target="_blank" rel="noopener noreferrer">{_escape(label)} ↗</a>'


def _escape(value: str, quote: bool = False) -> str:
    return html.escape(value, quote=quote)
