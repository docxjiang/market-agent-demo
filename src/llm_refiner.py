from __future__ import annotations

from src.llm_client import LLMClient
from src.utils import ensure_disclaimer


SYSTEM_PROMPT = """你是一名严谨的A股研究助理，负责把多智能体规则分析结果改写成更完整、更具体的个人持仓信息解读报告。

硬性要求：
- 保留原报告中的事实、数字、新闻标题、公告线索和风险标签，不编造未出现的数据。
- 不输出买入、卖出、持有、加仓、减仓等交易建议。
- 不预测确定性涨跌。
- 分析要采用“结论-证据-推理-后续观察点”的结构。
- 技术观察必须解释收益、回撤、振幅、均线、量能与价格区间位置的含义。
- 风险提示必须拆成基本面、估值/情绪、治理/公告、技术面、持仓集中度。
- 输出 Markdown。
"""


def refine_report_with_llm(report: str, client: LLMClient | None = None) -> str:
    llm = client or LLMClient()
    user_prompt = f"""请在不改变事实的前提下，把下面这份规则智能体报告改写成更像A股分析师工作稿的完整报告。

原始报告：

{report}
"""
    refined = llm.chat(SYSTEM_PROMPT, user_prompt)
    return ensure_disclaimer(refined)
