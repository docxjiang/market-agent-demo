from __future__ import annotations


DISCLAIMER = "免责声明：本系统仅用于课程演示和市场信息解读，不构成投资建议。"


def ensure_disclaimer(text: str) -> str:
    if "不构成投资建议" in text:
        return text
    return f"{text.rstrip()}\n\n{DISCLAIMER}"
