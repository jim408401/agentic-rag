"""從問題推斷 metadata 過濾條件。

規章常有新舊版本並存；一般問題應只看現行版，但「2023 年的時候…」「舊版規定…」這類問題
需要看歷史版本。這裡先用簡單的關鍵字規則判斷，之後的 Query Rewriting 會改由 LLM 判斷。
"""

import re
from typing import Any

CURRENT = {"status": "現行"}
HISTORICAL = re.compile(r"舊版|以前|過去|之前|當時|廢止|歷史|沿革|(?:19|20)\d{2}\s*年")


def infer_filters(question: str) -> dict[str, Any]:
    return {} if HISTORICAL.search(question) else dict(CURRENT)
