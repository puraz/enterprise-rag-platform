"""PII 脱敏（重导出 core 的实现）。

实现在 `rag_core.pii`：worker 也要用，而 worker 不能 import api。
这里保留模块路径是为了让既有 `from rag_api.security.pii import ...` 继续可用。
"""

from rag_core.pii import MASK, contains_pii, redact_text, redact_value

__all__ = ["MASK", "contains_pii", "redact_text", "redact_value"]
