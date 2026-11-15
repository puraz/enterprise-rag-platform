"""安全层：PII 脱敏、提示词注入防护。

与 `rag_core.logging` 的脱敏分工：

| 模块 | 拦截什么 | 触发方式 |
|---|---|---|
| `rag_core.logging`（A2-3） | 密钥/密码（按**键名**） | 每条日志自动 |
| `rag_api.security.pii` | 手机/身份证/邮箱/IP（按**内容模式**） | 显式调用或审计出口 |
| `rag_api.security.prompt_guard` | 提示词注入（结构化分离 + 输出检测） | 构造 prompt 时 |

前两者都要有：键名匹配挡不住"敏感信息藏在消息文本里"；
内容匹配挡不住"整个字段是密码但值是随机串"。
"""

from rag_api.security.pii import contains_pii, redact_text, redact_value
from rag_api.security.prompt_guard import (
    build_system_prompt,
    check_output,
    scan_for_injection,
    wrap_untrusted,
)

__all__ = [
    "build_system_prompt",
    "check_output",
    "contains_pii",
    "redact_text",
    "redact_value",
    "scan_for_injection",
    "wrap_untrusted",
]
