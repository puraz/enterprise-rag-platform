"""自省与生成：用 LLM 判断"材料能否回答问题"。

## 为什么不能靠检索分数拒答（实测证明）

`scripts/eval` 的 30 条查询给出了决定性数据：

| | 距离 |
|---|---|
| 有答案的**最大** top1 距离 | 0.527（"京沪广深算几线城市"） |
| 无答案的**最小** top1 距离 | 0.437（"食堂几点开饭"） |

两者区间**重叠**（0.437 < 0.527）。任何阈值都会误判：
- 设 0.45 → 误杀"京沪广深"（它其实有答案）；
- 设 0.55 → 放过"食堂几点"（它其实无答案，会诱导模型编）。

根因：embedding 相似度衡量"语义相近"，不衡量"能否回答问题"。
"京沪广深" 与 "一线城市（北京、上海、广州、深圳）" 语义上不够近
（缩写 vs 全称），但完全能回答问题；"食堂几点开饭" 与 "工资于每月
十五日发放" 都是"公司事务"语境，语义上不算远，但答不了。

**结论：拒答必须靠生成后的自省**，这是唯一能判断"材料是否足够"的机制。

## 自省为什么是独立的一次调用

理论上可以让生成模型自己说"我不知道"。但实测这个约束很弱——
模型被要求"不要编造"时仍会编造，因为它倾向于"给出有用的回答"。

独立调用换了任务框架（从"回答问题"变成"检查答案"），
且不需要维护对话连贯性，判断显著更准。

## 代价

每次问答多一次 LLM 调用（DeepSeek 约 500~1500ms、约 200 token）。
这是"可靠拒答"的成本，不可避免。可以按场景开关
（`RetrievalConfig.use_self_check`）：内部知识库可以关，
对外客服必须开。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from rag_core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SelfCheckResult:
    """自省/判断结果。

    同一个结构体服务两种判断（字段名按"能否回答"命名）：
    - **answerability**（生成前）：材料能否回答这个问题；
    - **support**（生成后）：答案中的事实是否都有材料依据。

    两者都用 `can_answer` 字段载 boolean，但语义不同——
    调用方必须清楚自己拿的是哪一个（见 `should_refuse` 的说明）。
    """

    can_answer: bool
    reason: str
    raw_output: str = ""
    """原始 LLM 输出，排障用（不进日志——可能含答案内容）。"""


def parse_self_check_output(raw: str) -> SelfCheckResult:
    """解析自省输出。

    同时接受 `{"can_answer": ...}`（answerability）与
    `{"supported": ...}`（support）两种键名，统一映射到 `can_answer`。
    LLM 严格按提示词输出时只有一种键，这种兼容让调用方不必
    为两种提示词各写一个解析器。

    LLM 输出 JSON 时不保证"纯 JSON"：常见包裹形式有
    ```json ... ```、前后加解释文字。所以先尝试直接解析，
    失败则从文本里提取第一个 `{...}` 片段。

    **解析失败时保守处理为 can_answer=False**：
    自省的作用是防止编造。解析不出来时"当成不能回答"会放弃一个
    可能正确的答案（体验损失），而"当成能回答"会放过一个可能编造的
    答案（正确性损失）。在知识库场景里后者严重得多。
    """
    text = raw.strip()

    # 去掉 ```json 围栏
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{"):
                text = candidate
                break

    data: dict[str, object] | None = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            data = parsed
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict):
                    data = parsed
            except json.JSONDecodeError:
                data = None

    if data is None:
        log.warning("self_check_parse_failed", output_length=len(raw))
        return SelfCheckResult(
            can_answer=False,
            reason="自省输出无法解析（保守判为不可回答）",
            raw_output=raw,
        )

    # 兼容两种键名：answerability 用 can_answer，support 用 supported
    if "can_answer" in data:
        can_answer = bool(data["can_answer"])
    else:
        can_answer = bool(data.get("supported", False))
    reason = str(data.get("reason", ""))[:200]
    return SelfCheckResult(can_answer=can_answer, reason=reason, raw_output=raw)


def should_refuse(*, materials: Sequence[str], answerability: SelfCheckResult | None) -> bool:
    """是否应该拒答。

    Args:
        materials: 检索到的材料。
        answerability: **生成前**的"材料能否回答问题"判断结果。
            None 表示未启用该判断。

    两个条件任一成立即拒答：
    1. 没有检索到任何材料；
    2. 判断结果表明材料不足以回答问题。

    ## 为什么参数是 answerability 而不是"答案是否有支撑"

    这两个是**不同的问题**，混淆会产生荒谬结果（A11 实测）：

        问"食堂几点开饭" → 模型生成"根据现有资料无法回答"
        → "答案是否有支撑"检查看到"没有事实性陈述" → 通过
        → 系统记为"正常回答"，误报率统计永远是 100%

    而"材料能否回答"在生成前就能判断，且直接对应业务语义
    （该不该答），所以拒答决策用它。

    `answerability=None` 时只能靠"有无材料"判断——不可靠
    （有材料不等于相关），但比完全不判断好。要可靠拒答必须启用它。
    """
    if not materials:
        return True
    return answerability is not None and not answerability.can_answer


REFUSAL_MESSAGE = "根据现有资料无法回答这个问题。请换个问法，或联系管理员补充相关资料。"


__all__ = [
    "REFUSAL_MESSAGE",
    "SelfCheckResult",
    "parse_self_check_output",
    "should_refuse",
]
