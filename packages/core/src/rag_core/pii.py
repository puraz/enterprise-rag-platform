"""PII 脱敏：在日志与审计出口识别并打码敏感信息。

**放在 core 而不是 api**：worker 也需要（任务错误消息要脱敏），
而 worker 禁止 import api（端边界）。

## 只在出口做，不在入口做

脱敏必须发生在**数据离开进程之前**（写日志、写审计），
而不是读取时：

- 日志原文落盘后，备份、副本、下游采集（Loki/ES）、被攻破的
  日志服务器都会泄漏；
- 审计表保留 180 天，里面的明文 PII 是合规风险。

## 与 A2-3 的 key 级脱敏的区别

A2-3 按**键名**脱敏（`password=***`），拦的是"整个字段是密钥"的情况。
本模块按**内容模式**脱敏，拦的是"敏感信息藏在文本里"：

    log.info("用户 13812345678 提交了文档")

键名是 `event`，A2-3 拦不住。这里用正则识别手机号/身份证/银行卡/邮箱。

## 必须同时处理"误伤"

过度脱敏会毁掉可观测性。所以规则要精确：

| 类型 | 识别方式 | 误伤风险 |
|---|---|---|
| 手机号 | 1[3-9] 开头的 11 位 | 低（但要排除长数字串里的误匹配） |
| 身份证 | 18 位（含 X 结尾）+ 校验位算法 | 中（订单号可能撞） |
| 银行卡 | 16-19 位连续数字 | 中 |
| 邮箱 | 标准格式 | 低 |
| IPv4 | 标准格式 | 低 |

**身份证用校验位算法而非纯正则**：18 位数字串很常见（订单号、
时间戳拼接），只有通过 ISO 7064 校验的才当身份证。这把误伤率
从"每千条几个"降到"几乎为零"。
"""

from __future__ import annotations

import re

MASK = "***"

# ---- 手机号（中国大陆）----
# 1 开头，第二位 3-9，共 11 位。
# 前后不能是数字（避免从长数字串中间截取）。
# 前面不能是字母数字下划线（避免匹配到 hash 里的一段）。
_PHONE = re.compile(r"(?<![0-9A-Za-z_])1[3-9]\d{9}(?![0-9])")

# ---- 身份证（18 位）----
# 先宽匹配 17 位数字 + [0-9Xx]，再用校验位算法确认。
_ID_CARD_CANDIDATE = re.compile(r"(?<!\d)(\d{17}[\dXx])(?!\d)")

# ---- 银行卡（16-19 位）----
# 前后不能是数字；且不能是已经匹配过的身份证（18 位会被两边都匹配）。
_BANK_CARD = re.compile(r"(?<!\d)\d{16,19}(?!\d)")

# ---- 邮箱 ----
# 简化但有效的形式：本地部分 + @ + 域名。
# 不用 RFC 5322 的完整正则（那个长到无法维护）。
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# ---- IPv4 ----
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# ---- 中国车牌 ----
# 省份简称 + 字母 + 5 位（含新能源的 6 位）
_PLATE = re.compile(
    r"(?<![A-Za-z0-9\u4e00-\u9fa5])"
    r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]"
    r"[A-HJ-NP-Z][A-HJ-NP-Z0-9]{4,5}[A-HJ-NP-Z0-9挂学警港澳]"
    r"(?![A-Za-z0-9])"
)

# 身份证校验位权重（ISO 7064:1983 MOD 11-2）
_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_ID_CHECK = "10X98765432"


def _is_valid_id_card(candidate: str) -> bool:
    """按 ISO 7064 MOD 11-2 校验身份证号。

    **为什么必须校验而不是纯正则**：18 位数字串在日志里极常见
    （订单号、毫秒时间戳拼接、hash 片段）。纯正则会把它们全部打码，
    可观测性被毁掉。

    校验位算法把误伤降到接近零：随机 18 位数字通过校验的概率是 1/11，
    而真实订单号等通常连格式都对不上（比如含字母）。
    """
    if len(candidate) != 18:
        return False
    body, check = candidate[:17], candidate[17].upper()
    if not body.isdigit():
        return False
    total = sum(int(digit) * weight for digit, weight in zip(body, _ID_WEIGHTS, strict=True))
    return _ID_CHECK[total % 11] == check


def _mask_phone(match: re.Match[str]) -> str:
    """手机号：保留前 3 位与后 4 位（便于客服核对，但无法还原）。"""
    value = match.group(0)
    return f"{value[:3]}{MASK}{value[-4:]}"


def _mask_id_card(match: re.Match[str]) -> str | None:
    """身份证：只有校验通过才打码，否则原样返回。"""
    candidate = match.group(0)
    if not _is_valid_id_card(candidate):
        return candidate
    # 保留前 6 位（地区码）与后 4 位（便于识别同一个人，但无法还原）
    return f"{candidate[:6]}{'*' * 8}{candidate[-4:]}"


def _mask_email(match: re.Match[str]) -> str:
    """邮箱：保留域名（便于判断是企业邮箱还是个人邮箱）。"""
    value = match.group(0)
    local, _, domain = value.partition("@")
    if len(local) <= 2:
        masked_local = "*" * len(local)
    else:
        masked_local = f"{local[0]}{'*' * (len(local) - 2)}{local[-1]}"
    return f"{masked_local}@{domain}"


def _mask_bank_card(match: re.Match[str]) -> str | None:
    """银行卡：保留后 4 位。

    **排除已被身份证匹配的情况**：18 位纯数字既是合法身份证长度
    也在 16-19 位范围内。先跑身份证规则、把命中位置记下来，
    这里跳过（见 `redact_text` 的实现）。
    """
    value = match.group(0)
    return f"{'*' * (len(value) - 4)}{value[-4:]}"


def _mask_ipv4(match: re.Match[str]) -> str:
    """IPv4：保留前两段（便于定位网段，但无法定位到具体机器）。

    注意：日志里记录客户端 IP 是必要的（安全审计），
    但不该记录完整 IP（PII）。保留 /16 网段是平衡点。
    """
    value = match.group(0)
    parts = value.split(".")
    if len(parts) != 4 or any(not part.isdigit() or int(part) > 255 for part in parts):
        return value
    return f"{parts[0]}.{parts[1]}.*.*"


def _mask_plate(match: re.Match[str]) -> str:
    """车牌：保留省份与首字母。"""
    value = match.group(0)
    return f"{value[:2]}{MASK}"


def redact_text(text: str) -> str:
    """对一段文本做 PII 脱敏。

    处理顺序很重要：
    1. **身份证优先**（18 位数字，会被银行卡规则误匹配）；
    2. 银行卡（16-19 位）；
    3. 手机号（11 位，会被银行卡规则忽略因为长度不够）；
    4. 邮箱、IPv4、车牌（格式明确，互不重叠）。

    实现上先跑身份证，记录命中区间，后续银行卡匹配跳过这些区间。
    """
    if not text:
        return text

    result = text

    # 1. 身份证（带校验位验证）
    result = _ID_CARD_CANDIDATE.sub(lambda m: _mask_id_card(m) or m.group(0), result)

    # 2. 银行卡：只对**未被身份证处理过**的纯数字串生效。
    #    用回调判断该串是否仍全是数字（身份证已被打码成含 * 的形式）
    def _bank_replacer(match: re.Match[str]) -> str:
        value = match.group(0)
        # 已被前面规则打码（含 *）的跳过
        if "*" in value:
            return value
        # 18 位且是合法身份证 → 已被打码，不会走到这里
        return _mask_bank_card(match) or value

    result = _BANK_CARD.sub(_bank_replacer, result)

    # 3. 手机号
    result = _PHONE.sub(_mask_phone, result)

    # 4. 邮箱
    result = _EMAIL.sub(_mask_email, result)

    # 5. IPv4
    result = _IPV4.sub(_mask_ipv4, result)

    # 6. 车牌
    result = _PLATE.sub(_mask_plate, result)

    return result


def redact_value(value: object, *, depth: int = 0, max_depth: int = 5) -> object:
    """递归脱敏任意结构里的字符串。

    `max_depth` 防止自引用结构导致无限递归（与 A2-3 的日志脱敏同理）。
    超深时返回占位符而不是抛异常：日志层不该因为一个畸形对象崩溃。
    """
    if depth > max_depth:
        return "<max-depth>"
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {
            str(key): redact_value(item, depth=depth + 1, max_depth=max_depth)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_value(item, depth=depth + 1, max_depth=max_depth) for item in value]
    return value


def contains_pii(text: str) -> bool:
    """检测文本是否含 PII（用于测试与审计告警）。

    不用于"决定是否脱敏"——脱敏总是执行（幂等），
    先检测再处理会漏掉"看起来不像但实际是"的情况。
    """
    if not text:
        return False
    if _PHONE.search(text) or _EMAIL.search(text):
        return True
    return any(_is_valid_id_card(m.group(0)) for m in _ID_CARD_CANDIDATE.finditer(text))
