"""结构化日志：JSON 输出 + contextvars 透传 + 出口统一脱敏。

三条设计原则：

1. **只打结构化字段，不打拼接字符串**。
   `log.info("upload done", doc_id=..., bytes=...)` 而不是
   `log.info(f"upload {doc_id} done, {n} bytes")`。前者可被 Loki/ES 建索引
   和聚合，后者只能 grep。

2. **request_id / tenant_id / user_id 走 contextvars，不进函数签名**。
   否则每个业务函数都要多接一个 ctx 参数，污染所有分层。

3. **脱敏在序列化出口统一做，且必须同时覆盖 key 和 value**。
   按 key 判断挡不住 `log.info(f"connecting {dsn}")` 这种把密钥拼进
   消息体的写法；按 value 做正则才能兜住。这是本项目"敏感字段统一在
   出口脱敏，不靠各处自觉"的落地点。
"""

from __future__ import annotations

import contextvars
import logging
import re
import sys
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any, TextIO

import structlog
from structlog.typing import EventDict, WrappedLogger

from rag_core.settings import Settings

# ---------------------------------------------------------------- 脱敏

# 键名命中即整体替换为 MASK。键名先做归一化（小写 + 连字符转下划线），
# 这样 x-api-key / X-Api-Key / x_api_key 都能命中。
_SENSITIVE_EXACT = frozenset(
    {
        "authorization",
        "proxy_authorization",
        "cookie",
        "set_cookie",
        "password",
        "passwd",
        "secret",
        "api_key",
        "apikey",
        # 裸 token 是凭据；复数 tokens 才是计数量，所以不能加进后缀规则里
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "session_token",
        "client_secret",
        "private_key",
        "dsn",
        "database_url",
        "redis_url",
        "credential",
        "credentials",
    }
)

# 后缀命中同样替换。这里刻意只收单数 _token：
# 计费相关的 input_tokens / output_tokens / total_tokens 必须以复数形式幸存，
# 否则成本可观测性直接被脱敏掉——这是"脱敏不能过度"的典型边界。
_SENSITIVE_SUFFIXES = (
    "_password",
    "_passwd",
    "_secret",
    "_token",
    "_api_key",
    "_apikey",
    "_private_key",
)

MASK = "***"
_MAX_REDACT_DEPTH = 6

# 可注册的额外脱敏器：让上层（如 rag_api 的 PII 模块）注入自己的规则，
# 而 core 不需要 import 上层——保持依赖方向单向。
#
# 为什么用注册而不是直接 import：`rag_core` 是被 api/worker 依赖的底层，
# 反过来 import `rag_api` 会造成循环依赖，也破坏"core 不含业务概念"的定位。
# 注册表让 core 提供机制、上层提供策略。
_extra_redactors: list[Callable[[str], str]] = []


def register_text_redactor(func: Callable[[str], str]) -> None:
    """注册一个文本脱敏函数（在 core 的脱敏之后依次执行）。

    典型用法（在应用启动时）：

        from rag_core.logging import register_text_redactor
        from rag_api.security.pii import redact_text

        register_text_redactor(redact_text)

    **必须幂等**：重复注册同一个函数不会重复执行（用 `in` 判断）。
    多次调用 setup_logging 时容易重复注册。
    """
    if func not in _extra_redactors:
        _extra_redactors.append(func)


def clear_text_redactors() -> None:
    """清空注册的脱敏器（测试用）。"""
    _extra_redactors.clear()


# URL 里的 userinfo：scheme://user:password@host -> scheme://user:***@host
_URL_CREDENTIALS = re.compile(
    r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*://)(?P<user>[^:/@\s]+):(?P<password>[^@/\s]+)@"
)
# 裸 Bearer / Basic 凭据
_AUTH_SCHEME_VALUE = re.compile(
    r"(?i)\b(?P<scheme>bearer|basic)(?P<space>\s+)(?P<value>[\w\-._~+/]+=*)"
)


def _normalize_key(key: str) -> str:
    """键名归一化：小写并把连字符转下划线。"""
    return key.strip().lower().replace("-", "_")


def _is_sensitive_key(key: str) -> bool:
    """判断一个键名是否整个字段都属于敏感数据。"""
    normalized = _normalize_key(key)
    return normalized in _SENSITIVE_EXACT or normalized.endswith(_SENSITIVE_SUFFIXES)


def _redact_text(text: str) -> str:
    """对字符串值做模式级脱敏。

    两层：
    1. 内置规则（URL 凭据 / Bearer Token）——core 自带，无依赖；
    2. 注册的额外规则（如 rag_api 的 PII 脱敏）。

    顺序：内置先跑（密钥优先），再跑注册的（PII）。
    两组规则不重叠：密钥是随机串，PII 有固定格式。
    """
    redacted = _URL_CREDENTIALS.sub(r"\g<scheme>\g<user>:***@", text)
    redacted = _AUTH_SCHEME_VALUE.sub(r"\g<scheme>\g<space>***", redacted)
    for redactor in _extra_redactors:
        redacted = redactor(redacted)
    return redacted


def _redact_value(value: Any, depth: int = 0) -> Any:
    """递归脱敏任意值。

    深度上限防止自引用 dict 触发无限递归——日志层不该因为一个畸形
    上下文对象把整个请求拖死。
    """
    if depth > _MAX_REDACT_DEPTH:
        return "<max-depth>"

    if isinstance(value, str):
        return _redact_text(value)

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            result[key] = MASK if _is_sensitive_key(key) else _redact_value(raw_value, depth + 1)
        return result

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_redact_value(item, depth + 1) for item in value]

    return value


def redact_processor(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """structlog 处理器：序列化前统一脱敏。

    放在处理器链末端（渲染器之前）才有意义：此时所有字段都已合并完毕，
    包括 contextvars 带的、以及业务代码随手加的。
    """
    redacted: dict[str, Any] = {}
    for raw_key, raw_value in event_dict.items():
        key = str(raw_key)
        # event（日志正文）也要过正则：密钥最常见的泄漏方式就是拼进正文
        redacted[key] = MASK if _is_sensitive_key(key) else _redact_value(raw_value)
    return redacted


# ------------------------------------------------------------ 上下文字段

# 这些字段允许被透传到日志，其它字段不进 contextvars（避免误带大对象）
_CONTEXT_FIELDS = ("request_id", "tenant_id", "user_id", "doc_id", "trace_id")


def new_request_id() -> str:
    """生成新的 request_id。

    用 uuid4 的 32 位十六进制：无中心分配、碰撞概率可忽略、
    正好能塞进 HTTP header 且不需要额外字符集校验。
    """
    return uuid.uuid4().hex


@contextmanager
def request_context(
    request_id: str | None = None,
    *,
    tenant_id: str | None = None,
    user_id: str | None = None,
    doc_id: str | None = None,
    trace_id: str | None = None,
) -> Iterator[str]:
    """绑定一次请求的日志上下文，退出时精确还原。

    为什么要精确还原而不是整体 clear：请求里可能再嵌套一个子上下文
    （比如 Worker 处理单条任务时又发一次下游调用），clear 会把外层的
    request_id 一起抹掉，之后所有日志丢失关联。

    yield 的 request_id 用于回写给客户端响应头 `X-Request-ID`，
    用户报障时可以直接把这个值贴给客服。
    """
    rid = request_id or new_request_id()
    tokens = structlog.contextvars.bind_contextvars(
        request_id=rid,
        # 过滤 None：把"没这个值"和"值是 null"区分开。
        # 全量绑定会让每行日志都挂一堆 doc_id=None，真正有值时反而被淹没。
        **{
            key: value
            for key, value in {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "doc_id": doc_id,
                "trace_id": trace_id,
            }.items()
            if value is not None
        },
    )
    try:
        yield rid
    finally:
        # 用 bind 返回的 token 逐个还原，不能用 clear_contextvars()
        structlog.contextvars.reset_contextvars(**tokens)


def bind_context(**kwargs: Any) -> None:
    """在已存在的请求上下文里补充/覆盖字段（例如鉴权通过后补 tenant_id）。"""
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """清空当前上下文的全部绑定。仅用于测试与后台任务边界。"""
    structlog.contextvars.clear_contextvars()


def contextual_call[T](func: Callable[..., T], /, *args: Any, **kwargs: Any) -> Callable[[], T]:
    """把当前上下文的快照包装成一个零参可调用对象，供线程池执行。

    contextvars 在 asyncio 里会随 Task 自动继承，但 `loop.run_in_executor`
    和 `asyncio.to_thread` **不会**跨线程传播。解析、OCR 这类 CPU 密集任务
    必须丢线程池/进程池，不显式传播的话那些日志全都丢掉 request_id，
    线上排查时表现为"下游日志凭空断掉"。

    进程池根本没法传播（独立内存空间），那种场景必须把 request_id
    作为显式参数传进去——这是设计约束，不是可以偷懒的地方。
    """
    ctx = contextvars.copy_context()
    return lambda: ctx.run(func, *args, **kwargs)


# ---------------------------------------------------------------- 装配


def _build_renderer(log_format: str) -> Any:
    """按配置选渲染器。"""
    if log_format == "json":
        # ensure_ascii=False：中文保持可读，日志体积也更小
        return structlog.processors.JSONRenderer(ensure_ascii=False)
    # ConsoleRenderer 在非 TTY 环境下自动关闭颜色，CI 里不会出现 ANSI 乱码
    return structlog.dev.ConsoleRenderer(colors=False)


def setup_logging(settings: Settings, *, stream: TextIO | None = None) -> None:
    """装配 structlog + stdlib logging 桥接。

    为什么要桥接 stdlib：uvicorn、sqlalchemy、httpx 这些库直接用 stdlib
    logging 打日志。不桥接的话，同一进程里会同时出现 JSON 行和裸文本行，
    日志采集器解析一半失败。

    可重复调用（每次重建 handler），便于测试和配置热加载。
    """
    log_format = settings.resolved_log_format
    level = getattr(logging, settings.log_level, logging.INFO)

    # 共享处理器：structlog 自己的调用和第三方 stdlib 记录都过这一条链
    shared_processors: list[Any] = [
        # 必须放最前：先把 contextvars 合并进 event_dict，
        # 后续处理器（含脱敏）才能看到 request_id 这些字段
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        # 在 except 块里打日志时自动带上堆栈，不需要每处手写 exc_info=True
        structlog.dev.set_exc_info,
        redact_processor,
    ]

    formatter = structlog.stdlib.ProcessorFormatter(
        # 第三方库的 LogRecord 走 foreign_pre_chain 补齐结构化字段
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            _build_renderer(log_format),
        ],
    )

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # 先清掉旧 handler，否则重复初始化会让每条日志打印多份
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    structlog.configure(
        processors=[
            *shared_processors,
            # 交给 ProcessorFormatter 统一渲染，与第三方日志走同一套格式
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        # 首次取 logger 后缓存，避免每个请求重复构造绑定链
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """取一个绑定好上下文的 logger。业务代码统一从这里拿。"""
    return structlog.stdlib.get_logger(name)
