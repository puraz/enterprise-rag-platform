"""OIDC 授权流程的 state 存储。

## 为什么 state 必须存在服务端（而不是只交给前端）

`state` 的作用是防 CSRF：攻击者诱导受害者用自己的授权码回调本系统，
如果服务端不校验 state，受害者的账号就会被绑到攻击者的外部身份上。

两种存法的区别：

| 存法 | CSRF 防护 | 问题 |
|---|---|---|
| 前端存 state，回调时比对 localStorage | 弱 | XSS 可直接读；多标签页互相覆盖 |
| **服务端存 state，回调时消费**（本模块） | 强 | 需要 Redis；多副本共享 |

服务端存的额外好处：**一次性消费**。同一个 state 用第二次说明是重放，
必须拒绝——而前端比对无法发现重放。

## 为什么 state 里不含 code_verifier

PKCE 的前提是"verifier 只在发起授权的客户端手里"。如果服务端也存一份，
那服务端就成了"能替任何人完成授权"的中心，PKCE 的防护价值打折。
所以 verifier 只在浏览器里，随 exchange 请求传上来。

## TTL 与多标签页

state 的 TTL 是 10 分钟（IdP 授权码通常也就几分钟有效）。
多标签页并发登录各自生成 state，互不干扰——每个 state 独立一条记录。
"""

from __future__ import annotations

import json
import secrets
import uuid
from dataclasses import asdict, dataclass

from redis.asyncio import Redis

from rag_core.logging import get_logger

log = get_logger(__name__)

_STATE_KEY = "oidc:state:{state}"
STATE_TTL_S = 600


@dataclass(frozen=True, slots=True)
class OIDCStateRecord:
    """一次授权请求的服务端状态。"""

    state: str
    redirect_uri: str
    nonce: str
    """随 id_token 往返的 nonce（防 token 重放）。

    标准 OIDC 里 nonce 由前端生成并在校验 id_token 时比对。
    本系统的交换在服务端完成，所以由服务端生成并校验。
    """

    created_by_ip: str | None = None


class OIDCStateStore:
    """state 的签发与一次性消费。"""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def issue(self, *, redirect_uri: str, client_ip: str | None = None) -> OIDCStateRecord:
        """生成并保存一个新的 state。"""
        record = OIDCStateRecord(
            # 32 字节 URL-safe 随机串：足够抵抗猜测，且能安全放进 URL
            state=secrets.token_urlsafe(32),
            redirect_uri=redirect_uri,
            nonce=secrets.token_urlsafe(16),
            created_by_ip=client_ip,
        )
        await self._redis.set(
            _STATE_KEY.format(state=record.state),
            json.dumps(asdict(record)),
            ex=STATE_TTL_S,
        )
        return record

    async def consume(self, *, state: str) -> OIDCStateRecord:
        """取出并删除 state（一次性消费）。

        Raises:
            ValidationFailedError: state 不存在（过期 / 伪造 / 已使用）

        ## 为什么用 `getdel` 而不是"先 get 再 delete"

        两条命令之间有竞态：两个并发回调都读到 state 存在，都通过校验。
        虽然授权码本身也只能用一次，但"重放被接受"这个事实本身就会
        让审计日志出现两条看起来都正确的登录记录。
        """
        from rag_core.errors import ValidationFailedError

        key = _STATE_KEY.format(state=state)
        raw = await getdel(self._redis, key)
        if raw is None:
            log.info("oidc_state_invalid", reason="missing_or_expired")
            raise ValidationFailedError(
                "登录状态已失效，请重新发起登录",
                details={"reason": "state_invalid"},
            )

        payload = json.loads(raw)
        try:
            return OIDCStateRecord(**payload)
        except TypeError as exc:
            # 记录格式损坏（升级变更字段导致）：按失效处理，不 500
            log.warning("oidc_state_corrupt", error_type=type(exc).__name__)
            raise ValidationFailedError("登录状态已失效，请重新发起登录") from exc


async def getdel(redis: Redis, key: str) -> str | None:
    """原子地取值并删除，兼容没有 `GETDEL` 的旧版本 Redis。

    Redis 6.2 才有 `GETDEL`。Valkey/Redis 7 都有，但本地可能跑旧版本，
    所以降级用 Lua 脚本（Lua 里 GET + DEL 是原子的）。

    **返回值统一解码成 str**：redis-py 在 `decode_responses=False`（本项目
    的默认配置）下返回 `bytes`，而调用方要做 `json.loads`。
    在这里统一转换，避免每个调用点各写一遍 isinstance 判断。
    """
    try:
        raw = await redis.getdel(key)
    except AttributeError:
        script = """
        local value = redis.call('GET', KEYS[1])
        if value then redis.call('DEL', KEYS[1]) end
        return value
        """
        raw = await redis.eval(script, 1, key)

    if raw is None:
        return None
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return raw


def new_state_hint() -> uuid.UUID:
    """给日志用的可关联 ID（不含任何用户信息）。"""
    return uuid.uuid4()


__all__ = ["STATE_TTL_S", "OIDCStateRecord", "OIDCStateStore", "new_state_hint"]
