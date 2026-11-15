"""refresh token 轮换与复用检测。

## 为什么 refresh token 必须轮换

不轮换（旧 refresh token 一直有效）的风险：
攻击者偷到 refresh token 后可以**长期**冒充用户，而受害者毫无察觉——
因为受害者的 refresh token 仍然能用。

轮换：每次用 refresh 换新 access token 时，**同时作废旧 refresh 并签发新的**。
这样每条 refresh token 只能用一次。

## 复用检测（Rotation with Reuse Detection）

轮换还不够：攻击者可能与受害者**同时**持有同一个 refresh token。
- 受害者先用 → 得到新的；
- 攻击者再用旧的 → 此时旧 token 已作废。

**关键设计**：看到已作废的 refresh token 被使用时，不能简单返回 401。
要判定为"凭据已泄漏"，并**作废该用户的所有会话**（整条 token 族）。

为什么：无法分辨谁是攻击者谁是受害者。安全的选择是让双方都重新登录
（受害者会抱怨一次，但攻击者的会话也被切断）。

这就是所谓"token family"机制：一条 refresh token 衍生出的所有后代
共享一个 family_id，检测到复用时整族作废。

## 为什么存在 Redis 而不是数据库

refresh token 的校验发生在**每个请求**（access 过期后），
需要低延迟。但更重要的是：它是**短期状态**（7-30 天），
不需要永久保存，也不需要参与事务。

Redis 的 TTL 天然适合：token 过期后自动清理，不需要定时任务。

**存的是 hash 不是原文**：数据库/Redis 泄漏时，攻击者拿到的是
SHA-256 摘要，无法直接使用。这与密码存储同理。

## 与 `tokens_valid_after` 的关系

`users.tokens_valid_after`（A3-1）用于"强制下线"：
早于该时刻签发的所有 token 立即失效。

两者的分工：
- `tokens_valid_after`：**用户级**全局失效（改密码、管理员禁用）；
- refresh 轮换/复用检测：**单个会话**粒度的失效（某设备被踢）。

两者都要检查，缺一不可。
"""

from __future__ import annotations

import hashlib
import json
import secrets
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from redis.asyncio import Redis

from rag_core.errors import (
    AuthenticationError,
    RefreshTokenReusedError,
    TokenInvalidError,
    TokenRevokedError,
)
from rag_core.logging import get_logger

log = get_logger(__name__)

# refresh token 的有效期（秒）。30 天是"可接受的安全/体验平衡点"：
# 更短会频繁要求重新登录；更长会放大 token 泄漏的窗口。
REFRESH_TTL_S = 30 * 24 * 3600

# Redis key 前缀
_FAMILY_KEY = "auth:refresh:family:{family_id}"
# S105 是误报：这是 Redis key 模板，不是密码
_TOKEN_KEY = "auth:refresh:token:{token_hash}"  # noqa: S105


@dataclass(frozen=True, slots=True)
class RefreshTokenRecord:
    """一条 refresh token 的元数据（存在 Redis 里）。"""

    token_hash: str
    family_id: str
    user_id: str
    tenant_id: str
    issued_at: str
    used: bool = False
    """是否已被使用过。已使用的 token 再次出现 = 复用攻击。"""


class RefreshTokenStore:
    """refresh token 的签发、轮换与复用检测。"""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    @staticmethod
    def _hash(token: str) -> str:
        """token 只存 hash。

        为什么要 hash：Redis 泄漏时攻击者拿到的是摘要，
        无法直接用来换取 access token。
        """
        return hashlib.sha256(token.encode()).hexdigest()

    async def issue(
        self,
        *,
        user_id: uuid.UUID,
        tenant_id: uuid.UUID,
        family_id: str | None = None,
    ) -> tuple[str, str]:
        """签发一个新的 refresh token。

        Args:
            family_id: 续期时传入原 family，保持"同一会话链"的关联；
                首次登录时传 None，会生成新 family。

        Returns:
            `(token, family_id)`。token 是原文（只返回给客户端一次），
            Redis 里只存 hash。
        """
        token = secrets.token_urlsafe(48)
        resolved_family = family_id or uuid.uuid4().hex

        record = RefreshTokenRecord(
            token_hash=self._hash(token),
            family_id=resolved_family,
            user_id=str(user_id),
            tenant_id=str(tenant_id),
            issued_at=datetime.now(UTC).isoformat(),
        )

        # 两个 key：
        # 1. token → 记录（用于校验与检测复用）
        # 2. family → 记录列表（用于"整族作废"）
        token_key = _TOKEN_KEY.format(token_hash=record.token_hash)
        family_key = _FAMILY_KEY.format(family_id=resolved_family)

        # 用 asdict 而不是 __dict__：`slots=True` 的 dataclass 没有 __dict__
        await self._redis.set(token_key, json.dumps(asdict(record)), ex=REFRESH_TTL_S)
        await self._redis.sadd(family_key, record.token_hash)
        await self._redis.expire(family_key, REFRESH_TTL_S)

        return token, resolved_family

    async def rotate(self, *, refresh_token: str) -> tuple[uuid.UUID, uuid.UUID, str, str]:
        """用 refresh token 换一对新的 token。

        返回 `(user_id, tenant_id, new_refresh_token, family_id)`。

        Raises:
            TokenInvalidError: token 不存在或格式错
            TokenRevokedError: token 已被主动撤销
            RefreshTokenReusedError: **检测到复用**（整族已作废）
        """
        token_hash = self._hash(refresh_token)
        token_key = _TOKEN_KEY.format(token_hash=token_hash)

        raw = await self._redis.get(token_key)
        if raw is None:
            # token 不存在：可能是过期、伪造，或**已被使用过**（轮换时删除）。
            # 后一种情况是复用攻击的典型特征，但我们不知道它属于哪个 family
            # （因为记录已被删），只能返回通用的"无效"。
            raise TokenInvalidError("refresh token 无效或已过期")

        payload = json.loads(raw)
        record = RefreshTokenRecord(**payload)

        if record.used:
            # **复用检测命中**：这个 token 已经被用过一次。
            # 说明有第三方持有同一个 token —— 判定为凭据泄漏，
            # 作废整族（无法分辨谁是攻击者，安全的选择是都踢下线）。
            log.warning(
                "refresh_token_reuse_detected",
                user_id=record.user_id,
                family_id=record.family_id,
            )
            await self.revoke_family(family_id=record.family_id)
            raise RefreshTokenReusedError()

        # 标记为已用（不删除，以便下次出现时能检测到复用）
        # 保留期比原 TTL 短：过足够久后没人会再用旧 token，
        # 长期保留只会浪费内存
        record = RefreshTokenRecord(
            token_hash=record.token_hash,
            family_id=record.family_id,
            user_id=record.user_id,
            tenant_id=record.tenant_id,
            issued_at=record.issued_at,
            used=True,
        )
        # 复用检测窗口：保留 1 小时。攻击者通常会在短时间内使用偷来的 token。
        await self._redis.set(
            token_key,
            json.dumps(asdict(record)),
            ex=min(3600, REFRESH_TTL_S),
        )

        # 签发同 family 的新 token
        new_token, family_id = await self.issue(
            user_id=uuid.UUID(record.user_id),
            tenant_id=uuid.UUID(record.tenant_id),
            family_id=record.family_id,
        )

        return (
            uuid.UUID(record.user_id),
            uuid.UUID(record.tenant_id),
            new_token,
            family_id,
        )

    async def revoke_family(self, *, family_id: str) -> int:
        """作废一整条 token 族（所有派生 token）。

        返回被作废的 token 数。用于复用检测命中时"踢掉整条会话链"。
        """
        family_key = _FAMILY_KEY.format(family_id=family_id)
        token_hashes = await self._redis.smembers(family_key)

        removed = 0
        for token_hash in token_hashes:
            token_key = _TOKEN_KEY.format(
                token_hash=token_hash.decode() if isinstance(token_hash, bytes) else token_hash
            )
            removed += int(await self._redis.delete(token_key))
        await self._redis.delete(family_key)

        log.info("refresh_family_revoked", family_id=family_id, tokens=removed)
        return removed

    async def revoke_token(self, *, refresh_token: str) -> None:
        """撤销单个 refresh token（登出）。

        登出时只撤销当前 token，不动 family —— 用户在其它设备上
        仍然应该保持登录。
        """
        token_hash = self._hash(refresh_token)
        await self._redis.delete(_TOKEN_KEY.format(token_hash=token_hash))

    async def is_valid(self, *, refresh_token: str) -> bool:
        """只检查有效性，不消耗 token（用于诊断，不用于认证）。"""
        raw = await self._redis.get(_TOKEN_KEY.format(token_hash=self._hash(refresh_token)))
        if raw is None:
            return False
        record = RefreshTokenRecord(**json.loads(raw))
        return not record.used


def require_not_revoked(
    *, tokens_valid_after: datetime, token_issued_at: int, user_active: bool
) -> None:
    """检查用户级失效条件。

    两个必须检查的点：
    - `tokens_valid_after`：管理员"强制下线"后，早于该时刻的 token 全部失效；
    - `user_active`：用户被禁用后，即使 token 没过期也不能用。

    不检查这两个的后果：管理员禁用了账号，但对方还能继续用 30 分钟
    （access token 的有效期），甚至用 refresh 无限续期。
    """
    if not user_active:
        raise AuthenticationError("账号已被禁用")

    issued = datetime.fromtimestamp(token_issued_at, tz=UTC)
    # 留 1 秒容忍：同一秒内的边界情况不误判
    if issued < tokens_valid_after.replace(microsecond=0):
        raise TokenRevokedError("登录状态已失效，请重新登录")
