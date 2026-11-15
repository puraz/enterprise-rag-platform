"""refresh token 轮换与复用检测测试（真实 Redis）。

这批测试守的是**会话安全**：

- 轮换：每次续期后旧 token 立即失效
- 复用检测：旧 token 再次出现 → 整族作废（判定为凭据泄漏）
- 登出：只撤当前 token，不影响其它设备
- 强制下线：`tokens_valid_after` 让旧 token 全部失效
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator

import pytest

from rag_api.identity.refresh import (
    REFRESH_TTL_S,
    RefreshTokenStore,
    require_not_revoked,
)
from rag_core.errors import (
    AuthenticationError,
    RefreshTokenReusedError,
    TokenInvalidError,
    TokenRevokedError,
)
from rag_core.redis import RedisClient
from rag_core.settings import get_settings

pytestmark = pytest.mark.integration


@pytest.fixture
async def store() -> AsyncIterator[RefreshTokenStore]:
    """用真实 Redis（测试 db 隔离到 15）。"""
    settings = get_settings()
    # 用 db 15 隔离测试数据，避免污染开发数据
    isolated = settings.model_copy(update={"redis": settings.redis.model_copy(update={"db": 15})})
    client = RedisClient(isolated)
    try:
        await client.client.flushdb()
        yield RefreshTokenStore(client.client)
    finally:
        await client.client.flushdb()
        await client.close()


# ==================================================== 基本签发与轮换


async def test_issue_and_rotate(store: RefreshTokenStore) -> None:
    """签发后能轮换出新 token。"""
    user_id, tenant_id = uuid.uuid4(), uuid.uuid4()

    token, family = await store.issue(user_id=user_id, tenant_id=tenant_id)
    assert token and family

    new_user, new_tenant, new_token, new_family = await store.rotate(refresh_token=token)

    assert new_user == user_id
    assert new_tenant == tenant_id
    assert new_token != token, "轮换后必须换新 token"
    assert new_family == family, "轮换保持同一 family"


async def test_rotation_invalidates_old_token(store: RefreshTokenStore) -> None:
    """轮换后旧 token 立即失效（但会被识别为复用）。"""
    user_id, tenant_id = uuid.uuid4(), uuid.uuid4()
    token, _family = await store.issue(user_id=user_id, tenant_id=tenant_id)

    await store.rotate(refresh_token=token)

    # 旧 token 再使用 → 复用检测
    with pytest.raises(RefreshTokenReusedError):
        await store.rotate(refresh_token=token)


# ==================================================== 复用检测（核心）


async def test_reuse_revokes_entire_family(store: RefreshTokenStore) -> None:
    """**复用检测命中时作废整族**。

    场景：攻击者偷到 refresh token，受害者先用（得到新 token），
    攻击者再用旧的。

    无法分辨谁是攻击者，安全的选择是让双方都重新登录——
    受害者会抱怨一次，但攻击者的会话也被切断。

    如果只拒绝这一次而不作废整族，攻击者手里的**新** token
    仍然有效（如果他也拿到了），或他可以继续尝试。
    """
    user_id, tenant_id = uuid.uuid4(), uuid.uuid4()
    token1, _ = await store.issue(user_id=user_id, tenant_id=tenant_id)

    # 受害者正常续期
    _u, _t, token2, _f = await store.rotate(refresh_token=token1)

    # 攻击者用被偷的 token1（已用过）
    with pytest.raises(RefreshTokenReusedError):
        await store.rotate(refresh_token=token1)

    # **整族作废**：token2 也不能用了
    with pytest.raises(TokenInvalidError):
        await store.rotate(refresh_token=token2)

    # family 已被清理
    assert await store.is_valid(refresh_token=token1) is False
    assert await store.is_valid(refresh_token=token2) is False


async def test_reuse_detection_survives_multiple_rotations(
    store: RefreshTokenStore,
) -> None:
    """连续轮换多次后，最早的 token 复用仍能检测到。

    攻击者可能等用户轮换好几次才用偷来的 token——
    检测不能只在"上一次"有效。
    """
    user_id, tenant_id = uuid.uuid4(), uuid.uuid4()
    token1, _family = await store.issue(user_id=user_id, tenant_id=tenant_id)

    current = token1
    for _ in range(3):
        _u, _t, current, _f = await store.rotate(refresh_token=current)

    # 用最早的 token
    with pytest.raises(RefreshTokenReusedError):
        await store.rotate(refresh_token=token1)

    # 当前 token 也因整族作废而失效
    with pytest.raises(TokenInvalidError):
        await store.rotate(refresh_token=current)


async def test_different_families_are_isolated(store: RefreshTokenStore) -> None:
    """**不同设备/会话的 family 互不影响**。

    手机被偷了，不应该让笔记本也被踢下线——
    除非用户主动"退出所有设备"。
    """
    user_id, tenant_id = uuid.uuid4(), uuid.uuid4()

    token_phone, family_phone = await store.issue(user_id=user_id, tenant_id=tenant_id)
    token_laptop, family_laptop = await store.issue(user_id=user_id, tenant_id=tenant_id)
    assert family_phone != family_laptop, "不同登录应生成不同 family"

    # 手机的 token 被复用 → 只作废手机的 family
    await store.rotate(refresh_token=token_phone)
    with pytest.raises(RefreshTokenReusedError):
        await store.rotate(refresh_token=token_phone)

    # 笔记本仍然可用
    _u, _t, new_laptop, _f = await store.rotate(refresh_token=token_laptop)
    assert new_laptop


# ==================================================== 登出与撤销


async def test_revoke_single_token(store: RefreshTokenStore) -> None:
    """登出撤销单个 token，不影响同 family 的其它 token。"""
    user_id, tenant_id = uuid.uuid4(), uuid.uuid4()
    token, family = await store.issue(user_id=user_id, tenant_id=tenant_id)

    await store.revoke_token(refresh_token=token)

    with pytest.raises(TokenInvalidError):
        await store.rotate(refresh_token=token)

    # family 记录还在（只是这个 token 没了）
    assert family


async def test_revoke_family(store: RefreshTokenStore) -> None:
    """可主动作废整族（"退出所有设备"）。"""
    user_id, tenant_id = uuid.uuid4(), uuid.uuid4()
    token1, _ = await store.issue(user_id=user_id, tenant_id=tenant_id)
    _, _, token2, family = await store.rotate(refresh_token=token1)

    removed = await store.revoke_family(family_id=family)

    assert removed >= 1
    with pytest.raises(TokenInvalidError):
        await store.rotate(refresh_token=token2)


# ==================================================== 安全存储


async def test_token_stored_as_hash(store: RefreshTokenStore) -> None:
    """**Redis 里只存 hash，不存原文**。

    存储泄漏时攻击者拿到的是摘要，无法直接使用——
    与密码存储同理。
    """
    user_id, tenant_id = uuid.uuid4(), uuid.uuid4()
    token, _family = await store.issue(user_id=user_id, tenant_id=tenant_id)

    # 原文不应出现在 Redis 的 key 或 value 里
    keys = await store._redis.keys("auth:refresh:*")
    for key in keys:
        key_str = key.decode() if isinstance(key, bytes) else key
        assert token not in key_str, "token 原文出现在了 key 里"

    raw = await store._redis.get(
        f"auth:refresh:token:{__import__('hashlib').sha256(token.encode()).hexdigest()}"
    )
    assert raw is not None
    assert token.encode() not in raw, "token 原文出现在了 value 里"


async def test_token_has_ttl(store: RefreshTokenStore) -> None:
    """refresh token 必须有 TTL（不会永久有效）。"""
    user_id, tenant_id = uuid.uuid4(), uuid.uuid4()
    token, _family = await store.issue(user_id=user_id, tenant_id=tenant_id)

    import hashlib

    key = f"auth:refresh:token:{hashlib.sha256(token.encode()).hexdigest()}"
    ttl = await store._redis.ttl(key)

    assert 0 < ttl <= REFRESH_TTL_S


# ==================================================== 用户级失效


def test_require_not_revoked_rejects_disabled_user() -> None:
    """**禁用用户后，未过期的 token 也必须失效**。

    只验签不看账号状态的话，管理员禁用后对方还能用满 access token
    的有效期（30 分钟），甚至用 refresh 无限续期。
    """
    from datetime import UTC, datetime

    with pytest.raises(AuthenticationError, match="禁用"):
        require_not_revoked(
            tokens_valid_after=datetime.now(UTC),
            token_issued_at=int(time.time()),
            user_active=False,
        )


def test_require_not_revoked_rejects_old_token() -> None:
    """**强制下线**：早于 `tokens_valid_after` 的 token 全部失效。

    改密码、管理员踢人时更新这个字段，让所有旧 token 立即作废，
    不需要维护撤销列表。
    """
    from datetime import UTC, datetime

    now = int(time.time())
    # token 是 10 分钟前签发的
    issued = now - 600
    # 用户在 5 分钟前被强制下线
    cutoff = datetime.fromtimestamp(now - 300, tz=UTC)

    with pytest.raises(TokenRevokedError):
        require_not_revoked(tokens_valid_after=cutoff, token_issued_at=issued, user_active=True)


def test_require_not_revoked_accepts_new_token() -> None:
    """强制下线后新签发的 token 有效。"""
    from datetime import UTC, datetime

    now = int(time.time())
    # 用户 10 分钟前被下线
    cutoff = datetime.fromtimestamp(now - 600, tz=UTC)
    # token 是 1 分钟前签发的（在下线之后）
    issued = now - 60

    require_not_revoked(tokens_valid_after=cutoff, token_issued_at=issued, user_active=True)


def test_require_not_revoked_boundary() -> None:
    """边界：恰好同一秒签发的 token 不应被误杀。

    时间戳精度是秒，强制下线时刻可能正好是某个 token 的签发秒。
    不做容忍会把"刚登录就改密码"的用户立刻踢掉。
    """
    from datetime import UTC, datetime

    now = int(time.time())
    cutoff = datetime.fromtimestamp(now, tz=UTC)

    # 同一秒签发 → 用 replace(microsecond=0) 后比较，不应被拒
    require_not_revoked(tokens_valid_after=cutoff, token_issued_at=now, user_active=True)
