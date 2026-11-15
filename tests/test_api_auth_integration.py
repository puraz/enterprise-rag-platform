"""端到端认证链路测试（真实 PG + Redis）。

覆盖从 HTTP 请求到数据库写入的完整路径：

    dev-bootstrap → 建用户/租户/成员 → 签发 token
                  → 带 token 调 /me → 查库返回角色
                  → 验证 RLS 在真实请求路径下生效

**为什么需要这层测试**：单测验证了路由契约，但"首登要绕过 RLS
才能建成员关系"这类跨层问题只有真连库才能发现（我在实现时确实踩到了）。
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from rag_api.main import create_app
from rag_core.settings import get_settings

pytestmark = pytest.mark.integration


@pytest.fixture
def app_client():
    """带真实依赖的 app 客户端。

    用 `get_settings()` 而不是直接构造 Settings——
    直接构造**不会加载 .env**，会拿到默认密码（A2-2 踩过的坑）。
    """
    settings = get_settings()
    app = create_app(settings)
    with TestClient(app) as client:
        yield client


def _cleanup_dev_data() -> None:
    """清理 dev 测试数据（用管理连接绕过 RLS）。"""
    import asyncio

    from rag_core.db import Database

    async def _run() -> None:
        db = Database(get_settings())
        try:
            async with db.transaction_bypass_rls() as session:
                await session.execute(
                    text(
                        "DELETE FROM tenant_memberships WHERE user_id IN "
                        "(SELECT id FROM users WHERE email LIKE '%@dev.local')"
                    )
                )
                await session.execute(text("DELETE FROM tenants WHERE slug LIKE 'dev-%'"))
                await session.execute(text("DELETE FROM identity_links WHERE issuer = 'dev'"))
                await session.execute(text("DELETE FROM users WHERE email LIKE '%@dev.local'"))
        finally:
            await db.close()

    asyncio.run(_run())


@pytest.fixture(autouse=True)
def _cleanup_around() -> None:
    _cleanup_dev_data()
    yield
    _cleanup_dev_data()


def test_full_login_flow(app_client: TestClient) -> None:
    """完整链路：bootstrap → /me 返回正确的角色。"""
    subject = f"e2e-{uuid.uuid4().hex[:8]}"

    response = app_client.post("/api/v1/auth/dev-bootstrap", json={"credential": f"dev:{subject}"})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["is_new_user"] is True
    assert data["tenant_id"]

    me = app_client.get("/api/v1/me", headers={"Authorization": f"Bearer {data['access_token']}"})
    assert me.status_code == 200
    body = me.json()
    assert body["tenant_role"] == "owner", "bootstrap 建的用户应是对应租户的 owner"
    assert body["platform_role"] == "none"
    assert body["user_id"] == data["user_id"]


def test_second_bootstrap_is_idempotent(app_client: TestClient) -> None:
    """重复 bootstrap 不建重复用户/租户。"""
    subject = f"e2e-{uuid.uuid4().hex[:8]}"

    first = app_client.post("/api/v1/auth/dev-bootstrap", json={"credential": f"dev:{subject}"})
    second = app_client.post("/api/v1/auth/dev-bootstrap", json={"credential": f"dev:{subject}"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["user_id"] == second.json()["user_id"]
    assert first.json()["tenant_id"] == second.json()["tenant_id"]
    assert second.json()["is_new_user"] is False


def test_login_without_tenant_is_forbidden(app_client: TestClient) -> None:
    """新用户走 `/dev-login`（非 bootstrap）时因无租户被拒。

    这是有意的设计：自动开通租户必须是管理动作，
    否则任何人登录即拥有租户，配额完全失控。
    """
    subject = f"e2e-{uuid.uuid4().hex[:8]}"

    response = app_client.post("/api/v1/auth/dev-login", json={"credential": f"dev:{subject}"})
    assert response.status_code == 403
    assert response.json()["code"] == 20001
    assert response.json()["code_name"] == "AUTHZ_FORBIDDEN"


def test_token_bound_to_tenant(app_client: TestClient) -> None:
    """token 里的 tenant_id 决定 RLS 上下文。

    两个不同用户各自 bootstrap 出独立租户，各自的 `/me` 返回
    **自己的** tenant_id —— 证明 token 里的租户被正确使用。
    """
    s1, s2 = f"e2e-{uuid.uuid4().hex[:8]}", f"e2e-{uuid.uuid4().hex[:8]}"

    r1 = app_client.post("/api/v1/auth/dev-bootstrap", json={"credential": f"dev:{s1}"})
    r2 = app_client.post("/api/v1/auth/dev-bootstrap", json={"credential": f"dev:{s2}"})

    me1 = app_client.get(
        "/api/v1/me", headers={"Authorization": f"Bearer {r1.json()['access_token']}"}
    )
    me2 = app_client.get(
        "/api/v1/me", headers={"Authorization": f"Bearer {r2.json()['access_token']}"}
    )

    assert me1.json()["tenant_id"] == r1.json()["tenant_id"]
    assert me2.json()["tenant_id"] == r2.json()["tenant_id"]
    assert me1.json()["tenant_id"] != me2.json()["tenant_id"]


def test_disabled_user_token_rejected(app_client: TestClient) -> None:
    """token 有效但用户被禁用 → 401。

    单纯验签不够：员工离职后 token 可能还没过期，
    必须在每次请求时确认账号状态。
    """
    subject = f"e2e-{uuid.uuid4().hex[:8]}"
    r = app_client.post("/api/v1/auth/dev-bootstrap", json={"credential": f"dev:{subject}"})
    token = r.json()["access_token"]
    user_id = r.json()["user_id"]

    # 直接改库禁用用户
    import asyncio

    from rag_core.db import Database

    async def _disable() -> None:
        db = Database(get_settings())
        try:
            async with db.transaction_bypass_rls() as session:
                await session.execute(
                    text("UPDATE users SET status = 'disabled' WHERE id = :i"),
                    {"i": uuid.UUID(user_id)},
                )
        finally:
            await db.close()

    asyncio.run(_disable())

    me = app_client.get("/api/v1/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 401


def test_readiness_reports_all_dependencies(app_client: TestClient) -> None:
    """readiness 在真实环境下三项全 ok。"""
    response = app_client.get("/health/ready")
    assert response.status_code == 200
    checks = response.json()["checks"]
    assert checks["postgres"] == "ok"
    assert checks["redis"] == "ok"
    assert checks["pgvector"] == "ok"
