"""上传 HTTP 端点的端到端测试（真实 PG + 本地存储）。

覆盖完整链路与安全边界：

- 五个端点（init / part / status / complete / abort）
- 秒传（同租户命中、跨租户不命中）
- 断点续传（查询缺失分片）
- 分片编号与大小校验
- 魔数校验在 complete 时生效
- hash 不匹配时作废
- 跨租户不能操作别人的会话
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from rag_api.main import create_app
from rag_core.settings import get_settings

pytestmark = pytest.mark.integration


@pytest.fixture
def client() -> Iterator[TestClient]:
    """带真实依赖的客户端。"""
    app = create_app(get_settings())
    with TestClient(app) as test_client:
        yield test_client


def _login(client: TestClient, subject: str) -> dict[str, str]:
    """登录并返回认证头。"""
    response = client.post("/api/v1/auth/dev-bootstrap", json={"credential": f"dev:{subject}"})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _cleanup() -> None:
    """清理测试数据（用管理连接绕过 RLS）。

    ## 为什么删除顺序这么长

    `B4` 之前上传链路止于 blob，旧版清理只删 `upload_sessions` + `blobs`
    就够。现在上传会继续建 `documents` / `document_versions` / `tasks`，
    而 `documents.created_by` / `knowledge_bases.created_by` 是
    `ON DELETE RESTRICT`——**用户行被引用时删不掉**，清理会抛
    `RestrictViolationError`，把整个测试文件染红。

    顺序：任务 → 文档（先解 current_version 引用）→ 版本 → 知识库
    → blob → 上传会话 → 成员 → 租户 → 身份 → 用户。

    ## 为什么按 `created_by → users.email` 定位而不是 slug LIKE

    旧版用 `slug LIKE 'dev-%'` 找租户。这会**误删真实的开发账号**
    （`dev-eval@eval.local`）及其数据——实测把开发环境的库清空过。
    改成"只删本测试建的邮箱域名下的租户"。
    """
    import asyncio

    from sqlalchemy import text

    from rag_core.db import Database

    # 本文件的 subject 形如 `up-{8位hex}`（见各处 _login 调用），
    # dev-bootstrap 生成的 slug 是 `dev-{subject}`。
    #
    # **不能用 `dev-%` 匹配**：真实开发账号（`dev-eval@eval.local`）
    # 也在 `dev-` 下，会被一起删掉。实测把开发环境的库清空过。
    _email_pattern = "up-%@dev.local"
    _slug_pattern = "dev-up-%"

    async def _run() -> None:
        db = Database(get_settings())
        try:
            async with db.transaction_bypass_rls() as session:
                # B4 之后上传会连带建这些表；顺序必须是子 → 父。
                #
                # 每条语句都写全（不用字符串拼接）：SQL 里没有插值，
                # 参数化传 email_pattern，静态检查与人工 review 都能直接读懂。
                for statement in (
                    "DELETE FROM tasks WHERE tenant_id IN ("
                    "  SELECT id FROM tenants WHERE id IN ("
                    "    SELECT tenant_id FROM tenant_memberships WHERE user_id IN ("
                    "      SELECT id FROM users WHERE email LIKE :email_pattern)))",
                    "UPDATE documents SET current_version_id = NULL WHERE tenant_id IN ("
                    "  SELECT id FROM tenants WHERE id IN ("
                    "    SELECT tenant_id FROM tenant_memberships WHERE user_id IN ("
                    "      SELECT id FROM users WHERE email LIKE :email_pattern)))",
                    "DELETE FROM documents WHERE tenant_id IN ("
                    "  SELECT id FROM tenants WHERE id IN ("
                    "    SELECT tenant_id FROM tenant_memberships WHERE user_id IN ("
                    "      SELECT id FROM users WHERE email LIKE :email_pattern)))",
                    "DELETE FROM document_versions WHERE tenant_id IN ("
                    "  SELECT id FROM tenants WHERE id IN ("
                    "    SELECT tenant_id FROM tenant_memberships WHERE user_id IN ("
                    "      SELECT id FROM users WHERE email LIKE :email_pattern)))",
                    "DELETE FROM knowledge_bases WHERE tenant_id IN ("
                    "  SELECT id FROM tenants WHERE id IN ("
                    "    SELECT tenant_id FROM tenant_memberships WHERE user_id IN ("
                    "      SELECT id FROM users WHERE email LIKE :email_pattern)))",
                ):
                    await session.execute(text(statement), {"email_pattern": _email_pattern})

                await session.execute(
                    text(
                        "DELETE FROM upload_sessions WHERE created_by IN "
                        "(SELECT id FROM users WHERE email LIKE :email_pattern)"
                    ),
                    {"email_pattern": _email_pattern},
                )
                await session.execute(
                    text(
                        "DELETE FROM blobs WHERE tenant_id IN "
                        "(SELECT id FROM tenants WHERE slug LIKE :slug_pattern)"
                    ),
                    {"slug_pattern": _slug_pattern},
                )
                await session.execute(
                    text(
                        "DELETE FROM tenant_memberships WHERE user_id IN "
                        "(SELECT id FROM users WHERE email LIKE :email_pattern)"
                    ),
                    {"email_pattern": _email_pattern},
                )
                await session.execute(
                    text(
                        "DELETE FROM tenants WHERE slug LIKE :slug_pattern "
                        "AND id NOT IN (SELECT tenant_id FROM tenant_memberships)"
                    ),
                    {"slug_pattern": _slug_pattern},
                )
                await session.execute(text("DELETE FROM identity_links WHERE issuer = 'dev'"))
                await session.execute(
                    text("DELETE FROM users WHERE email LIKE :email_pattern"),
                    {"email_pattern": _email_pattern},
                )
        finally:
            await db.close()

    asyncio.run(_run())


@pytest.fixture(autouse=True)
def _cleanup_around() -> Iterator[None]:
    _cleanup()
    yield
    _cleanup()


# ==================================================== 完整流程


def test_full_upload_flow(client: TestClient) -> None:
    """五步完整上传：init → part → status → complete。"""
    headers = _login(client, f"up-{uuid.uuid4().hex[:8]}")
    payload = b"%PDF-1.7\n" + b"test content " * 200
    content_hash = hashlib.sha256(payload).hexdigest()

    init = client.post(
        "/api/v1/uploads",
        headers=headers,
        json={
            "filename": "doc.pdf",
            "total_size": len(payload),
            "content_type": "application/pdf",
            "content_hash": content_hash,
        },
    )
    assert init.status_code == 201, init.text
    session = init.json()
    assert session["deduplicated"] is False
    upload_id = session["upload_id"]

    part = client.put(f"/api/v1/uploads/{upload_id}/parts/1", headers=headers, content=payload)
    assert part.status_code == 200, part.text
    assert part.json()["part_number"] == 1

    status = client.get(f"/api/v1/uploads/{upload_id}", headers=headers)
    assert status.status_code == 200
    assert status.json()["missing_parts"] == []

    complete = client.post(
        f"/api/v1/uploads/{upload_id}/complete",
        headers=headers,
        json={"content_hash": content_hash},
    )
    assert complete.status_code == 200, complete.text
    body = complete.json()
    assert body["size_bytes"] == len(payload)
    assert body["content_hash"] == content_hash


def test_dedup_on_second_upload(client: TestClient) -> None:
    """**秒传**：相同内容第二次上传直接返回 blob_id。"""
    headers = _login(client, f"up-{uuid.uuid4().hex[:8]}")
    payload = b"%PDF-1.7\ndedup test" * 50
    content_hash = hashlib.sha256(payload).hexdigest()

    # 第一次：正常上传
    init = client.post(
        "/api/v1/uploads",
        headers=headers,
        json={
            "filename": "a.pdf",
            "total_size": len(payload),
            "content_type": "application/pdf",
            "content_hash": content_hash,
        },
    )
    upload_id = init.json()["upload_id"]
    client.put(f"/api/v1/uploads/{upload_id}/parts/1", headers=headers, content=payload)
    first = client.post(
        f"/api/v1/uploads/{upload_id}/complete",
        headers=headers,
        json={"content_hash": content_hash},
    )
    assert first.status_code == 200
    blob_id = first.json()["blob_id"]

    # 第二次：秒传
    second = client.post(
        "/api/v1/uploads",
        headers=headers,
        json={
            "filename": "b.pdf",
            "total_size": len(payload),
            "content_type": "application/pdf",
            "content_hash": content_hash,
        },
    )
    assert second.status_code == 201
    assert second.json()["deduplicated"] is True
    assert second.json()["blob_id"] == blob_id
    assert second.json()["upload_id"] is None


def test_dedup_is_tenant_scoped(client: TestClient) -> None:
    """**秒传不能跨租户**（存在性侧信道）。"""
    headers_a = _login(client, f"up-a-{uuid.uuid4().hex[:6]}")
    headers_b = _login(client, f"up-b-{uuid.uuid4().hex[:6]}")
    payload = b"%PDF-1.7\ntenant scoped" * 50
    content_hash = hashlib.sha256(payload).hexdigest()

    # A 租户上传
    init = client.post(
        "/api/v1/uploads",
        headers=headers_a,
        json={
            "filename": "a.pdf",
            "total_size": len(payload),
            "content_type": "application/pdf",
            "content_hash": content_hash,
        },
    )
    upload_id = init.json()["upload_id"]
    client.put(f"/api/v1/uploads/{upload_id}/parts/1", headers=headers_a, content=payload)
    client.post(
        f"/api/v1/uploads/{upload_id}/complete",
        headers=headers_a,
        json={"content_hash": content_hash},
    )

    # B 租户上传相同内容 → 不能秒传
    second = client.post(
        "/api/v1/uploads",
        headers=headers_b,
        json={
            "filename": "b.pdf",
            "total_size": len(payload),
            "content_type": "application/pdf",
            "content_hash": content_hash,
        },
    )
    assert second.status_code == 201
    assert second.json()["deduplicated"] is False, "跨租户秒传是存在性侧信道"


# ==================================================== 断点续传


def test_resume_after_partial_upload(client: TestClient) -> None:
    """断点续传：查询已传分片，只补传缺失的。"""
    from rag_core.settings import get_settings as _gs

    settings = _gs()
    part_size = settings.storage.part_size
    # 两片的数据
    payload = b"%PDF-1.7\n" + b"A" * (part_size - 9) + b"tail"
    content_hash = hashlib.sha256(payload).hexdigest()

    headers = _login(client, f"up-{uuid.uuid4().hex[:8]}")
    init = client.post(
        "/api/v1/uploads",
        headers=headers,
        json={
            "filename": "resume.pdf",
            "total_size": len(payload),
            "content_type": "application/pdf",
            "content_hash": content_hash,
        },
    )
    session = init.json()
    upload_id = session["upload_id"]
    assert session["total_parts"] == 2

    # 只传第一片
    first_part = payload[:part_size]
    client.put(f"/api/v1/uploads/{upload_id}/parts/1", headers=headers, content=first_part)

    # 查询 → 应该显示缺第 2 片
    status = client.get(f"/api/v1/uploads/{upload_id}", headers=headers)
    assert status.status_code == 200
    body = status.json()
    assert body["missing_parts"] == [2]
    assert [p["part_number"] for p in body["uploaded_parts"]] == [1]

    # 补传第二片后完成
    client.put(f"/api/v1/uploads/{upload_id}/parts/2", headers=headers, content=payload[part_size:])
    complete = client.post(
        f"/api/v1/uploads/{upload_id}/complete",
        headers=headers,
        json={"content_hash": content_hash},
    )
    assert complete.status_code == 200


# ==================================================== 校验与拒绝


def test_missing_parts_cannot_complete(client: TestClient) -> None:
    """分片不全时 complete 必须失败。"""
    part_size = get_settings().storage.part_size
    payload = b"%PDF-1.7\n" + b"x" * (part_size + 10)
    headers = _login(client, f"up-{uuid.uuid4().hex[:8]}")

    init = client.post(
        "/api/v1/uploads",
        headers=headers,
        json={
            "filename": "gap.pdf",
            "total_size": len(payload),
            "content_type": "application/pdf",
            "content_hash": None,
        },
    )
    upload_id = init.json()["upload_id"]

    complete = client.post(f"/api/v1/uploads/{upload_id}/complete", headers=headers, json={})
    assert complete.status_code == 400
    assert "未上传" in complete.json()["message"]


def test_part_number_zero_rejected(client: TestClient) -> None:
    """**分片编号从 1 开始**，传 0 必须拒绝。"""
    headers = _login(client, f"up-{uuid.uuid4().hex[:8]}")
    init = client.post(
        "/api/v1/uploads",
        headers=headers,
        json={
            "filename": "x.pdf",
            "total_size": 100,
            "content_type": "application/pdf",
            "content_hash": None,
        },
    )
    upload_id = init.json()["upload_id"]

    response = client.put(
        f"/api/v1/uploads/{upload_id}/parts/0", headers=headers, content=b"x" * 100
    )
    assert response.status_code == 400


def test_wrong_part_size_rejected(client: TestClient) -> None:
    """分片大小不符必须拒绝。"""
    part_size = get_settings().storage.part_size
    headers = _login(client, f"up-{uuid.uuid4().hex[:8]}")
    payload_size = part_size * 2

    init = client.post(
        "/api/v1/uploads",
        headers=headers,
        json={
            "filename": "x.pdf",
            "total_size": payload_size,
            "content_type": "application/pdf",
            "content_hash": None,
        },
    )
    upload_id = init.json()["upload_id"]

    response = client.put(
        f"/api/v1/uploads/{upload_id}/parts/1", headers=headers, content=b"tooshort"
    )
    assert response.status_code == 400
    assert "大小应为" in response.json()["message"]


def test_magic_mismatch_rejected_at_complete(client: TestClient) -> None:
    """**核心安全测试**：声明 pdf 实际传非 pdf，complete 时拒绝。

    initiate 阶段没有内容只能校验扩展名；complete 时有完整文件，
    必须校验魔数——否则用户能把任意文件伪装成 pdf 入库。
    """
    headers = _login(client, f"up-{uuid.uuid4().hex[:8]}")
    payload = b"this is plain text, not a pdf at all" * 20

    init = client.post(
        "/api/v1/uploads",
        headers=headers,
        json={
            "filename": "fake.pdf",
            "total_size": len(payload),
            "content_type": "application/pdf",
            "content_hash": None,
        },
    )
    upload_id = init.json()["upload_id"]
    client.put(f"/api/v1/uploads/{upload_id}/parts/1", headers=headers, content=payload)

    complete = client.post(f"/api/v1/uploads/{upload_id}/complete", headers=headers, json={})
    assert complete.status_code == 415
    assert "文件内容与扩展名不符" in complete.json()["message"]


def test_hash_mismatch_invalidates_upload(client: TestClient) -> None:
    """**整体 hash 不匹配时作废上传**。

    分片 ETag 只能证明单片正确；客户端声明错误 hash、
    或传输损坏导致整体不一致时，必须拒绝入库。
    """
    headers = _login(client, f"up-{uuid.uuid4().hex[:8]}")
    payload = b"%PDF-1.7\ncontent" * 50
    wrong_hash = hashlib.sha256(b"different content").hexdigest()

    init = client.post(
        "/api/v1/uploads",
        headers=headers,
        json={
            "filename": "x.pdf",
            "total_size": len(payload),
            "content_type": "application/pdf",
            "content_hash": None,  # 不触发秒传
        },
    )
    upload_id = init.json()["upload_id"]
    client.put(f"/api/v1/uploads/{upload_id}/parts/1", headers=headers, content=payload)

    complete = client.post(
        f"/api/v1/uploads/{upload_id}/complete",
        headers=headers,
        json={"content_hash": wrong_hash},
    )
    assert complete.status_code == 400
    assert "校验失败" in complete.json()["message"]


def test_unsupported_extension_rejected_at_init(client: TestClient) -> None:
    """**扩展名白名单**：可执行文件在 initiate 就被拒绝。

    尽早拒绝，而不是等用户传完 2GB 才发现。
    """
    headers = _login(client, f"up-{uuid.uuid4().hex[:8]}")
    response = client.post(
        "/api/v1/uploads",
        headers=headers,
        json={
            "filename": "malware.exe",
            "total_size": 1000,
            "content_type": "application/octet-stream",
            "content_hash": None,
        },
    )
    assert response.status_code == 415


def test_oversized_file_rejected_at_init(client: TestClient) -> None:
    """超过上限的文件在 initiate 就被拒绝。"""
    headers = _login(client, f"up-{uuid.uuid4().hex[:8]}")
    response = client.post(
        "/api/v1/uploads",
        headers=headers,
        json={
            "filename": "big.pdf",
            "total_size": get_settings().storage.max_file_size + 1,
            "content_type": "application/pdf",
            "content_hash": None,
        },
    )
    assert response.status_code == 413


# ==================================================== 中止与隔离


def test_abort_upload(client: TestClient) -> None:
    """中止上传后会话状态变 aborted，且不能再传分片。"""
    headers = _login(client, f"up-{uuid.uuid4().hex[:8]}")
    init = client.post(
        "/api/v1/uploads",
        headers=headers,
        json={
            "filename": "x.pdf",
            "total_size": 100,
            "content_type": "application/pdf",
            "content_hash": None,
        },
    )
    upload_id = init.json()["upload_id"]

    abort = client.delete(f"/api/v1/uploads/{upload_id}", headers=headers)
    assert abort.status_code == 204

    # 中止后再传分片应失败
    part = client.put(f"/api/v1/uploads/{upload_id}/parts/1", headers=headers, content=b"x" * 100)
    assert part.status_code == 409


def test_cannot_access_other_tenant_session(client: TestClient) -> None:
    """**跨租户不能操作别人的上传会话**。

    session_id 是 UUID，但 UUID 不是权限凭证——
    必须有租户过滤，否则猜中 ID 就能往别人的会话里塞分片。
    """
    headers_a = _login(client, f"up-a-{uuid.uuid4().hex[:6]}")
    headers_b = _login(client, f"up-b-{uuid.uuid4().hex[:6]}")

    init = client.post(
        "/api/v1/uploads",
        headers=headers_a,
        json={
            "filename": "x.pdf",
            "total_size": 100,
            "content_type": "application/pdf",
            "content_hash": None,
        },
    )
    upload_id = init.json()["upload_id"]

    # B 尝试访问 A 的会话
    status = client.get(f"/api/v1/uploads/{upload_id}", headers=headers_b)
    assert status.status_code == 404, "跨租户访问必须返回 404（不泄漏存在性）"

    part = client.put(f"/api/v1/uploads/{upload_id}/parts/1", headers=headers_b, content=b"x" * 100)
    assert part.status_code == 404


def test_unauthenticated_rejected(client: TestClient) -> None:
    """未认证的请求全部 401。"""
    for method, path in [
        ("post", "/api/v1/uploads"),
        ("get", f"/api/v1/uploads/{uuid.uuid4()}"),
        ("delete", f"/api/v1/uploads/{uuid.uuid4()}"),
    ]:
        response = getattr(client, method)(path)
        assert response.status_code == 401, f"{method} {path} 应该 401"
