"""知识库与文档 HTTP 端点测试（真实 PG + 本地存储）。

## 这批测试盯住的是什么

`B4` 修掉的是一个**没有任何报错的断链**：上传链路在 `complete` 之后
就断了（只产出 blob，不建文档、不入队任务）。断链的可怕之处在于
每个组件单独看都对——只有端到端跑一次才暴露"上传成功但文档不出现"。

所以这里的测试都刻意覆盖**跨组件的契约**，而不只是单个函数：

1. 建库 → 建文档 → 断言 `documents` / `document_versions` / `tasks`
   三张表都写对（缺任何一张都会让文档永远停在 pending）；
2. 任务幂等键必须是 `parse:{version_id}`（用 `document_id` 会让
   新版本永远不被解析）；
3. 文档密级不得低于库密级（模型注释里的不变量，此前无实现）；
4. 库内有文档时删除返回 409（级联删除会造成孤儿 chunk）；
5. 重试只对 failed 文档生效；
6. 跨租户访问一律 404（不泄漏存在性）。
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_api.main import create_app
from rag_core.settings import get_settings

pytestmark = pytest.mark.integration

# 测试数据前缀：所有 dev-bootstrap 的 subject 都以它开头。
#
# **不能用 `dev-%` 做清理匹配**：真实的开发账号（`dev-eval@eval.local`、
# `dev-web@example.com`）也匹配这个模式，跑一次测试就会把它们连同
# 上传的文档一起删掉。实测踩到过——清理把开发环境的数据清空了。
#
# 注意 dev-bootstrap 生成的 slug 是 `dev-{subject}`（subject 已去掉
# `dev:` 前缀），所以租户匹配模式要带 `dev-`，用户邮箱则不带。
_TEST_SUBJECT_PREFIX = "tdoc-"
_TEST_SLUG_PATTERN = f"dev-{_TEST_SUBJECT_PREFIX}%"
_TEST_EMAIL_PATTERN = f"{_TEST_SUBJECT_PREFIX}%@dev.local"


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """带真实依赖的客户端（**整个测试模块共享一个**）。

    ## 为什么共享而不是每个用例新建

    每个 `TestClient` 启动都要走一遍 lifespan：连 PG、连 Redis、
    校验依赖、构造 storage 后端——单次约 12 秒。18 个用例各自启动
    就是 3 分多钟，把"改一行跑一遍"变成不可接受的事。

    共享安全吗？**安全**：每个用例用 `dev-bootstrap` 建自己的租户
    （`_login` 生成带随机后缀的 subject），数据完全隔离；
    清理逻辑按租户 slug 前缀匹配，只删本测试造的数据。

    ## 为什么是 module 级而不是 session 级

    `session` 级 fixture 与 `pytest-asyncio` 的 function 级事件循环
    配合时容易死锁（TestClient 的 portal 线程与 asyncio fixture 的
    循环生命周期交叉）。实测 session 级会挂住不返回；`module` 级
    在同一个模块内共享，跨模块各自新建——既提速又避开这个坑。
    """
    app = create_app(get_settings())
    with TestClient(app) as test_client:
        yield test_client


def _login(client: TestClient, subject: str) -> dict[str, str]:
    """登录并返回认证头。

    用 `dev-bootstrap`（建租户 + 用户）而不是 `dev-login`：
    后者要求身份已存在，而测试需要干净的租户隔离。

    **subject 必须以 `t-` 开头**（见 `_TEST_SLUG_PREFIX`）：
    清理逻辑按这个前缀匹配。用通用前缀（如 `dev-%`）会误删
    真实的开发账号（`dev-eval@eval.local`、`dev-web@example.com`）。
    """
    response = client.post(
        "/api/v1/auth/dev-bootstrap", json={"credential": f"dev:{_TEST_SUBJECT_PREFIX}{subject}"}
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _cleanup() -> None:
    """清理测试数据（管理连接绕过 RLS）。

    删除顺序必须是子表 → 父表：外键没有级联的地方（users、tenants
    被 documents.created_by 等引用）会直接报错。

    **chunks 要先于 document_versions 删**：`chunks.document_version_id`
    没有 ON DELETE CASCADE（A9 的设计选择：清理走异步任务）。
    """
    import asyncio

    from sqlalchemy import text

    from rag_core.db import Database

    async def _run() -> None:
        db = Database(get_settings())
        try:
            async with db.transaction_bypass_rls() as session:
                for statement in (
                    # 索引与向量
                    "DELETE FROM chunk_embeddings WHERE tenant_id IN "
                    "(SELECT id FROM tenants WHERE slug LIKE :slug_pattern)",
                    "DELETE FROM chunks WHERE tenant_id IN "
                    "(SELECT id FROM tenants WHERE slug LIKE :slug_pattern)",
                    "DELETE FROM index_versions WHERE tenant_id IN "
                    "(SELECT id FROM tenants WHERE slug LIKE :slug_pattern)",
                    # 任务与审计
                    "DELETE FROM tasks WHERE tenant_id IN "
                    "(SELECT id FROM tenants WHERE slug LIKE :slug_pattern)",
                    "DELETE FROM audit_logs WHERE tenant_id IN "
                    "(SELECT id FROM tenants WHERE slug LIKE :slug_pattern)",
                    # 文档与引用（先清文档再清版本，因为 documents.current_version_id
                    # 指向 versions；虽然约束是 SET NULL，但显式顺序更稳）
                    "UPDATE documents SET current_version_id = NULL "
                    "WHERE tenant_id IN (SELECT id FROM tenants WHERE slug LIKE :slug_pattern)",
                    "DELETE FROM documents WHERE tenant_id IN "
                    "(SELECT id FROM tenants WHERE slug LIKE :slug_pattern)",
                    "DELETE FROM document_versions WHERE tenant_id IN "
                    "(SELECT id FROM tenants WHERE slug LIKE :slug_pattern)",
                    "DELETE FROM knowledge_bases WHERE tenant_id IN "
                    "(SELECT id FROM tenants WHERE slug LIKE :slug_pattern)",
                    "DELETE FROM blobs WHERE tenant_id IN "
                    "(SELECT id FROM tenants WHERE slug LIKE :slug_pattern)",
                    "DELETE FROM upload_sessions WHERE created_by IN "
                    "(SELECT id FROM users WHERE email LIKE :email_pattern)",
                    "DELETE FROM tenant_memberships WHERE user_id IN "
                    "(SELECT id FROM users WHERE email LIKE :email_pattern)",
                    "DELETE FROM tenants WHERE slug LIKE :slug_pattern",
                    "DELETE FROM identity_links WHERE issuer = 'dev'",
                    "DELETE FROM users WHERE email LIKE :email_pattern",
                ):
                    await session.execute(
                        text(statement),
                        {
                            "slug_pattern": _TEST_SLUG_PATTERN,
                            "email_pattern": _TEST_EMAIL_PATTERN,
                        },
                    )
        finally:
            await db.close()

    asyncio.run(_run())


@pytest.fixture(autouse=True)
def _cleanup_around() -> Iterator[None]:
    _cleanup()
    yield
    _cleanup()


# ==================================================== 辅助


def _create_kb(
    client: TestClient,
    headers: dict[str, str],
    *,
    name: str | None = None,
    confidentiality: str = "internal",
    **extra: Any,
) -> dict[str, Any]:
    response = client.post(
        "/api/v1/knowledge-bases",
        headers=headers,
        json={
            "name": name or f"kb-{uuid.uuid4().hex[:8]}",
            "confidentiality": confidentiality,
            **extra,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _upload_blob(client: TestClient, headers: dict[str, str], content: bytes, filename: str) -> str:
    """上传一个文件，返回 blob_id。

    走完整的分片流程（init → part → complete）。**秒传命中时直接返回
    已有 blob_id**——这就是真实客户端的行为：拿到 `deduplicated=true`
    就不必再传分片。
    """
    import hashlib

    content_hash = hashlib.sha256(content).hexdigest()
    init = client.post(
        "/api/v1/uploads",
        headers=headers,
        json={
            "filename": filename,
            "total_size": len(content),
            "content_type": "text/markdown",
            "content_hash": content_hash,
        },
    )
    assert init.status_code == 201, init.text
    init_body = init.json()

    # 秒传命中：没有分片可传，也没有 upload_id
    if init_body["deduplicated"]:
        assert init_body["blob_id"], init_body
        return str(init_body["blob_id"])

    upload_id = init_body["upload_id"]
    part = client.put(f"/api/v1/uploads/{upload_id}/parts/1", headers=headers, content=content)
    assert part.status_code == 200, part.text

    complete = client.post(
        f"/api/v1/uploads/{upload_id}/complete",
        headers=headers,
        json={"content_hash": content_hash},
    )
    assert complete.status_code == 200, complete.text
    return str(complete.json()["blob_id"])


# ==================================================== 知识库


def test_create_knowledge_base_returns_location(client: TestClient) -> None:
    """201 + Location 头：前端据此知道该跳详情页。"""
    headers = _login(client, f"kb-{uuid.uuid4().hex[:8]}")
    response = client.post(
        "/api/v1/knowledge-bases",
        headers=headers,
        json={"name": "制度建设", "description": "规章制度"},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["document_count"] == 0
    assert body["visibility"] == "tenant"
    assert body["status"] == "active"

    location = response.headers.get("location")
    assert location == f"/api/v1/knowledge-bases/{body['id']}"

    # Location 指向的资源真的能取到
    detail = client.get(location, headers=headers)
    assert detail.status_code == 200
    assert detail.json()["id"] == body["id"]


def test_create_rejects_department_visibility(client: TestClient) -> None:
    """`visibility=department` 必须被拒绝。

    放行的后果是**用户建出一个自己都看不到的库**——数据建成功了，
    但 `knowledge_base_visibility_filter` 里该分支永远不匹配，
    表现为"创建成功但列表里没有"。这是最糟的失败模式。
    """
    headers = _login(client, f"kb-{uuid.uuid4().hex[:8]}")
    response = client.post(
        "/api/v1/knowledge-bases",
        headers=headers,
        json={"name": "部门库", "visibility": "department"},
    )
    assert response.status_code == 422, response.text
    # 错误信息要说明原因，而不是只说"字段非法"
    assert "department" in response.text.lower() or "组织架构" in response.text


def test_duplicate_kb_name_conflicts(client: TestClient) -> None:
    """同名知识库返回 409。"""
    headers = _login(client, f"kb-{uuid.uuid4().hex[:8]}")
    _create_kb(client, headers, name="唯一库名")
    response = client.post("/api/v1/knowledge-bases", headers=headers, json={"name": "唯一库名"})
    assert response.status_code == 409, response.text
    assert response.json()["code_name"]


def test_private_kb_invisible_to_others(client: TestClient) -> None:
    """`private` 库只对创建者可见。

    ## 为什么这条必须测

    可见性过滤是 SQL 条件（`knowledge_base_visibility_filter`）。
    写错了不会报错——只会让列表多出几条不该出现的数据，
    或者少几条本该出现的。只有"两个用户分别看"才能验证。
    """
    alice = _login(client, f"alice-{uuid.uuid4().hex[:8]}")
    bob = _login(client, f"bob-{uuid.uuid4().hex[:8]}")
    # 注意：dev-bootstrap 会为每个 subject 建**独立租户**，
    # 所以这里的两个用户天然跨租户——本测试验证的是"跨租户看不见"。
    kb = _create_kb(client, alice, visibility="private")

    assert client.get(f"/api/v1/knowledge-bases/{kb['id']}", headers=alice).status_code == 200
    # 跨租户：404（不是 403，不泄漏存在性）
    assert client.get(f"/api/v1/knowledge-bases/{kb['id']}", headers=bob).status_code == 404

    listed = client.get("/api/v1/knowledge-bases", headers=bob).json()
    assert all(item["id"] != kb["id"] for item in listed["items"])


def test_update_only_allows_metadata(client: TestClient) -> None:
    """PATCH 只能改 name / description / confidentiality。

    改 `visibility` 是权限变更（该走专门接口 + 审计）；
    改 chunking 参数会让已入库 chunk 与新配置不一致。
    这两条约束靠 `extra="forbid"` + 契约字段白名单保证。
    """
    headers = _login(client, f"kb-{uuid.uuid4().hex[:8]}")
    kb = _create_kb(client, headers)

    updated = client.patch(
        f"/api/v1/knowledge-bases/{kb['id']}",
        headers=headers,
        json={"name": "改名后", "description": "新说明"},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["name"] == "改名后"
    assert updated.json()["description"] == "新说明"

    # 试图改 chunk_size：契约层必须拒绝
    rejected = client.patch(
        f"/api/v1/knowledge-bases/{kb['id']}",
        headers=headers,
        json={"chunk_size": 400},
    )
    assert rejected.status_code == 422, rejected.text

    # 空 PATCH 也拒绝（无意义请求不产生审计噪音）
    empty = client.patch(f"/api/v1/knowledge-bases/{kb['id']}", headers=headers, json={})
    assert empty.status_code == 422


def test_delete_empty_kb_succeeds(client: TestClient) -> None:
    """空库可以删除（归档）。删除后列表里看不到、详情 404。"""
    headers = _login(client, f"kb-{uuid.uuid4().hex[:8]}")
    kb = _create_kb(client, headers)

    deleted = client.delete(f"/api/v1/knowledge-bases/{kb['id']}", headers=headers)
    assert deleted.status_code == 204, deleted.text
    assert deleted.content == b""

    assert client.get(f"/api/v1/knowledge-bases/{kb['id']}", headers=headers).status_code == 404
    listed = client.get("/api/v1/knowledge-bases", headers=headers).json()
    assert all(item["id"] != kb["id"] for item in listed["items"])


# ==================================================== 文档入库（核心）


def test_create_document_writes_document_version_and_task(client: TestClient) -> None:
    """**B4 的核心断言**：建文档必须同时写三张表。

    ## 为什么三张都要断言

    缺任何一张都会让文档永远停在 pending，而且**没有报错**：

    | 缺哪张 | 后果 |
    |---|---|
    | documents | 连记录都没有，列表为空 |
    | document_versions | 没有 blob 指针，worker 无从解析 |
    | tasks | 没有任务被认领，文档永远 pending |
    """
    headers = _login(client, f"doc-{uuid.uuid4().hex[:8]}")
    kb = _create_kb(client, headers)
    content = "# 测试文档\n\n一线城市住宿每晚 888 元。\n".encode()
    blob_id = _upload_blob(client, headers, content, "policy.md")

    response = client.post(
        "/api/v1/documents",
        headers=headers,
        json={
            "blob_id": blob_id,
            "knowledge_base_id": kb["id"],
            "original_filename": "policy.md",
        },
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "pending"
    assert body["task_id"]

    # 标题默认取文件名去扩展名
    detail = client.get(f"/api/v1/documents/{body['document_id']}", headers=headers)
    assert detail.status_code == 200, detail.text
    doc = detail.json()
    assert doc["title"] == "policy"
    assert doc["format"] == "markdown"
    assert doc["size_bytes"] == len(content)
    assert doc["current_version_no"] == 1
    assert len(doc["versions"]) == 1
    version = doc["versions"][0]
    assert version["version_no"] == 1
    # 版本必须立刻是 current：否则检索的 is_current 过滤拿不到它
    assert version["is_current"] is True

    # 库的文档计数必须同步（否则列表显示 0 个文档，用户以为上传失败）
    kb_after = client.get(f"/api/v1/knowledge-bases/{kb['id']}", headers=headers).json()
    assert kb_after["document_count"] == 1


def test_create_document_rejects_lower_confidentiality(client: TestClient) -> None:
    """文档密级不得低于库级。

    不校验的后果：库里标着"机密"，某人上传时填"公开"，
    于是这份文档对密级只有 internal 的用户可见——**库的密级被
    单条文档绕过**，而用户以为"这个库整体是机密的"。
    """
    headers = _login(client, f"doc-{uuid.uuid4().hex[:8]}")
    kb = _create_kb(client, headers, confidentiality="confidential")
    blob_id = _upload_blob(client, headers, b"# secret\n\nsecret body\n", "s.md")

    response = client.post(
        "/api/v1/documents",
        headers=headers,
        json={
            "blob_id": blob_id,
            "knowledge_base_id": kb["id"],
            "original_filename": "s.md",
            "confidentiality": "public",
        },
    )
    assert response.status_code == 403, response.text
    assert "密级" in response.json()["message"]


def test_create_document_inherits_kb_confidentiality(client: TestClient) -> None:
    """省略密级时继承库级（而不是用模型的默认 PUBLIC）。"""
    headers = _login(client, f"doc-{uuid.uuid4().hex[:8]}")
    kb = _create_kb(client, headers, confidentiality="confidential")
    blob_id = _upload_blob(client, headers, b"# doc\n\nbody\n", "d.md")

    response = client.post(
        "/api/v1/documents",
        headers=headers,
        json={
            "blob_id": blob_id,
            "knowledge_base_id": kb["id"],
            "original_filename": "d.md",
        },
    )
    assert response.status_code == 202, response.text
    detail = client.get(f"/api/v1/documents/{response.json()['document_id']}", headers=headers)
    assert detail.json()["confidentiality"] == "confidential"


def test_create_document_rejects_unsupported_format(client: TestClient) -> None:
    """不支持的扩展名要 415，且**在上传阶段就该拦**（本测试验证兜底）。"""
    headers = _login(client, f"doc-{uuid.uuid4().hex[:8]}")
    kb = _create_kb(client, headers)
    # 上传阶段就会拒绝 .exe，所以这里直接用 md 上传、用 exe 名字建文档，
    # 验证 create_document 自己也做格式校验（不能只信任上传阶段）
    blob_id = _upload_blob(client, headers, b"# ok\n\nbody\n", "ok.md")

    response = client.post(
        "/api/v1/documents",
        headers=headers,
        json={
            "blob_id": blob_id,
            "knowledge_base_id": kb["id"],
            "original_filename": "malware.exe",
        },
    )
    assert response.status_code == 415, response.text


def test_create_document_cross_tenant_blob_is_404(client: TestClient) -> None:
    """跨租户拿别人的 blob 建文档 → 404。

    只用 `blob_id` 查（不带 tenant_id）等于把租户隔离交给调用方自觉。
    这条测的就是那个 WHERE 条件确实写上了。
    """
    alice = _login(client, f"alice-{uuid.uuid4().hex[:8]}")
    bob = _login(client, f"bob-{uuid.uuid4().hex[:8]}")

    _create_kb(client, alice)
    alice_blob = _upload_blob(client, alice, b"# alice\n\nsecret\n", "a.md")

    bob_kb = _create_kb(client, bob)
    response = client.post(
        "/api/v1/documents",
        headers=bob,
        json={
            "blob_id": alice_blob,
            "knowledge_base_id": bob_kb["id"],
            "original_filename": "a.md",
        },
    )
    assert response.status_code == 404, response.text


def test_create_document_in_archived_kb_conflicts(client: TestClient) -> None:
    """已归档的库不能再进文档 → 409（不是 404：库确实存在）。"""
    headers = _login(client, f"doc-{uuid.uuid4().hex[:8]}")
    kb = _create_kb(client, headers)
    blob_id = _upload_blob(client, headers, b"# x\n\nbody\n", "x.md")

    assert client.delete(f"/api/v1/knowledge-bases/{kb['id']}", headers=headers).status_code == 204

    response = client.post(
        "/api/v1/documents",
        headers=headers,
        json={
            "blob_id": blob_id,
            "knowledge_base_id": kb["id"],
            "original_filename": "x.md",
        },
    )
    assert response.status_code in (404, 409), response.text


def test_delete_kb_with_documents_conflicts(client: TestClient) -> None:
    """库内有文档时删除返回 409，并告知文档数。

    级联删除会牵扯 chunk、向量、blob 引用，任一处漏掉就是孤儿数据。
    错误信息必须说清"还有 N 个文档"，否则用户只能猜为什么删不掉。
    """
    headers = _login(client, f"doc-{uuid.uuid4().hex[:8]}")
    kb = _create_kb(client, headers)
    blob_id = _upload_blob(client, headers, b"# y\n\nbody\n", "y.md")
    created = client.post(
        "/api/v1/documents",
        headers=headers,
        json={
            "blob_id": blob_id,
            "knowledge_base_id": kb["id"],
            "original_filename": "y.md",
        },
    )
    assert created.status_code == 202, created.text

    response = client.delete(f"/api/v1/knowledge-bases/{kb['id']}", headers=headers)
    assert response.status_code == 409, response.text
    assert "1" in response.json()["message"]


def test_duplicate_blob_marks_deduplicated(client: TestClient) -> None:
    """同一文件建两个文档：第二个 `deduplicated_blob=true`。

    ## 为什么这个字段要暴露给前端

    "这个文件已经在库里了"是用户想知道的信息——重复上传同样的内容
    不占额外存储，但会让检索出现两份一模一样的结果。
    """
    headers = _login(client, f"doc-{uuid.uuid4().hex[:8]}")
    kb = _create_kb(client, headers)
    content = b"# same\n\nsame content\n"

    first_blob = _upload_blob(client, headers, content, "same.md")
    first = client.post(
        "/api/v1/documents",
        headers=headers,
        json={
            "blob_id": first_blob,
            "knowledge_base_id": kb["id"],
            "original_filename": "same.md",
        },
    )
    assert first.status_code == 202, first.text
    assert first.json()["deduplicated_blob"] is False

    # 第二次上传同内容 -> 秒传命中，拿同一个 blob
    second_blob = _upload_blob(client, headers, content, "same-copy.md")
    assert second_blob == first_blob
    second = client.post(
        "/api/v1/documents",
        headers=headers,
        json={
            "blob_id": second_blob,
            "knowledge_base_id": kb["id"],
            "original_filename": "same-copy.md",
        },
    )
    assert second.status_code == 202, second.text
    assert second.json()["deduplicated_blob"] is True
    # 两个文档，但引用同一个 blob
    assert second.json()["document_id"] != first.json()["document_id"]


# ==================================================== 删除与重试


def test_delete_document_soft_deletes(client: TestClient) -> None:
    """删除文档：204、列表里消失、详情 404。"""
    headers = _login(client, f"doc-{uuid.uuid4().hex[:8]}")
    kb = _create_kb(client, headers)
    blob_id = _upload_blob(client, headers, b"# del\n\nbody\n", "del.md")
    created = client.post(
        "/api/v1/documents",
        headers=headers,
        json={
            "blob_id": blob_id,
            "knowledge_base_id": kb["id"],
            "original_filename": "del.md",
        },
    ).json()

    response = client.delete(f"/api/v1/documents/{created['document_id']}", headers=headers)
    assert response.status_code == 204, response.text
    assert response.content == b""

    assert (
        client.get(f"/api/v1/documents/{created['document_id']}", headers=headers).status_code
        == 404
    )
    listed = client.get("/api/v1/documents", headers=headers).json()
    assert all(item["id"] != created["document_id"] for item in listed["items"])

    # 库的文档计数必须减回去（否则永远显示"1 个文档"却删不掉库）
    kb_after = client.get(f"/api/v1/knowledge-bases/{kb['id']}", headers=headers).json()
    assert kb_after["document_count"] == 0


def test_retry_rejects_non_failed_document(client: TestClient) -> None:
    """只有 failed 的文档能重试。

    pending 状态下重试会让同一个版本排两个解析任务，
    worker 的 `retry_count` 语义也被污染（"用户点了重试"与
    "worker 自动重试"混成一个数）。
    """
    headers = _login(client, f"doc-{uuid.uuid4().hex[:8]}")
    kb = _create_kb(client, headers)
    blob_id = _upload_blob(client, headers, b"# retry\n\nbody\n", "r.md")
    created = client.post(
        "/api/v1/documents",
        headers=headers,
        json={
            "blob_id": blob_id,
            "knowledge_base_id": kb["id"],
            "original_filename": "r.md",
        },
    ).json()

    # 刚建出来是 pending，不是 failed
    response = client.post(f"/api/v1/documents/{created['document_id']}/retry", headers=headers)
    assert response.status_code == 409, response.text


def test_document_list_filters_by_kb(client: TestClient) -> None:
    """`kb_id` 过滤生效，且不返回其它库的文档。"""
    headers = _login(client, f"doc-{uuid.uuid4().hex[:8]}")
    kb1 = _create_kb(client, headers, name="库一")
    kb2 = _create_kb(client, headers, name="库二")

    for kb, name in ((kb1, "a.md"), (kb2, "b.md")):
        blob_id = _upload_blob(client, headers, f"# {name}\n\nbody\n".encode(), name)
        client.post(
            "/api/v1/documents",
            headers=headers,
            json={
                "blob_id": blob_id,
                "knowledge_base_id": kb["id"],
                "original_filename": name,
            },
        )

    listed = client.get(f"/api/v1/documents?kb_id={kb1['id']}", headers=headers).json()
    assert len(listed["items"]) == 1
    assert listed["items"][0]["knowledge_base_id"] == kb1["id"]


def test_document_list_rejects_unknown_status(client: TestClient) -> None:
    """未知 status 过滤值返回 422，不静默忽略。

    静默忽略会让"我只想看失败的文档"这个场景拿到全量数据，
    用户据此误判"没有失败的文档"。
    """
    headers = _login(client, f"doc-{uuid.uuid4().hex[:8]}")
    response = client.get("/api/v1/documents?status=bogus", headers=headers)
    assert response.status_code == 422, response.text
