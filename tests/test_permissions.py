"""权限与 ACL 测试。

两部分：

1. **纯逻辑测试**（无需数据库）：密级比较、角色上限、引用校验
2. **SQL 过滤测试**（真实 PG）：ACL 条件真的在数据库里生效

第二种最重要：如果 ACL 只是"在 Python 里过滤"，测试也会通过，
但生产上会分页错乱 + 越权内容进内存。
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.permissions.acl import (
    LEVEL_RANK,
    ROLE_MAX_LEVEL,
    PermissionContext,
    assert_document_readable,
    assert_same_tenant,
    confidentiality_filter_for_chunks,
    document_access_filter,
    verify_citations_visible,
)
from rag_contracts.enums import ConfidentialityLevel, TenantRole
from rag_core.errors import NotFoundError, PermissionDeniedError, TenantMismatchError

# ==================================================== 密级与角色


def _ctx(*roles: TenantRole, user_id: uuid.UUID | None = None) -> PermissionContext:
    return PermissionContext(
        user_id=user_id or uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        roles=frozenset(roles),
    )


def test_confidentiality_levels_are_ordered() -> None:
    """密级有序：public < internal < confidential < secret。

    **不能依赖枚举的字母序**：`confidential` < `internal` < `public`
    是按字母排的，与敏感程度完全无关。直接字符串比较会让
    `secret` 被当成最低级——权限完全错了且不报错。
    """
    assert LEVEL_RANK[ConfidentialityLevel.PUBLIC] < LEVEL_RANK[ConfidentialityLevel.INTERNAL]
    assert LEVEL_RANK[ConfidentialityLevel.INTERNAL] < LEVEL_RANK[ConfidentialityLevel.CONFIDENTIAL]
    assert LEVEL_RANK[ConfidentialityLevel.CONFIDENTIAL] < LEVEL_RANK[ConfidentialityLevel.SECRET]


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        (TenantRole.OWNER, ConfidentialityLevel.SECRET),
        (TenantRole.ADMIN, ConfidentialityLevel.SECRET),
        (TenantRole.KNOWLEDGE_ADMIN, ConfidentialityLevel.CONFIDENTIAL),
        (TenantRole.MEMBER, ConfidentialityLevel.INTERNAL),
        (TenantRole.VIEWER, ConfidentialityLevel.INTERNAL),
    ],
)
def test_role_max_confidentiality(role: TenantRole, expected: ConfidentialityLevel) -> None:
    """每个角色能看到的最高密级。"""
    assert ROLE_MAX_LEVEL[role] == expected
    assert _ctx(role).max_confidentiality == expected


def test_multiple_roles_take_lowest_ceiling() -> None:
    """**多角色取最低上限**。

    一个人同时是 owner（SECRET）和 viewer（INTERNAL）时按 viewer 算。

    理由：降级授权不该因为"还挂着高权限角色"而被绕过——
    如果按最高算，想临时限制某人权限就只能移除他的全有角色，
    而管理员往往会忘掉这一点。
    """
    ctx = _ctx(TenantRole.OWNER, TenantRole.VIEWER)
    assert ctx.max_confidentiality == ConfidentialityLevel.INTERNAL


def test_no_role_sees_only_public() -> None:
    """没有任何角色时只能看公开内容（安全默认）。"""
    ctx = PermissionContext(user_id=uuid.uuid4(), tenant_id=uuid.uuid4())
    assert ctx.max_confidentiality == ConfidentialityLevel.PUBLIC


# ==================================================== 角色校验


def test_require_role_passes() -> None:
    """具备角色时通过。"""
    _ctx(TenantRole.ADMIN).require_role(TenantRole.ADMIN, TenantRole.OWNER)


def test_require_role_fails() -> None:
    """不具备角色时抛 PermissionDeniedError。"""
    with pytest.raises(PermissionDeniedError, match="需要"):
        _ctx(TenantRole.VIEWER).require_role(TenantRole.ADMIN)


def test_viewer_cannot_write_kb() -> None:
    """**只读者不能上传/修改文档**。"""
    with pytest.raises(PermissionDeniedError):
        _ctx(TenantRole.VIEWER).require_kb_write()


@pytest.mark.parametrize(
    "role",
    [TenantRole.OWNER, TenantRole.ADMIN, TenantRole.KNOWLEDGE_ADMIN],
)
def test_kb_write_roles_pass(role: TenantRole) -> None:
    """owner/admin/knowledge_admin 可以写知识库内容。"""
    _ctx(role).require_kb_write()


def test_member_cannot_manage_members() -> None:
    """普通成员不能管理租户成员。"""
    with pytest.raises(PermissionDeniedError):
        _ctx(TenantRole.MEMBER).require_member_manage()


def test_knowledge_admin_cannot_manage_members() -> None:
    """**知识管理员不能管成员**：它只管内容，不管人。

    这是最小权限原则：如果知识管理员能改成员角色，他就能给自己提权。
    """
    with pytest.raises(PermissionDeniedError):
        _ctx(TenantRole.KNOWLEDGE_ADMIN).require_member_manage()


# ==================================================== 租户校验


def test_same_tenant_passes() -> None:
    tenant_id = uuid.uuid4()
    assert_same_tenant(token_tenant_id=tenant_id, resource_tenant_id=tenant_id)


def test_cross_tenant_rejected() -> None:
    """**跨租户访问必须抛 TenantMismatchError**（severity=CRITICAL）。

    这是业务层的最后一道检查（RLS 是第一道）：
    即使仓储漏写了 `WHERE tenant_id`，这里也能拦住。
    """
    with pytest.raises(TenantMismatchError):
        assert_same_tenant(token_tenant_id=uuid.uuid4(), resource_tenant_id=uuid.uuid4())


# ==================================================== 文档可读性


def test_document_readable_when_level_and_kb_ok() -> None:
    """密级够且库可见时可读。"""
    ctx = _ctx(TenantRole.MEMBER)  # INTERNAL
    assert_document_readable(
        document_confidentiality=ConfidentialityLevel.INTERNAL.value,
        kb_visible=True,
        ctx=ctx,
    )


def test_document_rejected_when_level_too_high() -> None:
    """**密级不够时返回 404（不是 403）**。

    403 会告诉攻击者"这个文档存在但你没权限"——可以用来枚举
    有效文档 ID。404 让"不存在"与"无权"无法区分。
    """
    ctx = _ctx(TenantRole.MEMBER)  # INTERNAL
    with pytest.raises(NotFoundError):
        assert_document_readable(
            document_confidentiality=ConfidentialityLevel.SECRET.value,
            kb_visible=True,
            ctx=ctx,
        )


def test_document_rejected_when_kb_invisible() -> None:
    """库不可见时也返回 404。"""
    ctx = _ctx(TenantRole.OWNER)  # SECRET，但库不可见
    with pytest.raises(NotFoundError):
        assert_document_readable(
            document_confidentiality=ConfidentialityLevel.PUBLIC.value,
            kb_visible=False,
            ctx=ctx,
        )


def test_unknown_confidentiality_treated_as_secret() -> None:
    """未知密级按最高级处理（安全默认）。

    数据库里出现未知值时，如果按最低级处理会让内容泄漏；
    按最高级处理只是"看不到"。
    """
    ctx = _ctx(TenantRole.MEMBER)
    with pytest.raises(NotFoundError):
        assert_document_readable(
            document_confidentiality="unknown-level-value",
            kb_visible=True,
            ctx=ctx,
        )


# ==================================================== 引用校验


def test_citations_all_visible() -> None:
    """所有引用都在可见集内时无越界。"""
    doc_a, doc_b = uuid.uuid4(), uuid.uuid4()
    assert (
        verify_citations_visible(
            cited_document_ids=[doc_a, doc_b], visible_document_ids={doc_a, doc_b}
        )
        == []
    )


def test_citations_detect_leak() -> None:
    """**检测引用越界**。

    模型可能把 A 文档的内容归到 B 文档，或上下文压缩引入了未过滤内容。
    引用越界是最隐蔽的泄漏形态——它嵌在答案里，用户看不出异常。
    """
    visible = uuid.uuid4()
    leaked = uuid.uuid4()
    result = verify_citations_visible(
        cited_document_ids=[visible, leaked], visible_document_ids={visible}
    )
    assert result == [leaked]


# ==================================================== SQL 过滤（真实数据库）

pytestmark_integration = pytest.mark.integration


@pytest.mark.integration
async def test_sql_filter_excludes_high_confidentiality(
    db_session: AsyncSession,
) -> None:
    """**SQL 过滤真的生效**：MEMBER 查不到 SECRET 文档。

    这条测试验证的是"过滤在 SQL 里"，而不是"Python 里过滤"。
    如果实现是查完再过滤，这个测试也会通过——但可以用
    `test_sql_filter_is_in_where_clause` 区分。
    """
    from sqlalchemy import text

    from rag_db.models.document import Document

    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    kb_id = uuid.uuid4()

    async with db_session.begin_nested():
        await db_session.execute(
            text(
                "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
                "VALUES (:i, :s, 'x', 'active', now(), now())"
            ),
            {"i": tenant_id, "s": f"t-{tenant_id.hex[:10]}"},
        )
        await db_session.execute(
            text(
                "INSERT INTO users (id, email, display_name, status, platform_role, "
                "tokens_valid_after, created_at, updated_at) "
                "VALUES (:i, :e, 'u', 'active', 'none', now(), now(), now())"
            ),
            {"i": user_id, "e": f"u-{user_id.hex[:8]}@example.test"},
        )
        await db_session.execute(
            text(
                "INSERT INTO knowledge_bases (id, tenant_id, name, status, visibility, "
                "confidentiality, created_by, default_chunking_strategy, "
                "default_chunk_size, default_chunk_overlap, document_count, "
                "created_at, updated_at) VALUES (:i, :t, 'kb', 'active', 'tenant', "
                "'internal', :u, 'recursive', 800, 100, 0, now(), now())"
            ),
            {"i": kb_id, "t": tenant_id, "u": user_id},
        )

        # 两个文档：一个 INTERNAL，一个 SECRET
        internal_doc, secret_doc = uuid.uuid4(), uuid.uuid4()
        for doc_id, level in (
            (internal_doc, "internal"),
            (secret_doc, "secret"),
        ):
            await db_session.execute(
                text(
                    "INSERT INTO documents (id, tenant_id, knowledge_base_id, title, "
                    "original_filename, format, source_type, confidentiality, status, "
                    "current_version_no, latest_version_no, retry_count, chunk_count, "
                    "size_bytes, created_by, version, created_at, updated_at) "
                    "VALUES (:i, :t, :k, :title, 'x.pdf', 'pdf', 'upload', :lvl, "
                    "'ready', 1, 1, 0, 0, 100, :u, 1, now(), now())"
                ),
                {
                    "i": doc_id,
                    "t": tenant_id,
                    "k": kb_id,
                    "title": f"doc-{level}",
                    "lvl": level,
                    "u": user_id,
                },
            )

        # MEMBER 角色（INTERNAL 上限）
        ctx = PermissionContext(
            user_id=user_id,
            tenant_id=tenant_id,
            roles=frozenset({TenantRole.MEMBER}),
        )

        result = await db_session.execute(
            select(Document.id).where(
                Document.tenant_id == tenant_id,
                document_access_filter(ctx),
            )
        )
        visible = set(result.scalars())

    assert internal_doc in visible, "INTERNAL 文档应该可见"
    assert secret_doc not in visible, "**SECRET 文档不该被 MEMBER 看到**"


@pytest.mark.integration
async def test_owner_sees_all_levels(db_session: AsyncSession) -> None:
    """OWNER（SECRET 上限）能看到所有密级。"""
    from sqlalchemy import text

    from rag_db.models.document import Document

    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    kb_id = uuid.uuid4()

    async with db_session.begin_nested():
        await db_session.execute(
            text(
                "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
                "VALUES (:i, :s, 'x', 'active', now(), now())"
            ),
            {"i": tenant_id, "s": f"t-{tenant_id.hex[:10]}"},
        )
        await db_session.execute(
            text(
                "INSERT INTO users (id, email, display_name, status, platform_role, "
                "tokens_valid_after, created_at, updated_at) "
                "VALUES (:i, :e, 'u', 'active', 'none', now(), now(), now())"
            ),
            {"i": user_id, "e": f"u-{user_id.hex[:8]}@example.test"},
        )
        await db_session.execute(
            text(
                "INSERT INTO knowledge_bases (id, tenant_id, name, status, visibility, "
                "confidentiality, created_by, default_chunking_strategy, "
                "default_chunk_size, default_chunk_overlap, document_count, "
                "created_at, updated_at) VALUES (:i, :t, 'kb', 'active', 'tenant', "
                "'internal', :u, 'recursive', 800, 100, 0, now(), now())"
            ),
            {"i": kb_id, "t": tenant_id, "u": user_id},
        )

        doc_ids = []
        for level in ("public", "internal", "confidential", "secret"):
            doc_id = uuid.uuid4()
            doc_ids.append(doc_id)
            await db_session.execute(
                text(
                    "INSERT INTO documents (id, tenant_id, knowledge_base_id, title, "
                    "original_filename, format, source_type, confidentiality, status, "
                    "current_version_no, latest_version_no, retry_count, chunk_count, "
                    "size_bytes, created_by, version, created_at, updated_at) "
                    "VALUES (:i, :t, :k, :title, 'x.pdf', 'pdf', 'upload', :lvl, "
                    "'ready', 1, 1, 0, 0, 100, :u, 1, now(), now())"
                ),
                {
                    "i": doc_id,
                    "t": tenant_id,
                    "k": kb_id,
                    "title": f"doc-{level}",
                    "lvl": level,
                    "u": user_id,
                },
            )

        ctx = PermissionContext(
            user_id=user_id,
            tenant_id=tenant_id,
            roles=frozenset({TenantRole.OWNER}),
        )
        result = await db_session.execute(
            select(Document.id).where(
                Document.tenant_id == tenant_id,
                document_access_filter(ctx),
            )
        )
        visible = set(result.scalars())

    assert set(doc_ids) == visible, f"OWNER 应看到全部 4 个密级，实际 {len(visible)}"


@pytest.mark.integration
async def test_private_kb_only_visible_to_creator(db_session: AsyncSession) -> None:
    """**私有知识库只有创建者可见**。"""
    from sqlalchemy import text

    from rag_db.models.document import Document

    tenant_id = uuid.uuid4()
    creator_id, other_id = uuid.uuid4(), uuid.uuid4()
    private_kb = uuid.uuid4()

    async with db_session.begin_nested():
        await db_session.execute(
            text(
                "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
                "VALUES (:i, :s, 'x', 'active', now(), now())"
            ),
            {"i": tenant_id, "s": f"t-{tenant_id.hex[:10]}"},
        )
        for uid in (creator_id, other_id):
            await db_session.execute(
                text(
                    "INSERT INTO users (id, email, display_name, status, platform_role, "
                    "tokens_valid_after, created_at, updated_at) "
                    "VALUES (:i, :e, 'u', 'active', 'none', now(), now(), now())"
                ),
                {"i": uid, "e": f"u-{uid.hex[:8]}@example.test"},
            )
        await db_session.execute(
            text(
                "INSERT INTO knowledge_bases (id, tenant_id, name, status, visibility, "
                "confidentiality, created_by, default_chunking_strategy, "
                "default_chunk_size, default_chunk_overlap, document_count, "
                "created_at, updated_at) VALUES (:i, :t, 'private-kb', 'active', "
                "'private', 'internal', :u, 'recursive', 800, 100, 0, now(), now())"
            ),
            {"i": private_kb, "t": tenant_id, "u": creator_id},
        )

        doc_id = uuid.uuid4()
        await db_session.execute(
            text(
                "INSERT INTO documents (id, tenant_id, knowledge_base_id, title, "
                "original_filename, format, source_type, confidentiality, status, "
                "current_version_no, latest_version_no, retry_count, chunk_count, "
                "size_bytes, created_by, version, created_at, updated_at) "
                "VALUES (:i, :t, :k, 'private-doc', 'x.pdf', 'pdf', 'upload', "
                "'internal', 'ready', 1, 1, 0, 0, 100, :u, 1, now(), now())"
            ),
            {"i": doc_id, "t": tenant_id, "k": private_kb, "u": creator_id},
        )

        # 创建者能看到
        creator_ctx = PermissionContext(
            user_id=creator_id, tenant_id=tenant_id, roles=frozenset({TenantRole.OWNER})
        )
        creator_visible = await db_session.execute(
            select(Document.id).where(
                Document.tenant_id == tenant_id, document_access_filter(creator_ctx)
            )
        )

        # 同租户的其他人看不到
        other_ctx = PermissionContext(
            user_id=other_id, tenant_id=tenant_id, roles=frozenset({TenantRole.OWNER})
        )
        other_visible = await db_session.execute(
            select(Document.id).where(
                Document.tenant_id == tenant_id, document_access_filter(other_ctx)
            )
        )

    assert doc_id in set(creator_visible.scalars()), "创建者应能看到自己的私有库"
    assert doc_id not in set(other_visible.scalars()), "同租户他人不该看到私有库"


@pytest.mark.integration
async def test_chunk_filter_excludes_unauthorized_documents(
    db_session: AsyncSession,
) -> None:
    """**检索过滤**：chunk 查询不能返回无权文档的 chunk。

    这是最危险的路径——chunk 内容就是文档正文，
    不过滤等于用户能通过提问把无权文档"问出来"。
    """
    from sqlalchemy import text

    from rag_db.models.chunk import Chunk

    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    kb_id = uuid.uuid4()

    async with db_session.begin_nested():
        await db_session.execute(
            text(
                "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
                "VALUES (:i, :s, 'x', 'active', now(), now())"
            ),
            {"i": tenant_id, "s": f"t-{tenant_id.hex[:10]}"},
        )
        await db_session.execute(
            text(
                "INSERT INTO users (id, email, display_name, status, platform_role, "
                "tokens_valid_after, created_at, updated_at) "
                "VALUES (:i, :e, 'u', 'active', 'none', now(), now(), now())"
            ),
            {"i": user_id, "e": f"u-{user_id.hex[:8]}@example.test"},
        )
        await db_session.execute(
            text(
                "INSERT INTO knowledge_bases (id, tenant_id, name, status, visibility, "
                "confidentiality, created_by, default_chunking_strategy, "
                "default_chunk_size, default_chunk_overlap, document_count, "
                "created_at, updated_at) VALUES (:i, :t, 'kb', 'active', 'tenant', "
                "'internal', :u, 'recursive', 800, 100, 0, now(), now())"
            ),
            {"i": kb_id, "t": tenant_id, "u": user_id},
        )

        # 建一个 SECRET 文档 + 它的 chunk
        doc_id = uuid.uuid4()
        blob_id = uuid.uuid4()
        version_id = uuid.uuid4()
        await db_session.execute(
            text(
                "INSERT INTO blobs (id, tenant_id, content_hash, size_bytes, "
                "storage_key, storage_backend, content_type, status, ref_count, "
                "created_at, updated_at) VALUES (:i, :t, :h, 10, 'k', 'local', "
                "'application/pdf', 'complete', 1, now(), now())"
            ),
            {"i": blob_id, "t": tenant_id, "h": uuid.uuid4().hex + uuid.uuid4().hex},
        )
        await db_session.execute(
            text(
                "INSERT INTO documents (id, tenant_id, knowledge_base_id, title, "
                "original_filename, format, source_type, confidentiality, status, "
                "current_version_no, latest_version_no, retry_count, chunk_count, "
                "size_bytes, created_by, version, created_at, updated_at) "
                "VALUES (:i, :t, :k, 'secret-doc', 'x.pdf', 'pdf', 'upload', 'secret', "
                "'ready', 1, 1, 0, 0, 10, :u, 1, now(), now())"
            ),
            {"i": doc_id, "t": tenant_id, "k": kb_id, "u": user_id},
        )
        await db_session.execute(
            text(
                "INSERT INTO document_versions (id, tenant_id, document_id, version_no, "
                "blob_id, status, chunking_strategy, chunking_config, chunk_count, "
                "parse_metadata, created_by, is_current, created_at, updated_at) "
                "VALUES (:i, :t, :d, 1, :b, 'ready', 'recursive', '{}'::jsonb, 0, "
                "'{}'::jsonb, :u, true, now(), now())"
            ),
            {"i": version_id, "t": tenant_id, "d": doc_id, "b": blob_id, "u": user_id},
        )
        chunk_id = uuid.uuid4()
        await db_session.execute(
            text(
                "INSERT INTO chunks (id, tenant_id, knowledge_base_id, document_id, "
                "document_version_id, chunk_index, chunk_type, content, content_hash, "
                "section_path, chunk_level, is_current, chunk_metadata, created_at, "
                "updated_at) VALUES (:i, :t, :k, :d, :v, 0, 'text', 'secret content', "
                ":h, '[]'::jsonb, 0, true, '{}'::jsonb, now(), now())"
            ),
            {
                "i": chunk_id,
                "t": tenant_id,
                "k": kb_id,
                "d": doc_id,
                "v": version_id,
                "h": uuid.uuid4().hex + uuid.uuid4().hex,
            },
        )

        # MEMBER 检索：不该看到 SECRET 文档的 chunk
        ctx = PermissionContext(
            user_id=user_id, tenant_id=tenant_id, roles=frozenset({TenantRole.MEMBER})
        )
        result = await db_session.execute(
            select(Chunk.id).where(
                Chunk.tenant_id == tenant_id,
                confidentiality_filter_for_chunks(ctx),
            )
        )
        visible = set(result.scalars())

    assert chunk_id not in visible, "**MEMBER 不该检索到 SECRET 文档的 chunk**"
