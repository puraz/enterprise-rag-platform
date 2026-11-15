"""解析 handler 端到端测试（真实 PG + 真实本地存储 + 真实 PDF）。

## 为什么必须端到端

单测能证明"每一步的逻辑对"，但证明不了这条链能跑通：

    任务 payload → 读 version+blob（eager load）→ 存储下载
      → 解析 → 偏移校验 → 写派生产物 → 乐观锁推进状态

其中任何一环的签名/字段名不匹配都会在这条链上暴露。A8-2 之前
就出现过 handler 对着"想象的表结构"写（用了不存在的列），
单测发现不了，只有真跑才知道。

## 用真实 PDF 而不是 mock

解析器的输入是真实二进制。mock 掉 `parse_document` 就测不到
"PyMuPDF 能不能打开这个文件"、"偏移校验通过吗"、"页码对不对"。

PDF 用 fpdf2 现场生成（dev 依赖里已有），不引入 fixture 文件：
- 不占仓库体积；
- 内容由测试代码生成，意图一目了然（含中文断言）。

## 为什么不用 db_session fixture

`db_session` 在外层事务里，测试结束回滚。但 handler 内部会
`commit()`（真实代码路径就这么走），提交会结束外层事务，
回滚语义失效。

所以这里用**独立引擎 + 显式清理**，与并发测试同一策略。
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from rag_contracts.enums import DocumentFormat, DocumentStatus
from rag_core.settings import get_settings
from rag_worker.handlers.parse import ParseDocumentHandler

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------- fixture


@pytest.fixture
async def env() -> AsyncIterator[tuple[Any, Any, Any, Path]]:
    """准备：独立引擎 + 临时存储 + 临时目录。

    返回 `(session_factory, storages, tenant_id, storage_root)`。
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from rag_adapters.storages.local import LocalStorageBackend
    from rag_core.settings.storage import StorageSettings

    settings = get_settings()
    db = settings.database
    user = db.migration_user or db.user
    password = (
        db.migration_password.get_secret_value()
        if db.migration_password is not None
        else db.password.get_secret_value()
    )
    dsn = f"postgresql+asyncpg://{user}:{password}@{db.host}:{db.port}/{db.name}"

    engine = create_async_engine(dsn, pool_size=4)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)

    # 临时存储根目录：用 pytest 的 tmp_path 语义（这里手动管理，
    # 因为 fixture 需要它同时给存储后端和断言用）
    import tempfile

    storage_root = Path(tempfile.mkdtemp(prefix="rag-worker-test-"))
    storage = LocalStorageBackend(StorageSettings(backend="local", local_root=str(storage_root)))

    tenant_id = uuid.uuid4()
    await _create_tenant(factory, tenant_id=tenant_id)
    try:
        yield factory, {"local": storage}, tenant_id, storage_root
    finally:
        await _cleanup(factory, tenant_id=tenant_id)
        await engine.dispose()
        import shutil

        shutil.rmtree(storage_root, ignore_errors=True)


# ---------------------------------------------------------------- 主流程


async def test_parse_pdf_end_to_end(env: tuple[Any, Any, Any, Path]) -> None:
    """端到端：上传的 PDF → 解析 → 落产物 → 状态推进到 CHUNKING。"""
    factory, storages, tenant_id, _storage_root = env

    # 用 ASCII 内容：fpdf2 内置字体不含 CJK，中文会被跳过变成占位符，
    # 那样断言"解析出了什么文字"就没有意义了。
    # 中文文档的解析能力由 test_parsers.py 用夹具覆盖。
    pdf_bytes = _make_pdf(["Page one: enterprise policy", "Page two: expense flow"])
    document_id, version_id, storage_key = await _create_document(
        factory,
        tenant_id=tenant_id,
        content=pdf_bytes,
        filename="制度.pdf",
        file_format=DocumentFormat.PDF,
    )
    # 把"上传的原始文件"写进存储（模拟 complete 阶段已完成）
    storage = storages["local"]
    await storage.put(storage_key, pdf_bytes, content_type="application/pdf")

    handler = ParseDocumentHandler(storages=storages)

    task = _make_task(
        tenant_id=tenant_id,
        payload={"document_id": str(document_id), "document_version_id": str(version_id)},
    )

    progress_calls: list[int] = []

    async def progress(percent: int, stage: str = "") -> None:
        progress_calls.append(percent)

    async with factory() as session:
        await handler(session=session, task=task, progress=progress)
        await session.commit()

    # ---- 断言 1：状态推进到 CHUNKING ----
    async with factory() as session:
        row = await session.execute(
            _sql(
                "SELECT status, page_count, error_code FROM documents WHERE id = :id",
                id=document_id,
            )
        )
        status, page_count, error_code = row.one()
    assert status == DocumentStatus.CHUNKING.value, "解析成功后应进入 CHUNKING（等切分）"
    assert page_count == 2
    assert error_code is None

    # ---- 断言 2：版本的解析统计与产物指针 ----
    async with factory() as session:
        row = await session.execute(
            _sql(
                "SELECT parser_name, page_count, parse_metadata FROM document_versions "
                "WHERE id = :id",
                id=version_id,
            )
        )
        parser_name, version_pages, parse_metadata = row.one()
    assert parser_name, "必须记录实际使用的解析器（供追溯）"
    assert version_pages == 2
    assert parse_metadata["parsed_key"].startswith("parsed/")
    assert parse_metadata["char_count"] > 0
    assert parse_metadata["has_scanned_pages"] is False

    # ---- 断言 3：派生产物真的写进了存储，且内容可读 ----
    parsed_key = parse_metadata["parsed_key"]
    chunks: list[bytes] = []
    async for chunk in storage.get(parsed_key):
        chunks.append(chunk)
    parsed_text = b"".join(chunks).decode("utf-8")
    # 断言真实正文：证明"下载→解析→落盘"这条链传递的是文件内容本身，
    # 不是占位符或空串
    assert "enterprise policy" in parsed_text
    assert "expense flow" in parsed_text

    # ---- 断言 4：进度按里程碑上报，且没有报 100 ----
    assert progress_calls, "长任务应上报进度"
    assert 100 not in progress_calls, "100 只能由 complete() 设置"


async def test_parse_missing_version_is_permanent_failure(
    env: tuple[Any, Any, Any, Path],
) -> None:
    """版本不存在 → NotFoundError（分类为 PERMANENT，不重试）。

    重试一个"查不到的行"毫无意义，只会让用户多等三轮退避。
    """
    from rag_core.errors import NotFoundError

    factory, storages, tenant_id, _ = env
    handler = ParseDocumentHandler(storages=storages)

    task = _make_task(
        tenant_id=tenant_id,
        payload={
            "document_id": str(uuid.uuid4()),
            "document_version_id": str(uuid.uuid4()),
        },
    )

    async def progress(percent: int, stage: str = "") -> None:
        return None

    async with factory() as session:
        with pytest.raises(NotFoundError):
            await handler(session=session, task=task, progress=progress)


async def test_parse_payload_mismatch_is_rejected(
    env: tuple[Any, Any, Any, Path],
) -> None:
    """payload 里 document_id 与 version 的真实归属不一致 → 拒绝。

    不能"按 version 猜一个 document"：猜错会改错文档的状态，
    而且这个错误不会有任何报错——被改的文档从此显示"已解析"，
    但它其实从未被解析。
    """
    from rag_adapters.parsers import ParseError

    factory, storages, tenant_id, _storage_root = env
    pdf_bytes = _make_pdf(["content"])
    _document_id, version_id, storage_key = await _create_document(
        factory,
        tenant_id=tenant_id,
        content=pdf_bytes,
        filename="a.pdf",
        file_format=DocumentFormat.PDF,
    )
    await storages["local"].put(storage_key, pdf_bytes)

    handler = ParseDocumentHandler(storages=storages)
    wrong_document_id = uuid.uuid4()

    task = _make_task(
        tenant_id=tenant_id,
        payload={
            "document_id": str(wrong_document_id),  # 错的不属于这个 version
            "document_version_id": str(version_id),
        },
    )

    async def progress(percent: int, stage: str = "") -> None:
        return None

    async with factory() as session:
        with pytest.raises(ParseError) as exc_info:
            await handler(session=session, task=task, progress=progress)

    assert exc_info.value.retryable is False, "payload 错误不该重试"


async def test_parse_unknown_storage_backend_is_permanent(
    env: tuple[Any, Any, Any, Path],
) -> None:
    """blob 记录的后端未配置 → 永久失败（配置错误，重试无用）。"""
    from rag_adapters.parsers import ParseError

    factory, _storages, tenant_id, storage_root = env
    pdf_bytes = _make_pdf(["content"])
    document_id, version_id, _ = await _create_document(
        factory,
        tenant_id=tenant_id,
        content=pdf_bytes,
        filename="a.pdf",
        file_format=DocumentFormat.PDF,
        storage_backend="s3",  # 记录成 s3，但只提供 local 后端
    )

    # 只给一个空的 local 后端注册表（不含 s3）
    from rag_adapters.storages.local import LocalStorageBackend
    from rag_core.settings.storage import StorageSettings

    only_local = {
        "local": LocalStorageBackend(StorageSettings(backend="local", local_root=str(storage_root)))
    }
    handler = ParseDocumentHandler(storages=only_local)

    task = _make_task(
        tenant_id=tenant_id,
        payload={"document_id": str(document_id), "document_version_id": str(version_id)},
    )

    async def progress(percent: int, stage: str = "") -> None:
        return None

    async with factory() as session:
        with pytest.raises(ParseError) as exc_info:
            await handler(session=session, task=task, progress=progress)

    assert exc_info.value.retryable is False
    assert "s3" in str(exc_info.value)


async def test_parse_is_idempotent(env: tuple[Any, Any, Any, Path]) -> None:
    """同一任务跑两遍结果一致（重派是正常路径，不是异常）。

    为什么重要：业务提交与"标记任务成功"之间崩溃时，任务会被回收重派。
    如果 handler 不幂等，重跑会产生重复 chunk / 重复计费 / 状态错乱。
    """
    factory, storages, tenant_id, storage_root = env
    pdf_bytes = _make_pdf(["idempotency check"])
    document_id, version_id, storage_key = await _create_document(
        factory,
        tenant_id=tenant_id,
        content=pdf_bytes,
        filename="idem.pdf",
        file_format=DocumentFormat.PDF,
    )
    await storages["local"].put(storage_key, pdf_bytes)

    handler = ParseDocumentHandler(storages=storages)

    task = _make_task(
        tenant_id=tenant_id,
        payload={"document_id": str(document_id), "document_version_id": str(version_id)},
    )

    async def progress(percent: int, stage: str = "") -> None:
        return None

    for _ in range(2):
        async with factory() as session:
            await handler(session=session, task=task, progress=progress)
            await session.commit()

    # 跑两遍后：文档状态仍是 CHUNKING，page_count 没变成 4
    async with factory() as session:
        row = await session.execute(
            _sql(
                "SELECT status, page_count, version FROM documents WHERE id = :id",
                id=document_id,
            )
        )
        status, page_count, _row_version = row.one()
    assert status == DocumentStatus.CHUNKING.value
    assert page_count == 1

    # 派生产物只有一个（确定性键，没有产生第二个对象）
    parsed_dir = storage_root / "parsed" / str(tenant_id)
    assert len(list(parsed_dir.glob("*.txt"))) == 1, "重跑不该留下多个派生产物"


# ---------------------------------------------------------------- 数据准备


def _make_task(*, tenant_id: uuid.UUID, payload: dict[str, str]) -> Any:
    """构造最小任务对象。

    用 dataclass 而不是 `class _Task: tenant_id = tenant_id`：
    类体内右侧的同名引用**看不到外层函数变量**（类作用域不参与闭包查找），
    会直接 NameError。这是 Python 作用域的经典陷阱。
    """

    @dataclass
    class _Task:
        id: uuid.UUID
        tenant_id: uuid.UUID
        task_type: str
        payload: dict[str, str]

    return _Task(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        task_type="parse_document",
        payload=payload,
    )


def _sql(statement: str, **params: Any) -> Any:
    from sqlalchemy import text

    return text(statement).bindparams(**params)


async def _create_tenant(factory: Any, *, tenant_id: uuid.UUID) -> None:
    from sqlalchemy import text

    async with factory() as session:
        await session.execute(
            text(
                "INSERT INTO tenants (id, slug, name, status, created_at, updated_at) "
                "VALUES (:id, :slug, :name, 'active', now(), now())"
            ),
            {"id": tenant_id, "slug": f"t-{tenant_id.hex[:8]}", "name": "解析测试租户"},
        )
        await session.commit()


async def _create_document(
    factory: Any,
    *,
    tenant_id: uuid.UUID,
    content: bytes,
    filename: str,
    file_format: DocumentFormat,
    storage_backend: str = "local",
) -> tuple[uuid.UUID, uuid.UUID, str]:
    """建 tenant → user → kb → blob → document → version 整条链。

    返回 `(document_id, version_id, storage_key)`。

    用仓储层而不是裸 INSERT：这些对象有大量字段与默认值，
    裸 INSERT 会在 schema 变更时静默写漏字段。
    """
    import hashlib

    from sqlalchemy import text

    from rag_db.repositories import (
        BlobRepository,
        DocumentRepository,
        KnowledgeBaseRepository,
    )

    user_id = uuid.uuid4()
    async with factory() as session:
        # 用户是全局表（无 tenant_id），见 models/user.py 的说明
        await session.execute(
            text(
                "INSERT INTO users (id, email, display_name, status, platform_role, "
                "created_at, updated_at) "
                "VALUES (:id, :email, :name, 'active', 'none', now(), now())"
            ),
            {
                "id": user_id,
                "email": f"{user_id.hex[:8]}@test.local",
                "name": "解析测试用户",
            },
        )
        kb = await KnowledgeBaseRepository(session).create(
            tenant_id=tenant_id,
            name=f"kb-{uuid.uuid4().hex[:8]}",
            created_by=user_id,
        )
        content_hash = hashlib.sha256(content).hexdigest()
        storage_key = f"blobs/{tenant_id}/{content_hash[:2]}/{content_hash}"
        # get_or_create 返回 (blob, created)：created=False 表示秒传命中
        blob, _created = await BlobRepository(session).get_or_create(
            tenant_id=tenant_id,
            content_hash=content_hash,
            size_bytes=len(content),
            storage_key=storage_key,
            storage_backend=storage_backend,
            content_type="application/pdf",
        )
        doc = await DocumentRepository(session).create(
            tenant_id=tenant_id,
            knowledge_base_id=kb.id,
            title=filename,
            original_filename=filename,
            format=file_format.value,
            created_by=user_id,
        )
        repo = DocumentRepository(session)
        version_no = await repo.allocate_version_no(doc_id=doc.id)
        version = await repo.create_version(
            tenant_id=tenant_id,
            document_id=doc.id,
            version_no=version_no,
            blob_id=blob.id,
            created_by=user_id,
            chunking_strategy="recursive",
            chunking_config={},
        )
        doc.status = DocumentStatus.PARSING
        await session.commit()
        return doc.id, version.id, storage_key


async def _cleanup(factory: Any, *, tenant_id: uuid.UUID) -> None:
    """按外键顺序清理。

    顺序不能反：document_versions → documents → knowledge_bases → blobs → tenants。
    反过来会撞 FK RESTRICT（blob 被 version 引用时不能删）。

    users 是**全局表**（无 tenant_id、无 created_by 关联到 tenant），
    只能按测试专用邮箱后缀清理 —— 这是为什么 `_create_document`
    统一用 `@test.local`：清理的 pattern 必须不可能命中真实数据。
    """
    from sqlalchemy import text

    async with factory() as session:
        # chunks/index_versions/tasks 由 FK CASCADE 跟随 document 删除
        await session.execute(
            text("DELETE FROM document_versions WHERE tenant_id = :tid"), {"tid": tenant_id}
        )
        await session.execute(
            text("DELETE FROM documents WHERE tenant_id = :tid"), {"tid": tenant_id}
        )
        await session.execute(
            text("DELETE FROM knowledge_bases WHERE tenant_id = :tid"), {"tid": tenant_id}
        )
        await session.execute(text("DELETE FROM blobs WHERE tenant_id = :tid"), {"tid": tenant_id})
        await session.execute(text("DELETE FROM users WHERE email LIKE '%@test.local'"), {})
        await session.execute(text("DELETE FROM tenants WHERE id = :tid"), {"tid": tenant_id})
        await session.commit()


def _make_pdf(pages: list[str]) -> bytes:
    """用 fpdf2 生成多页 PDF，每页一段正文。

    **只接受 ASCII**：fpdf2 内置字体（Helvetica）不含 CJK 字形，
    中文会渲染成空/占位符。中文 PDF 的解析能力已由 `test_parsers.py`
    用真实夹具覆盖，这里只需验证管线能把内容传递到位。

    用真实 PDF 而不是 mock：要验证 PyMuPDF 能打开、页码从 1 开始、
    偏移校验通过——这些 mock 全都测不到。
    """
    from fpdf import FPDF

    pdf = FPDF()
    for page_text in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 10, page_text)
    return bytes(pdf.output())
