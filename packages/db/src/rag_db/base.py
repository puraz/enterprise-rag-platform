"""ORM 基类与通用列约定。

所有模型继承 `Base`，共享三件事：
1. **命名约定**：约束/索引自动生成确定的名字。不约定的话，PG 自动起的名字
   形如 `documents_tenant_id_fkey`/`pk_1234`，Alembic 生成的迁移里名字随机，
   回滚和 `DROP CONSTRAINT` 时无法预测。
2. **时间戳**：created_at/updated_at 由数据库侧生成（server_default），
   不靠 Python 侧赋值——多副本/多 worker 时 Python 时钟不一致会导致乱序。
3. **主键**：统一 UUID，由应用侧生成（而非 DB 的 gen_random_uuid），
   这样插入前就知道 ID，便于 outbox 消息里带上目标 ID。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# 显式命名约定：迁移脚本里出现的约束名必须是可预测的
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """全项目 ORM 基类。"""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class UUIDPrimaryKeyMixin:
    """UUID 主键。

    为什么用 UUID 而不是自增 BIGINT：
    - 跨租户/跨实例的 ID 不泄漏"总量"（自增 ID 会暴露业务规模）；
    - 分布式插入不需要中心序列，避免序列成为写热点；
    - 代价是索引更大、随机写入导致 B-tree 页分裂更频繁。
      用 uuid7/顺序 UUID 可缓解，但 Python 3.12 标准库没有 uuid7，
      而引入第三方库的收益目前不足以抵消依赖成本，故先用 uuid4。
    """

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="主键",
    )


class TimestampMixin:
    """创建/更新时间。

    用 server_default=func.now() 让**数据库**决定时间：
    API 与 worker 是不同进程，各自时钟可能有毫秒级偏差，
    由 Python 侧赋时间会导致"worker 写入的记录 created_at 早于 API 创建时间"，
    排查时序问题时极其误导。
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="创建时间（数据库侧生成，UTC）",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="最后更新时间（数据库侧生成，UTC）",
    )


class TenantScopedMixin:
    """租户隔离字段。

    所有租户内资源都必须有 tenant_id，且：
    - 有索引（过滤条件第一列）；
    - 有 RLS 策略（A3 建立）；
    - **由服务端注入**，绝不接受请求参数里的 tenant_id。

    这里只定义列，RLS 策略在迁移里创建。
    """

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="租户 ID（服务端从 token 注入，不取请求参数）",
    )
