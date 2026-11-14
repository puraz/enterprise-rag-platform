"""迁移链路自检表。

A2-6 只做迁移基础设施，还没有业务表。但"迁移能跑通"必须被**证明**，
不能靠"我觉得配好了"。这张表的作用就是给迁移链路一个可验证的落点：

- 它存在，说明 alembic upgrade head 真的执行了 DDL；
- 它的结构能被 downgrade 干净删除，说明回滚可用；
- 它的命名符合 NAMING_CONVENTION，说明约束命名约定生效。

表名带 `_meta` 前缀表明这是元数据表，不是业务数据。
它会在 A3 建真实业务表时保留——版本记录看 alembic_version，
这张表额外记录"这个库由哪个应用版本初始化"，用于跨环境排障。
"""

from __future__ import annotations

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from rag_db.base import Base, TimestampMixin


class SchemaHealth(Base, TimestampMixin):
    """记录 schema 初始化信息。

    只有一行，`key` 唯一。用途是排查"这个库到底是不是我们这个应用建的、
    是哪个版本建的"——多环境（dev/staging/prod）混用时这个问题很常见。
    """

    __tablename__ = "schema_health"

    key: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        comment="标识键，如 app_name",
    )
    value: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="键对应的值，如 enterprise-rag-platform",
    )
