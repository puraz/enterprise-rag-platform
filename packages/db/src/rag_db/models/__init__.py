"""ORM 模型集合。

所有模型模块必须在这里 import 一次，否则 Alembic 的 autogenerate
看不到它们（metadata 只包含被 import 过的模型），后果是"改了模型但
生成不出迁移"，且没有任何报错。

A3 按域拆文件后，继续在这里逐个 import。
"""

from rag_db.models.health import SchemaHealth

__all__ = ["SchemaHealth"]
