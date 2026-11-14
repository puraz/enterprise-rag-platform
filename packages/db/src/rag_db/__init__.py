"""rag_db —— ORM 模型、仓储实现、数据库迁移的唯一入口。

为什么这个包存在（而不是把模型放 core 或 apps/api）：
- worker 必须读写 documents/tasks/chunks 等表，但禁止 import apps/api；
- core 的定位是"不含任何业务概念"，ORM 模型全是业务概念。

依赖方向：rag_contracts ← rag_core ← rag_db ← rag_api / rag_worker
"""

from importlib.metadata import version

__version__ = version("rag-db")
