"""ORM 模型集合。

所有模型模块必须在这里 import 一次，否则 Alembic 的 autogenerate
看不到它们（metadata 只包含被 import 过的模型），后果是"改了模型但
生成不出迁移"，且没有任何报错。

按域分组，新增模型时加进对应分组。
"""

from rag_db.models.blob import Blob
from rag_db.models.document import Document, DocumentVersion
from rag_db.models.health import SchemaHealth
from rag_db.models.knowledge import KnowledgeBase
from rag_db.models.membership import TenantMembership
from rag_db.models.tenant import Tenant
from rag_db.models.user import IdentityLink, User

__all__ = [
    "Blob",
    "Document",
    "DocumentVersion",
    "IdentityLink",
    "KnowledgeBase",
    "SchemaHealth",
    "Tenant",
    "TenantMembership",
    "User",
]
