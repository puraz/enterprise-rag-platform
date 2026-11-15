"""四端共享枚举：按业务域分包组织。

为什么不是一个 `enums.py`：
业务表会增长到 30+ 张，枚举也会有几十个。单文件会变成
"滚动查找 + 重名冲突 + 改一处触发全员 review" 的负担，
而拆成域文件后，改文档枚举不会碰租户枚举。

## 组织规则

- **一个文件 = 一个领域聚合**，与 `rag_db/models/` 的模型文件大致对应
- 领域内所有枚举写在同一个文件里；真正跨领域共用的放 `common.py`
- **不按"前端页面"或"接口"分组**——枚举属于领域，不属于界面。
  按界面分会导致同一个状态在两个接口里定义两份，然后漂移。

当前域文件：

| 文件 | 内容 | 对应模型 |
|---|---|---|
| `tenancy.py` | 租户/用户/成员关系/平台角色 | `tenant.py` / `user.py` / `membership.py` |
| `common.py` | 跨域共用的基础枚举（后续按需建立） | — |

后续按 A3 阶段往下加：`documents.py`（文档状态/类型/密级）、
`tasks.py`（任务状态/类型）、`upload.py`（上传会话状态）、
`retrieval.py`（检索路由与策略）。**只在真需要时建文件，不预建空模块。**

## 命名规则

- 类名 PascalCase，域内**不加前缀**：写 `TenantStatus` 而不是 `TenancyTenantStatus`
- 值一律小写 snake_case
- 必须继承 `StrEnum`（不是 `str, Enum`）：可 JSON 序列化、可按字符串比较，
  且 `make gen-contracts` 能直接导出成 TS 字符串字面量联合类型

> 错误码是唯一的例外：它在 `rag_contracts.errors` 里，值用 `DOMAIN.REASON`
> 大写格式，因为那是**对外协议**的一部分，需要人眼快速分类。

## 加值的兼容性

前端会有 `switch` 分支，新增枚举值会落到 `default`。
**加值时必须同步检查前端是否需要处理**，不要只在后端加完就提交。
`tests/test_enums.py` 会确保新增枚举被导出到 TS 类型（漏导出会红）。
"""

from rag_contracts.enums.documents import (
    BlobStatus,
    ConfidentialityLevel,
    DocumentFormat,
    DocumentStatus,
    DocumentVersionStatus,
    KnowledgeBaseStatus,
    KnowledgeBaseVisibility,
    SourceType,
)
from rag_contracts.enums.retrieval import (
    ChunkingStrategy,
    ChunkType,
    DistanceMetric,
    IndexType,
    IndexVersionStatus,
    RetrievalMode,
)
from rag_contracts.enums.tenancy import (
    MembershipStatus,
    PlatformRole,
    TenantRole,
    TenantStatus,
    UserStatus,
)
from rag_contracts.enums.uploads import (
    OutboxStatus,
    TaskStatus,
    TaskType,
    UploadSessionStatus,
)

# RUF022 要求 __all__ 全局字母序，因此这里不按域分组。
# 域归属看上面的 import 段——那里是按域分组的，且带了注释。
__all__ = [
    "BlobStatus",
    "ChunkType",
    "ChunkingStrategy",
    "ConfidentialityLevel",
    "DistanceMetric",
    "DocumentFormat",
    "DocumentStatus",
    "DocumentVersionStatus",
    "IndexType",
    "IndexVersionStatus",
    "KnowledgeBaseStatus",
    "KnowledgeBaseVisibility",
    "MembershipStatus",
    "OutboxStatus",
    "PlatformRole",
    "RetrievalMode",
    "SourceType",
    "TaskStatus",
    "TaskType",
    "TenantRole",
    "TenantStatus",
    "UploadSessionStatus",
    "UserStatus",
]
