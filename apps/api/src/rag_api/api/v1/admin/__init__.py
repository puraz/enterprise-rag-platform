"""管理面：运营看板、任务台账、文档台账、索引版本、chunk 预览。

## 这个目录与用户面的分工（目录存在的意义）

| | 用户面 `v1/user/` | 管理面 `v1/admin/` |
|---|---|---|
| 读者 | 员工 | 运营 / 审计员 / 平台超管 / 租户管理员 |
| 鉴权 | `get_permission_context`（租户角色 + ACL） | `get_admin_context`（平台 ∪ 租户管理员） |
| 数据范围 | 自己 + 自己可见的库 | 租户内全部（平台角色可跨租户） |
| URL | `/api/v1/*` | `/api/v1/admin/*` |
| 网关 | 用户前端可访问 | **禁止从用户前端访问** |

**每个路由都必须显式声明能力**（`READ` / `OPERATE` / `AUDIT` / `ADMIN`），
`tests/test_api_layering.py` 会检查 admin 包下没有任何路由漏挂管理依赖。
漏挂的表现是"任何登录用户都能调管理接口"，而且返回 200 完全看不出来。

## 为什么管理面不返回文档正文

见 `rag_contracts.api.admin` 的模块 docstring：文档台账只给元数据。
唯一的例外是 chunk 预览（切分质量检查必须看正文），
且该接口单独审计、单独能力（`OPERATE`，不是 `READ`）。
"""

from __future__ import annotations

from fastapi import APIRouter

from rag_api.api.v1.admin.audit import router as audit_router
from rag_api.api.v1.admin.dashboard import router as dashboard_router
from rag_api.api.v1.admin.documents import index_router
from rag_api.api.v1.admin.documents import router as documents_router
from rag_api.api.v1.admin.indexing import router as indexing_router
from rag_api.api.v1.admin.tasks import router as tasks_router


def build_admin_router() -> APIRouter:
    """构造管理面路由（`/api/v1/admin/*`）。

    ## 为什么这里没有顶层 `dependencies=[...]`

    读者可能期望在 router 级挂一个"必须是管理面用户"的依赖，
    让所有子路由自动受保护。**不这么做**的原因：

    router 级依赖只校验"是管理面用户"，不校验**具体能力**。
    挂着它会让"审计员能不能重试任务"这类问题退化成
    "每个接口自己记得调 ctx.require(OPERATE)"——而那正是
    我们要消除的漏判来源。

    所以每个路由用 `dependencies=[Depends(require_admin_capability(X))]`
    显式声明能力，判定逻辑集中在 `permissions/admin.py`。
    """
    api_v1 = APIRouter(prefix="/api/v1/admin")

    # 运营看板：计数、成本拆解、死信数（C2）
    api_v1.include_router(dashboard_router)
    # 任务台账与死信重试（C3）
    api_v1.include_router(tasks_router)
    # 文档台账、版本、chunk 预览（C3）
    api_v1.include_router(documents_router)
    # 索引版本与 alias 状态（C3，只读）
    api_v1.include_router(index_router)
    # 索引重建 / 切换 / 回滚（C3，写：三段独立接口）
    api_v1.include_router(indexing_router)
    # 安全审计查询与链验证（C5）
    api_v1.include_router(audit_router)

    return api_v1


__all__ = ["build_admin_router"]
