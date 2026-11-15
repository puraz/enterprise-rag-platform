"""依赖注入包：按用途分层，统一从这里导出。

## 为什么按用途拆而不是按文件变大再拆

依赖之间是有**依赖链**的：

    app（进程级单例）
      └→ auth（解析 token）
           └→ database（用 tenant_id 开事务）

按这个层次拆文件，读代码时能直接看出"谁能依赖谁"。
如果只按"函数多了就分文件"来拆（比如按首字母），
依赖方向就会被藏起来，容易出现"database 依赖 auth，auth 又依赖 database"的环。

## 文件职责

| 文件 | 内容 | 有状态？ |
|---|---|---|
| `app.py` | 进程级单例（DB/Redis/TokenService/配置） | 无（读 app.state） |
| `auth.py` | 身份解析、平台角色校验 | 无（只读 token） |
| `database.py` | 会话（租户上下文 / 普通 / 跨租户） | 无（由 Database 管） |
| `pagination.py` | 分页参数解析 | 无 |

**业务代码统一 `from rag_api.deps import get_db`**，
不要 import 内部模块——这样以后调整文件划分不影响调用方。
"""

from rag_api.deps.app import (
    get_database,
    get_redis,
    get_settings_from_app,
    get_token_service,
)
from rag_api.deps.auth import Principal, get_current_principal, require_platform_role
from rag_api.deps.database import get_admin_db, get_db, get_session
from rag_api.deps.pagination import PageQuery, page_query

__all__ = [
    "PageQuery",
    "Principal",
    "get_admin_db",
    "get_current_principal",
    "get_database",
    "get_db",
    "get_redis",
    "get_session",
    "get_settings_from_app",
    "get_token_service",
    "page_query",
    "require_platform_role",
]
