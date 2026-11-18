"""用户面 API v1 装配（`/api/v1/*`）。

## 边界（这条目录存在的理由）

用户面是**员工日常使用**的接口：登录、知识库、上传、文档、问答、会话。
它的鉴权是 `get_permission_context`（租户角色 + ACL + 密级），
**任何登录用户都可访问**，权限差异由角色能力表达。

管理面（`/api/v1/admin/*`）的鉴权完全不同（平台角色 ∪ 租户 owner/admin
+ 能力集），装配在 `rag_api.api.v1.admin`。两面**目录分开、装配分开**，
部署时网关按 `/api/v1/admin` 前缀禁止用户前端访问（见 main.py docstring）。

## 为什么用工厂函数而不是模块级 router

dev 登录端点是否注册取决于配置（且必须在生产关闭）。
模块级 router 单例会让"某个 app 实例注册过 dev 端点后，
同进程的其它 app 也能看到它"——测试里直接暴露，生产里更隐蔽。
所以每次 `create_app()` 都调一次本函数，拿到全新的 router。

## 路由顺序无关但分组有关

同一前缀下顺序不影响匹配（Starlette 按声明顺序，但这里的路径
没有前缀重叠歧义）。按**业务域**分组声明（auth → knowledge → uploads
→ documents → chat → tasks → me），让阅读者能一眼看到"这个面提供什么"。
"""

from __future__ import annotations

from fastapi import APIRouter

from rag_api.api.v1.user.auth import build_auth_router, build_me_router
from rag_api.api.v1.user.chat import router as chat_router
from rag_api.api.v1.user.documents import router as documents_router
from rag_api.api.v1.user.knowledge_bases import router as knowledge_bases_router
from rag_api.api.v1.user.tasks import router as tasks_router
from rag_api.api.v1.user.uploads import router as uploads_router
from rag_api.api.v1.user.usage import router as usage_router


def build_user_router(*, dev_enabled: bool) -> APIRouter:
    """构造用户面路由。

    Args:
        dev_enabled: 是否注册 dev 登录端点。生产环境必须为 False
            （`main._validate_startup_config` 会二次校验）。
    """
    api_v1 = APIRouter(prefix="/api/v1")

    # 认证：登录 / 刷新 / 登出 / 当前用户 / 租户列表与切换
    api_v1.include_router(build_auth_router(dev_enabled=dev_enabled))
    api_v1.include_router(build_me_router())

    # 知识库与文档（B4/B5）：KB CRUD + blob 登记为文档 + 下载 + 版本回滚
    api_v1.include_router(knowledge_bases_router)
    api_v1.include_router(documents_router)

    # 上传（A6）：分片上传的五个端点
    api_v1.include_router(uploads_router)

    # 问答与会话（A16）：SSE 流式 chat + 会话 CRUD + 反馈
    api_v1.include_router(chat_router)

    # 任务进度轮询：`POST /documents` 返回的 task_id 要有地方查
    api_v1.include_router(tasks_router)

    # 个人中心：用量与审计记录（员工看自己的数据）
    api_v1.include_router(usage_router)

    return api_v1


__all__ = ["build_user_router"]
