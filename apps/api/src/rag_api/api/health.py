"""健康检查：liveness 与 readiness。

## 为什么必须分开

| 端点 | 语义 | 失败时的编排器行为 |
|---|---|---|
| `/health/live` | 进程活着吗 | 重启容器 |
| `/health/ready` | 能接流量吗 | **摘除流量，不重启** |

合并成一个的后果：PG 短暂抖动时容器被重启，
而重启并不能修复 PG —— 结果是所有副本一起滚，故障放大成雪崩。

**readiness 失败不是 bug，是保护**：摘掉流量等依赖恢复，
恢复后自动加回来。

## readiness 检查什么

- PG 可连（`SELECT 1`）
- Redis 可连（`PING`）
- pgvector 扩展存在（缺了检索会静默变慢）

**不检查**外部 LLM 供应商：它们不可用时系统应该降级而不是整体不可用
（`docs/PROMPT.md` 的降级链要求）。供应商健康单独暴露在管理接口。
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from rag_contracts.api.auth import HealthResponse
from rag_core.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get(
    "/live",
    response_model=HealthResponse,
    summary="存活检查",
    description="只验证进程能响应，不检查依赖。失败时编排器应重启容器。",
)
async def liveness(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        version=request.app.state.app_version,
        environment=str(settings.env),
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="就绪检查",
    description=(
        "验证 PG / Redis / pgvector 可用。"
        "失败时编排器应摘除流量（**不要重启**）："
        "依赖故障重启容器没有帮助，只会放大故障。"
    ),
)
async def readiness(request: Request, response: Response) -> HealthResponse:
    settings = request.app.state.settings
    database = request.app.state.database
    redis = request.app.state.redis

    checks: dict[str, str] = {}

    try:
        await database.ping()
        extensions = await database.installed_extensions()
        checks["postgres"] = "ok"
        checks["pgvector"] = "ok" if "vector" in extensions else "missing"
    except Exception as exc:
        checks["postgres"] = f"error: {type(exc).__name__}"

    try:
        await redis.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {type(exc).__name__}"

    healthy = all(value == "ok" for value in checks.values())
    if not healthy:
        # 用 503：让负载均衡器/编排器明确知道不该转发流量
        response.status_code = 503
        log.warning("readiness_failed", checks=checks)

    return HealthResponse(
        status="ok" if healthy else "degraded",
        version=request.app.state.app_version,
        environment=str(settings.env),
        checks=checks,
    )
