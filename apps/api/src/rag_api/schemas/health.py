"""健康检查响应实体与映射。

契约形状来自 `rag_contracts.api.auth`；本模块只做 re-export 与
domain → 响应实体的映射。
"""

from __future__ import annotations

from rag_contracts.api.auth import HealthResponse


def to_response(
    *,
    status: str,
    version: str,
    environment: str,
    checks: dict[str, str] | None = None,
) -> HealthResponse:
    """健康检查结果 → 响应实体。"""
    return HealthResponse(
        status=status,
        version=version,
        environment=environment,
        checks=checks or {},
    )


__all__ = ["HealthResponse", "to_response"]
