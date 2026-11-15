"""测试辅助工具（模拟 IdP 等）。

**不打包进生产镜像**：这些模块只在测试与本地开发中使用。
打包排除由 `[tool.hatch.build.targets.wheel]` 的 include 规则控制
（当前实现未排除，因为它只有几百 KB，且便于集成测试引用）。
"""

from rag_api.testing.mock_oidc import MockOIDCServer

__all__ = ["MockOIDCServer"]
