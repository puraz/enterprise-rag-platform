"""身份提供者工厂：按配置选择实现。

## 为什么现在只有 dev

A7 才会接 Keycloak（OIDC + PKCE）。A4 需要能跑通受保护接口，
所以先提供 dev 实现。工厂的设计保证 A7 只需：

    1. 新增 `KeycloakIdentityProvider` 类
    2. 在 `_BUILDERS` 注册表里加一行
    3. 配置 `RAG_IDENTITY__PROVIDER=keycloak`

**业务代码不会改动** —— 它只依赖 `IdentityProvider` 抽象。

## 为什么用注册表而不是 if-else

`docs/PROMPT.md` 明确要求「见到 `if provider == "xxx"` 一律改成
能力匹配或策略注册」。注册表让新增实现不需要修改工厂逻辑本身，
也避免每加一个 IdP 就往 if 链上挂一节。

## 配置项

`RAG_IDENTITY__PROVIDER`（默认 `dev`）。

生产环境下若仍是 `dev`，`Settings` 的生产守卫会拒绝启动
（`dev_auth_enabled` 与 provider=dev 双重检查）。
"""

from __future__ import annotations

from collections.abc import Callable

from rag_api.identity.provider import DevIdentityProvider, IdentityProvider
from rag_core.errors import ConfigError
from rag_core.settings import Settings

# 提供者注册表：name -> 构造函数
# 构造函数接收 Settings，返回 IdentityProvider 实例。
# 用 ``Callable`` 而不是直接实例：不同实现可能需要不同的配置项，
# 延迟构造能让配置校验发生在真正使用时。
_BUILDERS: dict[str, Callable[[Settings], IdentityProvider]] = {
    "dev": lambda settings: DevIdentityProvider(),
    # A7 会加：
    # "keycloak": lambda s: KeycloakIdentityProvider(issuer=s.identity.issuer, ...),
    # "wecom": ...
}


def build_identity_provider(settings: Settings) -> IdentityProvider:
    """按配置构造身份提供者。

    配置错误（未知 provider）时抛 `ConfigError` 而不是静默回退 dev：
    静默回退会让"本想用 Keycloak 却实际用 dev"变成看不见的配置事故，
    而这个事故的后果是任何人都能登录。
    """
    provider_name = settings.identity.provider
    builder = _BUILDERS.get(provider_name)
    if builder is None:
        known = ", ".join(sorted(_BUILDERS))
        raise ConfigError(f"未知的身份提供者 {provider_name!r}（可选：{known}）")
    return builder(settings)


def available_providers() -> tuple[str, ...]:
    """当前支持的提供者名（供健康检查与诊断输出）。"""
    return tuple(sorted(_BUILDERS))
