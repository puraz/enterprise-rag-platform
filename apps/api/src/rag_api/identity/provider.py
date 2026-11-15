"""身份提供者抽象（IdentityProvider）。

## 为什么 A4 就要定义这个抽象

`docs/PROMPT.md` 要求「换 IdP 只应动一个适配层文件 + 一行配置」。
A4 需要 `get_current_user` 才能跑通任何受保护接口，而 A7 才接 Keycloak。
如果先写死 dev 登录、A7 再改，会牵动所有路由。

**所以现在定义接口 + 一个 dev 实现，A7 加 Keycloak/企业微信实现即可。**

## 抽象边界

`IdentityProvider` **只负责"把凭证换成外部身份"**：

    verify(token) -> ExternalIdentity(issuer, subject, email, ...)

它**不负责**：
- 本地用户映射（那是 service 层）；
- 签发本系统 JWT（那是 TokenService）；
- 权限判断（那是 ACL / RBAC 层）。

这样切分的原因：不同 IdP 的差异只在"如何校验凭证"，
而"校验后做什么"对所有 IdP 都一样。把后者放进抽象会让每个
实现重复同样的逻辑。

## 凭证形态的差异

| IdP | 凭证形态 |
|---|---|
| dev | 自定义不透明字符串 |
| Keycloak（SPA） | OIDC access token（JWT） |
| 企业微信 | 自定义 access_token + userid |
| 飞书 | tenant_access_token |

所以 `verify` 的入参是**原始字符串**，由实现自己解释格式。
接口不预设 JWT，否则不是所有 IdP 都能接入。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ExternalIdentity:
    """从 IdP 解析出的外部身份。

    字段设计对齐 OIDC 的标准声明，这样 Keycloak/通用 OIDC 实现
    可以直接填充，其它 IdP 也能映射过来。
    """

    issuer: str
    """IdP 标识（OIDC 的 iss）。用于构造 (issuer, subject) 唯一键。"""

    subject: str
    """IdP 侧的用户唯一标识（OIDC 的 sub）。"""

    email: str | None = None
    display_name: str | None = None
    # IdP 声明的租户/组织标识（部分 IdP 有，如 Keycloak 的 organization claim）。
    # 为 None 时由上层决定用哪个租户（例如从成员关系里选）。
    tenant_hint: str | None = None
    # IdP 下发的角色（可能与本地角色不同，需映射策略）
    roles: tuple[str, ...] = field(default_factory=tuple)
    # 原始 claims 里可公开的部分，供审计与排障。
    # **不得包含 token 本身或任何密钥**。
    attributes: dict[str, str] = field(default_factory=dict)


class IdentityProviderError(Exception):
    """身份校验失败的统一异常。

    不区分"token 格式错/签名错/过期"对内是必要的（审计需要），
    对外一律回 `AUTH.TOKEN_INVALID`，避免给攻击者反馈。
    """

    def __init__(self, message: str, *, reason: str = "invalid") -> None:
        super().__init__(message)
        # reason 只进日志与审计，不进响应
        self.reason = reason


class IdentityProvider(ABC):
    """身份提供者接口。

    实现约定：
    - `verify` 成功时返回 `ExternalIdentity`，失败抛 `IdentityProviderError`；
    - 实现类**不得**在 `verify` 里做数据库访问（保持职责单一、便于单测）；
    - 实现类必须声明 `name`，用于审计与日志（"这次是哪个 IdP 验的"）。
    """

    #: 提供者标识（如 "dev" / "keycloak" / "wecom"），进审计日志
    name: str = "unknown"

    @abstractmethod
    async def verify(self, credential: str) -> ExternalIdentity:
        """校验凭证并返回外部身份。

        Args:
            credential: 原始凭证字符串（Bearer token 的值）。

        Raises:
            IdentityProviderError: 校验失败（格式/签名/过期/撤销）。
        """
        raise NotImplementedError

    async def health(self) -> bool:
        """提供者是否可用（如 JWKS 端点可达）。

        默认返回 True：不依赖外部服务的实现（如 dev）无需健康检查。
        有外部依赖的实现（如 Keycloak）应覆盖它。
        """
        return True


class DevIdentityProvider(IdentityProvider):
    """开发用身份提供者：把 token 当成 `issuer:subject` 直接解析。

    **仅用于本地开发与自动化测试**，绝不允许在生产启用。

    ## 三重防误入生产

    1. `Settings.env` 是 `RunEnv` 枚举，写 `production` 这种拼写
       会在**启动时**被 Pydantic 拒绝（不是当成非生产放过）；
    2. `dev_auth_enabled` 默认 `False`，必须显式设置环境变量开启；
    3. `Settings` 的生产守卫里，`staging`/`prod` + `dev_auth_enabled=true`
       **直接启动失败**（而不是记个警告）。

    路由层还会再检查一次（`register_dev_routes` 只在显式开启时挂载），
    所以即使前两层都失效，dev 端点也不会存在。

    ## token 格式

        dev:<subject>                   -> issuer=dev, subject=<subject>
        dev:<issuer>:<subject>          -> 指定 issuer（测多 IdP 场景）

    例：`dev:alice` → `issuer=dev, subject=alice`，
    邮箱等属性按 subject 生成，方便本地调试。
    """

    name = "dev"

    async def verify(self, credential: str) -> ExternalIdentity:
        if not credential.startswith("dev:"):
            raise IdentityProviderError(
                "dev provider 只接受 'dev:<subject>' 格式的凭证", reason="bad_format"
            )

        parts = credential.split(":")
        if len(parts) == 2:
            issuer, subject = "dev", parts[1]
        elif len(parts) == 3:
            _, issuer, subject = parts
        else:
            raise IdentityProviderError("凭证格式错误", reason="bad_format")

        if not subject:
            raise IdentityProviderError("subject 不能为空", reason="empty_subject")

        return ExternalIdentity(
            issuer=issuer,
            subject=subject,
            email=f"{subject}@dev.local",
            display_name=subject,
            attributes={"provider": "dev"},
        )
