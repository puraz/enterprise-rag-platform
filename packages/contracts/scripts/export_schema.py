"""契约导出：把 rag_contracts 里的 Pydantic 模型与枚举导出成 JSON Schema。

为什么要导出 JSON Schema 而不是直接拼 TS 字符串：
- JSON Schema 是**中间表示**，可以被多个消费者使用（TS 类型、前端 zod 校验、
  文档、Mock 生成器）。直接拼 TS 会把契约锁死在"只有 TS 能用"上。
- Pydantic 自己就是按 JSON Schema 规范导出的，不需要手写映射。

输出到 `packages/contracts/schema/`，入库——这样前端 clone 完不装 Python 也能用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from rag_contracts.enums import (
    AuditAction,
    AuditOutcome,
    AuditTargetType,
    BlobStatus,
    ChunkingStrategy,
    ChunkType,
    ConfidentialityLevel,
    DistanceMetric,
    DocumentFormat,
    DocumentStatus,
    DocumentVersionStatus,
    IndexType,
    IndexVersionStatus,
    KnowledgeBaseStatus,
    KnowledgeBaseVisibility,
    MembershipStatus,
    ModelCallStatus,
    ModelTaskType,
    OutboxStatus,
    PlatformRole,
    RetrievalMode,
    SourceType,
    TaskStatus,
    TaskType,
    TenantRole,
    TenantStatus,
    UploadSessionStatus,
    UserStatus,
)
from rag_contracts.errors import ErrorCategory, ErrorCode, ErrorResponse, FieldError, Severity

# 需要导出的模型。新增对外契约模型时加到这里。
EXPORTED_MODELS: tuple[type[BaseModel], ...] = (
    ErrorResponse,
    FieldError,
)

# 需要导出的枚举。导出成 TS 字符串联合类型，而不是 TS enum
# （TS enum 有运行时开销，且与字符串字面量类型不兼容）
#
# 枚举按业务域组织在 `rag_contracts.enums` 包内（tenancy.py / documents.py / …），
# 新增域文件后必须把其中的枚举登记到这里——否则前端拿不到类型。
# `tests/test_enums.py` 会自动扫描 enums 包校验是否漏登记，漏了会红。
EXPORTED_ENUMS: dict[str, type[str]] = {
    # ---- 错误码域（在 rag_contracts.errors，不是 enums 包）----
    "ErrorCode": ErrorCode,
    "ErrorCategory": ErrorCategory,
    "Severity": Severity,
    # ---- 租户/身份域（rag_contracts.enums.tenancy）----
    "TenantStatus": TenantStatus,
    "UserStatus": UserStatus,
    "TenantRole": TenantRole,
    "PlatformRole": PlatformRole,
    "MembershipStatus": MembershipStatus,
    # ---- 知识库/文档域（rag_contracts.enums.documents）----
    "KnowledgeBaseStatus": KnowledgeBaseStatus,
    "KnowledgeBaseVisibility": KnowledgeBaseVisibility,
    "ConfidentialityLevel": ConfidentialityLevel,
    "DocumentStatus": DocumentStatus,
    "DocumentVersionStatus": DocumentVersionStatus,
    "DocumentFormat": DocumentFormat,
    "SourceType": SourceType,
    "BlobStatus": BlobStatus,
    # ---- 检索/分块域（rag_contracts.enums.retrieval）----
    "ChunkType": ChunkType,
    "ChunkingStrategy": ChunkingStrategy,
    "IndexType": IndexType,
    "IndexVersionStatus": IndexVersionStatus,
    "DistanceMetric": DistanceMetric,
    "RetrievalMode": RetrievalMode,
    # ---- 上传/任务域（rag_contracts.enums.uploads）----
    "UploadSessionStatus": UploadSessionStatus,
    "TaskType": TaskType,
    "TaskStatus": TaskStatus,
    "OutboxStatus": OutboxStatus,
    # ---- 审计/成本域（rag_contracts.enums.audit）----
    "AuditAction": AuditAction,
    "AuditOutcome": AuditOutcome,
    "AuditTargetType": AuditTargetType,
    "ModelTaskType": ModelTaskType,
    "ModelCallStatus": ModelCallStatus,
}


def _enum_schema(name: str, enum_type: type[str]) -> dict[str, Any]:
    """把 StrEnum 导出成 JSON Schema 的 enum 定义。"""
    values = [member.value for member in enum_type]  # type: ignore[attr-defined]
    return {
        "title": name,
        "type": "string",
        "enum": values,
        "description": (enum_type.__doc__ or "").strip(),
    }


def build_schema_bundle() -> dict[str, Any]:
    """构建完整的 schema 包。

    结构：
        {
          "$schema": "...",
          "enums":  { "ErrorCode": {...}, ... },
          "models": { "ErrorResponse": {...}, ... }
        }

    故意把枚举与模型分开：生成 TS 时两者处理方式不同
    （枚举 → 字符串联合类型，模型 → interface）。
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "enums": {name: _enum_schema(name, enum) for name, enum in EXPORTED_ENUMS.items()},
        "models": {
            model.__name__: model.model_json_schema(ref_template="#/$defs/{model}")
            for model in EXPORTED_MODELS
        },
    }


def write_schema(output_dir: Path) -> Path:
    """写出 schema 文件，返回文件路径。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "contracts.schema.json"
    bundle = build_schema_bundle()
    # ensure_ascii=False：中文描述保持可读，review 时能看清是什么
    target.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return target


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    output = here / "schema"
    path = write_schema(output)
    print(f"已导出 schema: {path}")
    bundle = build_schema_bundle()
    print(f"  枚举: {len(bundle['enums'])} 个，模型: {len(bundle['models'])} 个")


if __name__ == "__main__":
    main()
