"""可插拔适配层（api 与 worker 共用）。

## 这个包为什么存在

它装的是"和外部世界打交道"的三层实现，每一层都只面向 ABC + 工厂 + 配置：

| 子包 | 抽象 | 可替换的实现 |
|---|---|---|
| `parsers` | `BaseParser` | PDF/DOCX/PPTX/XLSX/MD/HTML/TXT/CSV |
| `storages` | `StorageBackend` | local / s3（AWS·MinIO·OSS·COS） |
| `providers` | `LLMProvider` / `EmbeddingProvider` | deepseek / openai 兼容 / bge |

## 为什么它不在 apps/api 里

两条硬约束在这里交叉：

1. worker 必须解析文档、下载对象、算 embedding——它是这些能力的**主要**消费者；
2. 「worker 禁止 import apps/api」是端边界，`tests/test_workspace.py` 静态检查。

而 core 规定「不含业务概念」，把解析器塞进 core 会让 core 依赖 pymupdf/torch。

所以适配层落地为独立共享包。判断标准很简单：**api 与 worker 都要用，
且不认识 ORM / 路由 / 请求上下文**。

## 依赖方向

    contracts ← core ← adapters ← api / worker

本包**不得** import `rag_db`：适配层只做协议转换，不读业务表。
需要"从库里取 blob 地址再下载"这类编排，放在调用方（worker handler / service）。
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
