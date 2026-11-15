"""Embedding 模型配置。

## 维度必须与数据库一致

`chunks.embedding` 是 `vector(1024)`（A3-3 固定）。配置里的 `dimension`
必须相同，否则入库报错。`LocalBGEEmbedding` 在初始化时会校验。

**改维度是 expand-contract 迁移**（A3-3 的 docstring 已写）：
新增列 → 双写 → 回填 → 切索引 → 删旧列。不要直接 `ALTER COLUMN TYPE`。

## 为什么不放在 LLMSettings 里

Embedding 与 LLM 是两类完全不同的能力：
- 计价方式不同（本地免费 / API 按 token）；
- 部署形态不同（本地模型占 2GB 内存，LLM 是远程 HTTP）；
- 更换频率不同（embedding 换了要重建全部索引，LLM 换了只影响新请求）。

放在一起会让人误以为"换 LLM 模型会同时换 embedding"——那是灾难性的
（所有向量失效）。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingSettings(BaseModel):
    """Embedding 模型配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    provider: str = "local_bge"
    """实现名。对应 `rag_adapters.providers.factory` 的注册表键。

    可选：`local_bge`（sentence-transformers）、`openai`（API）、`fake`（测试）。
    """

    model_name: str = "BAAI/bge-m3"
    """HuggingFace 模型标识或本地路径。

    `bge-m3`：1024 维，8192 token 上下文，多语言。**当前选择**。
    `BAAI/bge-large-zh-v1.5`：1024 维，512 token，纯中文更强但上下文短。
    """

    dimension: int = Field(default=1024, ge=1)
    """输出维度。**必须与 `chunks.embedding` 的列定义一致**（A3-3 是 1024）。"""

    device: str = "cpu"
    """`cpu` / `cuda` / `mps`。本机无 GPU（NVML 驱动不匹配），故默认 cpu。"""

    batch_size: int = Field(default=32, ge=1, le=256)
    """单次 encode 的批大小。

    太大会一次性占满内存（bge-m3 每条最长 8192 token，32 条已接近 1GB
    显存/内存）；太小则浪费批处理优势。32 是 CPU 上的经验值。
    """

    max_input_tokens: int = Field(default=8192, ge=1)
    """单条文本的最大 token 数。bge-m3 支持 8192。

    超长会被模型静默截断（损失语义），所以切分策略（A9）要保证
    chunk 不超过这个值。
    """

    cache_dir: str | None = None
    """模型缓存目录。None 用 HuggingFace 默认（~/.cache/huggingface）。"""

    hf_endpoint: str | None = None
    """HuggingFace 镜像地址。

    **国内必须配置**：直连 huggingface.co 会超时（实测）。
    用 `https://hf-mirror.com`。

    这个值会在模型加载前写入 `HF_ENDPOINT` 环境变量——
    sentence-transformers 通过该变量选择下载源。
    """

    query_instruction: str | None = None
    """查询时的指令前缀。

    `bge-m3` 不需要（官方说明），保持 None。
    换成 `bge-large-zh-v1.5` 时需要配：
    `"为这个句子生成表示以用于检索相关文章："`

    用错前缀不会报错，只会让召回率下降。
    """

    max_concurrency: int = Field(default=2, ge=1)
    """并发上限。

    CPU 上同时跑多个 embedding 请求只会互相抢核，吞吐反而下降。
    默认 2：一个跑批、一个等待，避免长任务饿死短任务。
    """
