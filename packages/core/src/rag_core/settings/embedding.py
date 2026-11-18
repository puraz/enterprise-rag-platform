"""Embedding 模型配置（远程 API 形态）。

## 为什么从"本地模型"换成"远程 API"

本地 bge-m3 需要 `sentence-transformers` + `torch`。PyPI 上的 torch 默认是
CUDA 构建，会连带拉 `nvidia-*` / `triton` 共 2GB+：一次镜像构建要十几分钟，
最终镜像 7GB，而实际推理跑在 CPU 上——这些 CUDA 运行时从未被使用。

换成 OpenAI 兼容的 embedding API（当前是智谱 `embedding-3`）后：
- 镜像不再需要 torch，构建从十几分钟降到一两分钟；
- 不再需要 2GB 模型缓存卷与 HF 镜像地址；
- 代价是每次入库/查询都走网络，且数据要离开进程（合规需评估）。

**插拔能力完整保留**：`provider` 是注册表键，换供应商只改配置；
所有实现都必须满足同一份 `EmbeddingProvider` 契约（维度、顺序、归一化）。

## 维度必须与数据库一致

`chunks.embedding` 是 `vector(1024)`（A3-3 固定）。配置里的 `dimension`
必须相同，否则入库报错。工厂会在构造时校验（`build_embedding_provider`）。

**换 provider / 换模型 = 换向量空间**：旧向量与新向量不可比较，
必须重建索引（A10 的版本化重建），否则检索结果会静默变差（不报错）。

## 为什么 `api_key` 用 SecretStr

配置对象会被 `safe_dump()` 打进启动日志。SecretStr 序列化成 `**********`，
避免"启动日志里躺着明文 key"这种最常见也最致命的泄漏方式。
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

#: 默认的智谱（BigModel）OpenAI 兼容端点。
#:
#: 智谱的 embeddings 接口与 OpenAI 的 `/v1/embeddings` 形状一致：
#: `{"model": ..., "input": [...] , "dimensions": 1024}` + Bearer 鉴权。
DEFAULT_EMBEDDING_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


class EmbeddingSettings(BaseModel):
    """Embedding 配置（远程 API）。"""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    provider: str = "zhipu"
    """实现名。对应 `rag_adapters.providers.factory` 的注册表键。

    可选：
    - `zhipu`：智谱 OpenAI 兼容端点（当前默认）；
    - `siliconflow`：硅基流动（同样是 OpenAI 兼容）；
    - `openai_compatible` / `openai`：任何提供 `/embeddings` 的服务
      （阿里百炼兼容模式 / vLLM / TEI 的 OpenAI 适配层等）。

    这几个键指向**同一个实现**（LangChain `OpenAIEmbeddings` 适配），
    区别只在 `base_url` 与模型名——形状相同的东西不该写多遍。
    """

    model_name: str = "embedding-3"
    """模型标识。由供应商定义，不要臆造。

    智谱 `embedding-3`：支持自定义输出维度（`dimensions` 参数）。
    """

    dimension: int = Field(default=1024, ge=1, le=4096)
    """输出维度。**必须与 `chunks.embedding` 的列定义一致**（当前 1024）。

    模型原生维度可能更大（例如 2048），通过 `dimensions` 参数降维后
    仍是同一向量空间的前缀投影，可与库里的 1024 维向量比较。
    """

    base_url: str = DEFAULT_EMBEDDING_BASE_URL
    """API 根地址（**包含版本段**，如 `.../api/paas/v4`）。

    实现内部会拼成 `{base_url}/embeddings`。
    """

    api_key: SecretStr | None = None
    """API Key。只走环境变量（`RAG_EMBEDDING__API_KEY`），不进库、不进日志。"""

    timeout_s: int = Field(default=30, ge=1, le=300)
    """单次 HTTP 请求超时。批量越大越需要放宽。"""

    batch_size: int = Field(default=32, ge=1, le=256)
    """单次请求携带多少条文本。

    批量太小 → 请求数线性增长、延迟翻倍；
    批量太大 → 单次请求的 token 数触顶（服务端 400），且失败时整批重试。

    32 是 1024 维中文短文本的经验值；服务端报"批量过大"时调小即可。
    """

    max_input_tokens: int = Field(default=8192, ge=1)
    """单条文本的长度上限（**按字符数近似**，由调用方/切分保证）。

    这里不是精确 tokenizer：它只是防止把整篇文档当一条发出去的护栏。
    真正的向量化对象是 chunk（≤ 800 字符），远小于该值。
    """

    send_dimensions: bool = True
    """是否在请求里带 `dimensions` 参数。

    支持自定义维度的模型（智谱 `embedding-3`）必须带，否则返回原生维度，
    与 `vector(1024)` 不匹配。个别自建服务不认这个字段，可关闭。
    """

    normalize: bool = True
    """是否对返回向量做 L2 归一化。

    **默认开**：归一化后余弦相似度 = 内积，与 pgvector 的
    `vector_cosine_ops` 语义一致，且未来切内积索引无需改数据。
    服务端多数不保证归一化，所以这一层必须自己做。
    """

    query_instruction: str | None = None
    """查询指令前缀（仅部分模型需要）。

    - 智谱 `embedding-3`：不需要，保持 None；
    - 某些 bge 系 API：需要 `"为这个句子生成表示以用于检索相关文章："`。

    加错前缀不会报错，只会让召回率下降——所以它是配置项而不是硬编码。
    """

    max_concurrency: int = Field(default=4, ge=1, le=64)
    """并发请求上限（ResilientCaller 的信号量）。

    远程 API 的瓶颈在服务端配额；并发过高只会换来 429。
    """

    unit_price_input: Decimal | None = None
    """输入 token 单价（USD / 1M tokens），用于 `model_calls` 成本入账。

    **默认 None（成本记 0）而不是猜一个价**：单价随合同/阶梯/活动变化，
    写死会在对账时造成系统性偏差。None 时仍记录 token 用量，
    补上单价后可按历史重算（与 LLM 的 `TokenPricing` 口径一致）。
    """

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        url = value.strip().rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"embedding.base_url 必须带协议，当前是 {value!r}")
        return url

    @field_validator("unit_price_input", mode="before")
    @classmethod
    def _empty_price_is_none(cls, value: object) -> object:
        """空字符串 → None。

        compose/容器环境里测一个变量时通常只能传空字符串，
        而 `Decimal | None` 不接受 ""。显式归一化避开了“配了个空价就起不来”。
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def configured(self) -> bool:
        """是否具备调用远程 API 的条件。"""
        return bool(self.api_key)

    @property
    def redacted_summary(self) -> dict[str, object]:
        """可安全打印的配置摘要。"""
        return {
            "provider": self.provider,
            "model": self.model_name,
            "base_url": self.base_url,
            "dimension": self.dimension,
            "batch_size": self.batch_size,
            "has_api_key": self.configured,
        }


__all__ = ["DEFAULT_EMBEDDING_BASE_URL", "EmbeddingSettings"]
