"""本地 bge-m3 Embedding 实现（sentence-transformers）。

## 为什么用本地模型

`docs/PROMPT.md` 要求"Embedding / Reranker 优先本地方案降低开销"。
一份 300 页 PDF 切出 2000 个 chunk，如果用 API embedding，每次入库都
要付费；而文档重索引（A10）会反复调用。本地模型一次下载，长期零成本。

## 维度不变量

`bge-m3` 输出 **1024 维**，与 A3-3 的 `chunks.embedding vector(1024)` 一致。
这个必须校验：维度不匹配时写入会报
`expected 1024 dimensions, not 768`，但那时已经切分完几百个 chunk，
重跑成本高。所以在 `__init__` 就校验并明确报错。

## CPU 上的现实性能（本机 16 核，无 GPU）

实测数据（详见测试输出）：
- 短文本（<100 字）批量 32 条：约 40-60 条/秒
- 长文本（500 字）批量 8 条：约 8-15 条/秒

这意味着 2000 chunk 的文档约需 30-50 秒。对入库是可接受的
（异步后台任务），但**不能阻塞事件循环**：必须 `run_in_executor` 或
`to_thread`。

## 模型加载的并发问题

`SentenceTransformer` 加载模型约需 2-5 秒、占用 2.2GB 内存。
每个 worker 进程各一份，所以：
- **不要在 `__init__` 里同步加载**（会阻塞启动几十秒且无法并发）；
- 用 `asyncio.Lock` 保证只加载一次（多协程并发调用时）；
- 延迟到首次使用时加载（`_ensure_loaded`）。

## 归一化

`normalize_embeddings=True` 让向量长度为 1，此时**余弦相似度 = 内积**。
这带来两个好处：
1. 计算更快（省去模长除法）；
2. 与 `vector_cosine_ops` 索引语义一致（A3-3 用的是 cosine）。

不做归一化也能用 cosine 索引（pgvector 会自己算），但归一化后
未来若切到 `vector_ip_ops`（内积索引）也不用改数据。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from rag_adapters.providers.embedding import (
    EmbeddingCapabilities,
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingResult,
    EmbeddingTask,
)
from rag_core.logging import get_logger
from rag_core.settings.embedding import EmbeddingSettings

log = get_logger(__name__)

# bge-m3 的向量维度。与 A3-3 的 vector(1024) 必须一致。
BGE_M3_DIMENSION = 1024


class LocalBGEEmbedding(EmbeddingProvider):
    """本地 bge 系列模型（sentence-transformers）。

    支持 `bge-m3` / `bge-large-zh-v1.5` 等 bge 模型：
    差异只在模型名与是否加查询指令前缀，其余逻辑相同。
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings
        self.name = settings.model_name
        self._model: Any = None  # SentenceTransformer，延迟加载
        self._load_lock = asyncio.Lock()
        self._expected_dimension = settings.dimension

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            dimension=self._expected_dimension,
            max_input_tokens=self._settings.max_input_tokens,
            supports_batch=True,
            is_local=True,
            requires_query_instruction=self._settings.query_instruction is not None,
        )

    async def _ensure_loaded(self) -> Any:
        """延迟加载模型。用锁保证只加载一次。

        **不在这里做 CPU 密集的加载而不切线程**：模型加载是同步阻塞的
        （读 2.2GB 权重 + 构建计算图），直接 await 会阻塞整个事件循环，
        期间所有请求都卡住。
        """
        # 用局部变量而不是直接读 self._model：mypy 对 `Any` 类型的属性
        # 收窄行为不一致（会把双击检查的第二处 return 判为不可达）。
        loaded: Any = self._model
        if loaded is not None:
            return loaded

        async with self._load_lock:
            # 双重检查：等锁期间可能已被其它协程加载
            loaded = self._model
            if loaded is not None:
                return loaded

            started = time.perf_counter()
            try:
                # 同步加载放到线程池，避免阻塞事件循环
                self._model = await asyncio.to_thread(self._load_model)
            except Exception as exc:
                raise EmbeddingProviderError(
                    f"加载模型 {self._settings.model_name} 失败", retryable=False
                ) from exc

            elapsed = time.perf_counter() - started
            log.info(
                "embedding_model_loaded",
                model=self._settings.model_name,
                device=self._settings.device,
                elapsed_s=round(elapsed, 1),
                dimension=self._expected_dimension,
            )
            return self._model

    def _load_model(self) -> Any:
        """同步加载（在线程池里执行）。

        **必须在 import sentence_transformers 之前设置 HF_ENDPOINT**：
        它读取该环境变量决定下载源。直连 huggingface.co 在国内会超时
        （实测 5 次重试后失败），必须用镜像。
        """
        import os

        if self._settings.hf_endpoint:
            os.environ["HF_ENDPOINT"] = self._settings.hf_endpoint

        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {"device": self._settings.device}
        if self._settings.cache_dir is not None:
            kwargs["cache_folder"] = self._settings.cache_dir
        return SentenceTransformer(self._settings.model_name, **kwargs)

    async def embed_batch(
        self,
        texts: list[str],
        *,
        task: EmbeddingTask = EmbeddingTask.DOCUMENT,
    ) -> EmbeddingResult:
        """批量生成向量。

        空输入返回空结果（不报错）：空文档切分后可能没有 chunk，
        这不该让整条管线失败。
        """
        if not texts:
            return EmbeddingResult(vectors=[], dimension=self._expected_dimension, model=self.name)

        # 查询指令前缀：bge-*-zh-v1.5 需要，bge-m3 不需要
        prepared = self._apply_instruction(texts, task)

        model = await self._ensure_loaded()
        started = time.perf_counter()
        try:
            # encode 是同步 CPU 密集操作，必须切线程
            raw = await asyncio.to_thread(
                model.encode,
                prepared,
                batch_size=self._settings.batch_size,
                normalize_embeddings=True,  # 归一化：余弦相似度 = 内积
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise EmbeddingProviderError("embedding 计算失败", retryable=True) from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        vectors = [row.tolist() for row in raw]

        # **维度校验必须在返回前做**：维度不符会让入库报错，
        # 但那时已经切分完几百个 chunk。这里失败能立刻定位到配置问题。
        if vectors and len(vectors[0]) != self._expected_dimension:
            raise EmbeddingProviderError(
                f"模型输出维度 {len(vectors[0])} 与配置的 {self._expected_dimension} 不一致，"
                f"请检查 RAG_EMBEDDING__DIMENSION 与 chunks.embedding 的列定义",
                retryable=False,
            )

        # **顺序校验**：错位会导致"检索能跑但相关性全乱"，且不报错
        if len(vectors) != len(texts):
            raise EmbeddingProviderError(
                f"返回向量数 {len(vectors)} 与输入条数 {len(texts)} 不一致",
                retryable=False,
            )

        return EmbeddingResult(
            vectors=vectors,
            dimension=self._expected_dimension,
            model=self.name,
            latency_ms=latency_ms,
        )

    def _apply_instruction(self, texts: list[str], task: EmbeddingTask) -> list[str]:
        """按任务类型加指令前缀。

        bge-m3 **不需要**前缀（官方说明），配置里 `query_instruction=None`
        时跳过。换成 bge-*-zh-v1.5 时在配置里配上即可，代码不用改。

        用错前缀不会报错，只会让向量空间偏移、召回率下降——
        所以这个开关放在配置里而不是硬编码。
        """
        instruction = self._settings.query_instruction
        if instruction is None or task is not EmbeddingTask.QUERY:
            return texts
        return [f"{instruction}{text}" for text in texts]

    async def health(self) -> bool:
        """探活：确认模型能加载并产出正确维度的向量。

        用极短文本（"hi"）让成本可忽略。首次调用会触发模型加载。
        """
        try:
            result = await self.embed_batch(["hi"])
            return bool(result.vectors) and len(result.vectors[0]) == self._expected_dimension
        except Exception as exc:
            log.warning("embedding_health_failed", error=type(exc).__name__)
            return False

    async def aclose(self) -> None:
        """释放模型占用的内存。

        显式置 None 并触发 GC：2.2GB 的模型不释放会让 worker 重启时
        短时间占用双倍内存。
        """
        self._model = None
        import gc

        gc.collect()
