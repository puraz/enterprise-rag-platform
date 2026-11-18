"""检索链路配置：召回 / 融合 / 重排 / 自省。

## 为什么单独一个配置域

这些参数只影响"怎么检索"，不影响"用哪个模型"：

- `vector_limit` / `keyword_limit`：两路各召多少；
- `final_limit`：真正拼进 prompt 的有几条；
- `rrf_k`：融合的平滑常数；
- `use_*`：每个检索节点一个开关。

放在 `LLMSettings` 里会让人误以为"换模型"与"调召回"是同一件事；
放在代码常量里则无法在不改代码的情况下做 A/B 对比
（AGENTS.md 4.14：每个节点一个独立开关，评估时只改一个变量）。

## 默认值的取舍

| 开关 | 默认 | 理由 |
|---|---|---|
| `use_rewrite` | 开 | 多轮必需（"那二线呢"单独检索召回为零） |
| `use_keyword` | 开 | 精确词（编号、姓名、错误码）向量检索天然弱 |
| `use_rerank` | 关 | 当前只有 `NoOpReranker`（直通）；接远程 rerank API 后用评估验证增益再开 |
| `use_self_check` | 关 | 每次问答多一次 LLM 调用（约 500~1500ms）；对外客服场景应打开 |

`self_check_action` 决定"生成后自省未通过"时怎么处理：

- `flag`（默认）：保留答案，`done.degraded=true` + 审计/指标，供人工观察；
- `refuse`：把答案替换为拒答文案（流式已经发出的 token 无法收回，
  但 `done` 事件是权威状态，前端应据此覆盖）。

**不要默认 `refuse`**：自省模型本身也会误判，直接替换答案会让
"一次判断错误"变成"用户完全拿不到答案"。先用 `flag` 观察误报率。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


class RetrievalSettings(BaseModel):
    """检索链路配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    # ---- 节点开关 ----
    use_keyword: bool = True
    """关键词检索并与向量结果做 RRF 融合（混合检索）。"""

    use_rewrite: bool = True
    """多轮对话时用 LLM 做独立化改写（指代消解）。"""

    use_rerank: bool = False
    """用 cross-encoder 重排候选。

    需要 `RAG_RERANKER__PROVIDER` 不是 `none`；否则只会在融合顺序上
    做一次无意义的"全 0 分排序"。
    """

    use_self_check: bool = False
    """生成后判断"答案中的事实是否都有材料依据"（拦幻觉）。

    注意：这与生成前的 `answerability`（材料能否回答 → 决定是否拒答）
    是两个判断，不能混用（AGENTS.md 4.14）。
    """

    self_check_action: Literal["flag", "refuse"] = "flag"
    """自省未通过时的动作：只标记 / 替换为拒答。见模块 docstring。"""

    # ---- 召回 ----
    vector_limit: int = Field(default=20, ge=1, le=200)
    """向量路召回条数（融合前）。比最终 limit 大，给融合/重排留余量。"""

    keyword_limit: int = Field(default=20, ge=1, le=200)
    """关键词路召回条数。"""

    candidate_multiplier: int = Field(default=4, ge=1, le=10)
    """HNSW 候选倍数（第一段取 `limit × multiplier` 个候选）。

    ACL / 知识库过滤只能在第二段做，所以第一段必须多取。
    过滤极严（只允许 1% 文档）时 4 倍也可能不够，需要把条件下推
    到第一段（代价是失去 HNSW，见 ADR 0002）。
    """

    max_distance: float | None = Field(default=None, ge=0.0, le=2.0)
    """向量距离上限（余弦距离，0 最相关）。

    **默认 None（不截断）**：A11 实测"有答案/无答案"的距离区间重叠，
    任何阈值都会误判（见 ADR 0003）。这里保留参数用于特定场景
    （例如明确只要高置信命中），不用于拒答决策。
    """

    # ---- 融合 ----
    rrf_k: int = Field(default=60, ge=1, le=1000)
    """RRF 平滑常数，原论文默认 60。对结果影响很小，通常不用调。"""

    final_limit: int = Field(default=5, ge=1, le=50)
    """最终拼进 prompt 的材料条数。太多会稀释上下文并增加成本。"""

    # ---- 关键词提取 ----
    max_keywords: int = Field(default=8, ge=1, le=32)
    """单次查询最多提取多少个关键词。

    关键词之间是 OR 关系：词越多匹配越宽、排序越没有意义，
    且 SQL 的 OR 链越长。实测 12+ 个关键词会把延迟从 6ms 推到 650ms。
    """

    # ---- 重排 ----
    rerank_top_n: int = Field(default=10, ge=1, le=100)
    """只重排融合结果的前 N 条。

    重排是 O(n) 次模型推理；排在 50 名之后的候选本来也进不了
    `final_limit`。尾部结果保留但不重排（丢弃会伤召回）。
    """

    @model_validator(mode="after")
    def _check_limits(self) -> RetrievalSettings:
        """交叉校验：最终条数不能超过召回总数。"""
        if self.final_limit > self.vector_limit + self.keyword_limit:
            raise ValueError(
                "final_limit 不能大于 vector_limit + keyword_limit："
                "否则永远凑不够材料，配置本身是矛盾的"
            )
        return self

    @property
    def self_check_refuse(self) -> bool:
        """自省未通过时是否替换为拒答（供 `RetrievalConfig` 映射）。"""
        return self.use_self_check and self.self_check_action == "refuse"


class RerankerSettings(BaseModel):
    """Reranker（远程 rerank API）配置。

    ## 现在接的是 SiliconFlow 的 `/rerank`

    ```
    POST {base_url}/rerank
    {"model": "Qwen/Qwen3-Reranker-4B", "query": "...",
     "documents": ["..."], "return_documents": false, "top_n": N}
    → {"results": [{"index": 0, "relevance_score": 0.93}, ...]}
    ```

    这个形状与 Cohere 的 rerank 一致，但它**不是 OpenAI 协议的一部分**，
    LangChain 也没有通用的 OpenAI 兼容 rerank 封装
    （`langchain_community` 里只有 Cohere 与本地 CrossEncoder 两种实现，
    都依赖额外 SDK / torch）。所以 rerank 保留一个薄适配器：
    协议不变，换家只改 `base_url` + `model_name`。

    ## 为什么不放在 `EmbeddingSettings`

    Reranker 与 Embedding 是两种模型：
    - Embedding 一次编码一条文本，输出向量；
    - Reranker 一次编码一个 (query, document) 对，输出相关性分数。

    换 reranker 不影响已入库向量，换 embedding 则要重建全部索引。
    混在同一个配置域里，改一个字段时很难判断影响面。
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    provider: str = "none"
    """实现名。可选：`none`（直通）/ `siliconflow`（远程 /rerank）。"""

    model_name: str = "Qwen/Qwen3-Reranker-4B"
    """供应商模型标识（不臆造：这是当前实际使用的模型）。"""

    base_url: str = "https://api.siliconflow.cn/v1"
    """API 根地址（不包含 `/rerank`）。"""

    api_key: SecretStr | None = None
    """API Key。只走环境变量（`RAG_RERANKER__API_KEY`）。"""

    timeout_s: int = Field(default=30, ge=1, le=300)
    """单次请求超时。重排一次要跑 N 对，比 embedding 慢。"""

    batch_size: int = Field(default=8, ge=1, le=64)
    """单次请求的重排对数（批量上限）。"""

    max_pairs: int = Field(default=100, ge=1, le=1000)
    """单次（分批后）处理的最大对数，防止候选过多时一次性吃满配额。"""

    max_concurrency: int = Field(default=2, ge=1, le=32)
    """并发请求上限（ResilientCaller 信号量）。"""

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        url = value.strip().rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"reranker.base_url 必须带协议，当前是 {value!r}")
        return url

    @property
    def configured(self) -> bool:
        """是否具备调用远程 rerank 的条件。"""
        return self.provider != "none" and bool(self.api_key)

    @property
    def redacted_summary(self) -> dict[str, object]:
        """可安全打印的配置摘要。"""
        return {
            "provider": self.provider,
            "model": self.model_name,
            "base_url": self.base_url,
            "has_api_key": bool(self.api_key),
        }


__all__ = ["RerankerSettings", "RetrievalSettings"]
