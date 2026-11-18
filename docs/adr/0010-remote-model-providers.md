# ADR 0010：远程模型 Provider（LangChain Embedding + Cohere 风格 Rerank）

> 状态：已采纳
> 日期：2026-11-17
> 关联：ADR 0003（检索层）、ADR 0006（可观测性）、ADR 0009（混合检索）

## 背景

A11 的检索链路需要 embedding 与 rerank。原实现是本地模型：

- `LocalBGEEmbedding`（bge-m3，sentence-transformers）
- `LocalBGEReranker`（bge-reranker，CrossEncoder）

它们把 `torch` 变成硬依赖。PyPI 上的 torch 默认是 **CUDA 构建**，
连带 `nvidia-*` / `triton` 共 2GB+：

| 指标 | 实测 |
|---|---|
| 单次镜像构建下载量 | ~2.5GB（其中 nvidia-* ≈2.3GB） |
| 镜像体积 | 20.3GB / 7GB（含/不含层复用） |
| 构建耗时 | 十几分钟（国内直连 pypi） |

而本项目的 embedding 配置是 `device=cpu`，GPU 只是可选 profile。
**为从未使用的 CUDA 运行时付出构建成本**，是明显的负收益。

同时团队已有云厂商额度（智谱 `embedding-3`、SiliconFlow `Qwen3-Reranker-4B`），
切换后：

- 镜像不再包含 torch/CUDA，构建降到一两分钟、体积降到几百 MB；
- 不再需要 2GB 模型缓存卷与 HuggingFace 镜像地址；
- 代价是向量化/重排要走网络，且文档内容会发给第三方（合规需评估）。

## 决策

### 1. Embedding 用 LangChain `OpenAIEmbeddings`

`/embeddings` 是 OpenAI 协议的一部分，LangChain 已封装鉴权、批量、
维度参数与错误类型。自研第二套的结局是"某个字段只在一家实现里修好"。

我们只保留一层薄适配器（`LangChainEmbedding`），负责 LangChain 不做的事：

| 事项 | 为什么必须自己做 |
|---|---|
| `check_embedding_ctx_length=False` | 默认 True 会用 tiktoken 把文本转成 token id 数组发出，第三方端点不认 |
| `max_retries=0` | 与 `ResilientCaller` 叠加会变成 9 次重试，且熔断器看不到真实失败节奏 |
| 独立 `http_async_client` | 共享 client 时某个实例 `aclose()` 会让其它实例抛 `Event loop is closed` |
| L2 归一化 | 服务端不保证单位向量；归一化后余弦 = 内积，与 `vector_cosine_ops` 一致 |
| 维度/条数校验 | 服务端忽略 `dimensions` 时，会在**批量入库**才暴露 |

**已知取舍**：`check_embedding_ctx_length=False` 时，LangChain 按响应
`data` 顺序装配向量，不读 `index`（`langchain_openai/embeddings/base.py:750`）。
所有 OpenAI 兼容端点都按输入顺序返回，所以实践上成立；这条假设写进了
模块 docstring，并由单测锁住实际行为（框架若改变会立刻失败）。

### 2. Rerank 自写一个 Cohere 风格适配器

**rerank 不属于 OpenAI 协议**，LangChain 没有通用远程封装。实测
`langchain_community`（官方已 sunset）里的实现都不适用：

| 类 | 不适用原因 |
|---|---|
| `InfinityRerank` | 硬编码 `localhost:7997`；需 `infinity_client`；需自行注入 client |
| `JinaRerank` | 硬编码 Jina 端点；无 base_url 参数 |
| `FlashrankRerank` | 本地模型（重） |
| `CohereRerank` | 0.4.x 已移除；绑定 cohere SDK |

而且它们都是 **sync 的 `Document` 压缩器**，默认 `top_n=3` 会丢候选并打乱
输入顺序；检索图需要"每个输入一条分数 + 保留尾部"。

因此实现 `CohereStyleReranker`（约 100 行）：协议只有四个字段，
换供应商只改 `base_url + model_name`，与 embedding 的插拔体验一致。

### 3. 移除本地模型与 GPU 覆盖

- 删除 `LocalBGEEmbedding` / `LocalBGEReranker` 与 `torch` 依赖；
- 删除 `deploy/docker/compose.gpu.yml`（镜像已无 CUDA）；
- 工厂注册表保留：新增 provider 仍是"一个实现 + 一行注册"。

### 4. 换模型必须重建向量（零停机）

不同模型产出不同向量空间，旧向量与新查询不可比较。`scripts/reembed_all.py`
（`make docker-reembed`）按 A10 做版本化重建：

1. 新建索引版本（`building`，模型 = 当前配置）；
2. embed 任务 payload 带 `index_version_id`，向量写入 building 版本，
   **ACTIVE 不变**——回填期间检索完全不受影响；
3. 轮询新版本的 chunk 覆盖数，达到当前 chunk 总数才切换 ACTIVE；
4. 超时/失败**不切换**，旧索引继续服务；旧版本置 `retired` 可回滚。

模型未变但想重建（调索引参数/演练）用 `--force-new-version`；
只建版本不切换用 `--no-wait`。

## 成本入账

embedding / rerank 与 LLM 走同一张 `model_calls` 明细表：

| task_type | provider | 用量来源 | trace |
|---|---|---|---|
| `embedding` | `zhipu` | `usage.prompt_tokens` | 有（API 检索与 worker 任务都带） |
| `rerank` | `siliconflow` | `meta.tokens.input_tokens`（兼容多种形状） | 有 |

单价默认 `None`：只记 token、成本 0，避免拿错价格做对账；
配置 `RAG_EMBEDDING__UNIT_PRICE_INPUT` / `RAG_RERANKER__UNIT_PRICE_INPUT`
后计入成本，历史记录也能按新单价重算。
消息/会话的 `cost_usd` 现在包含本轮全部辅助调用
（改写、能否回答、查询 embedding、rerank），不再只算生成那一次。

## 后果

### 正面

- 镜像从 7GB 降到几百 MB，构建时间从十几分钟降到一两分钟；
- 换供应商 = 改配置（base_url / model / dimension / key），不改代码；
- 重排链路真正跑起来（此前 `use_rerank` 无可用实现）。

### 负面 / 需要接受

- **网络依赖**：embedding/rerank 不可用时，入库与问答受影响。降级链：
  rerank 失败 → RRF 顺序；embedding 失败 → 任务重试（可恢复）或死信（配置错误）。
- **数据出境**：文档正文会发给第三方。高敏场景需要单独评估或换自建 TEI/vLLM
  （它们同样兼容 `/embeddings`，只改配置）。
- **成本模型变化**：从"一次性模型下载"变成"按 token 计费"。
  embedding / rerank 的用量已写入 `model_calls`（`task_type=embedding|rerank`，
  带 trace_id），与 LLM 同口径；单价默认 `None`（只记 token、成本记 0），
  按合同配置 `RAG_EMBEDDING__UNIT_PRICE_INPUT` / `RAG_RERANKER__UNIT_PRICE_INPUT`
  （USD / 1M tokens）后即可计入消息与会话成本，并可按历史重算。

## 被否掉的替代方案

| 方案 | 否决理由 |
|---|---|
| 继续用 CUDA torch | 2.3GB 下载换一个从未启用的能力 |
| 锁到 CPU-only torch | 保留本地模型，但 GPU 用户需要第二份 lock，维护成本高 |
| 用 `InfinityRerank` 接 SiliconFlow | 需引入 `infinity_client`、注入 client，胶水比协议适配器更多 |
| 直接调 `openai` SDK 写 embedding | 与"用框架"的目标相反，且要重写批量/重试/维度逻辑 |
