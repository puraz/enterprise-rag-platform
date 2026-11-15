# A16 对话链路：多轮 RAG 的正确做法

> 日期：2026-11-15
> 复现：`uv run pytest tests/test_chat.py -q -m "integration or slow"`

## 1. 补的是一个真实缺口

课程的用户面接口清单（`docs/PROMPT.md` 第 68 行）写着：

> 用户面 `/api/v1/`：auth、knowledge、documents、**chat（SSE）、conversations**、feedback、me

但 A3 的表清单（第 297 行）**没有列 conversations/messages**，A4–A15 也没实现这两个接口。
所以后端一直缺"问答主链路"——这是 B3 对话页面的前提。

本次补齐：2 张表 + 1 个 SSE 端点 + 4 个会话端点 + 完整服务层。

## 2. 多轮对话对 RAG 的影响（核心设计）

### 问题：检索必须用独立化的问题，生成必须用完整历史

用户第 2 轮问 **"那二线呢？"**。这句话：

- **拿去检索** → 召回为零（不含任何可检索语义）；
- **拿去问 LLM** → 模型不知道在问什么（如果没带历史）。

正确做法是**两套输入**：

| 环节 | 用什么 | 实测证据 |
|---|---|---|
| 改写 / 检索 | **独立化问题** | `"那二线呢？"` → `"二线城市出差住宿每晚最多能报销多少？"` |
| 生成 | 历史 + 材料 + 独立化问题 | 回答正确提到 400 元 |

**独立化靠 LLM 做指代消解**（`ChatService._standalone_query`）。
提示词的三条约束：
1. 只输出改写结果（否则说明文字会一起送进检索）；
2. 若已完整则原样返回（防止画蛇添足）；
3. **禁止引入历史里没有的信息**（避免虚构出用户没问的内容）。

实测（`test_multi_turn_standalone_rewrite`）：

```
第 1 轮：一线城市出差住宿每晚最多能报销多少？
  → 答：600 元
  → 落库 standalone_query = "一线城市出差住宿每晚最多能报销多少？"

第 2 轮：那二线呢？
  → 落库 standalone_query = "二线城市出差住宿每晚最多能报销多少？"
  → 答：400 元  ← 独立化生效
```

### 坑一：历史不能无限增长

20 轮对话后 prompt 会撑爆上下文窗口。裁剪规则（`_trim_history`）：

- **从最旧开始丢**（最近的几轮才相关）；
- **必须成对丢弃**（2 条 = 1 轮）——只丢 `user` 留 `assistant`
  会让模型看到"凭空出现的回答"，开始瞎猜或复述；
- 预算是字符数（4000），中文约等于 token 数。

### 坑二：同一会话不能并发提问

两个请求同时基于"历史到第 4 条"去改写，各写第 5、6 条。
历史里这两条是**并列的**（互不知情），下一轮追问时模型看到错乱对话。

用 `SELECT FOR UPDATE NOWAIT` 锁会话行：
- **拿不到锁立即失败**（409），不是等待——等待会让用户以为卡住；
- **持锁只覆盖"分配序号 + 读历史"**，不覆盖 LLM 生成（几秒），
  否则追问都要排队。

### 坑三：流式历史里的引用不能混入当前答案

上一轮引用了文档 A，这一轮引用 B。`messages.citations` 只存本轮来源——
混入 A 会让用户以为本轮答案也有 A 的依据。

## 3. 序号分配的并发安全

朴素实现（`max(sequence) + 1`）在并发下两个请求拿到同一个号 → 撞唯一约束。
本实现用会话行的原子递增：

```sql
UPDATE conversations SET message_count = message_count + 1
WHERE id = :id RETURNING message_count
```

`message_count` 即"已分配的最大序号"——两个值天然同步。
实测 `test_sequence_allocation_is_atomic`：20 个并发分配，序号精确为 1..20 无重复。

## 4. SSE 的三个技术要点

1. **必须感知客户端断连**：`StreamingResponse` 断开时取消迭代器，
   捕获 `CancelledError` 先落库（保留部分内容）再 `raise`。
   不处理的话 LLM 会继续生成到结束（浪费钱），且消息永远停在 `streaming`。
2. **必须禁用代理缓冲**：`X-Accel-Buffering: no`（Nginx 认这个头，
   不需要改配置）。否则用户等 5 秒后所有 token 一次涌出。
3. **不能设 Content-Length**：SSE 是无限流，走 chunked 编码。

事件顺序：`retrieval`（先发引用，让等待期有内容）→ `delta` × N → `done`。

## 5. 审计 / 成本 / 限流 / 指标的接线

| 能力 | 接入点 | 实测结果 |
|---|---|---|
| **限流** | `enforce_chat_ratelimit`：租户级 20 QPS + 用户级 2 QPS | 双层，Redis 故障时放行（保护而非拒绝） |
| **审计** | `CHAT_COMPLETION` + `CONVERSATION` 目标，在 `finally` 里写 | 流结束/出错/断连都留痕 |
| **成本** | 每次 LLM 调用 → `model_calls`，另更新 message + conversation | chat + query_rewrite 分别记账 |
| **指标** | `record_llm_call` + `record_retrieval` | 接在 done 之前 |
| **日志** | request_id 已有；conversation_id/message_id 进日志字段 | — |

### 成本核算修的三个 bug（都是实测抓到的）

1. **`model_name` 属性不存在**：provider 只有私有 `_model_name`，
   导致 `getattr(llm, "model_name", llm.name)` 回退到供应商名 `"deepseek"`，
   而 pricing 表的键是 `deepseek-v4-pro` → 单价查不到 → **成本恒为 0**。
   修：`LLMProvider.model_name` 抽象属性 + 实现类覆盖。
2. **`ModelPricing` 有 peak/off_peak 两档**：直接读 `.input` 会 AttributeError
   （字段在 `TokenPricing` 上）。必须用 `pricing.price_at(now)`。
   DeepSeek 峰时价是谷时的 **2 倍**，用错档会让成本偏差一倍。
3. **缓存单价没取**：`input_cached` 是常规输入的 1/30（0.022 vs 0.66）。
   不传它会让命中 prompt cache 的调用被高估 30 倍。

修复后实测：`cost=$0.00031878`（357 输入 + 42 输出），
`model_calls` 里单价快照齐全（in=$0.66, out=$1.98）。

### 流式 usage 的获取（A5 的遗留缺口）

OpenAI 兼容协议只在**最后一个 chunk** 给 usage，而 `stream()` 只 yield 文本。
不收集的话流式路径成本永远记 0。

修：给 `stream()` 加 `usage_callback`（向后兼容的加法，
不改 yield 类型）。`LangChainLLMProvider.stream` 在 `finally` 里回调
最后一个 chunk 的 `usage_metadata`——异常时也回调（供应商可能已计费）。

## 6. 为什么不用 LangGraph checkpointer

**判断：不用，但保留扩展点。**

| 考量 | 结论 |
|---|---|
| checkpointer 解决什么 | 图执行中途中断/恢复（人工审批、长时间挂起） |
| 我们的场景 | 单次请求内跑完的线性链路 |
| 它带来的冲突 | 会话状态会序列化到 Redis/PG，而我们**自己就要存 messages 表**（B3 需要历史列表、点赞点踩、引用回溯）——两份状态会冲突，且"哪份是权威"变模糊 |

保留扩展点的具体做法：`RetrievalGraph.run()` 已支持 `history` 参数，
形状上兼容将来的 `StateGraph` 改造（自省不通过 → 重新检索的循环）。
**等真需要循环时再改，不为用框架而用框架。**

## 7. 遗留

1. **引用未做片段级定位**：流式生成时答案还没定稿，无法用 A12 的
   `locate_in_chunk` 定位"那句话"。当前引用高亮是整块。
   要精确定位需在生成完成后二次组装，但引用会延迟到流结束才出现
   （前端体验更差）。取舍记录在此。
2. **`refusal` 与 `citations 为空` 已分开**（契约层），但前端要按这个区分做 UI。
3. **历史摘要未实现**：超长对话会直接丢弃最旧的轮次，而不是压缩成摘要。
   摘要能让"20 轮前的关键约束"仍被记住，代价是一次额外 LLM 调用。
4. **限流未含配额（月度预算）**：`7xxxx` 错误码段已留，配额表未建。
5. **`_usage_from_stream` 取不到 usage 时成本记 0** 并打 warning——
   这是可观测的降级，不是静默失败。
