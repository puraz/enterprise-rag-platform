"""任务处理器注册表。

## 为什么集中在这里注册，而不是让 handler 自己"声明"

两种做法的差别在**可发现性**：

- 自注册（装饰器写进模块）：handler 必须被 import 才会注册。
  worker 启动时如果漏 import 某个模块，任务就"没有处理器"而进死信，
  且日志里只有一个 `no_handler_for_task`，看不出是漏 import。
- 集中注册（本文件）：启动时构造的 dict 就是**全部**能力。
  `worker_started` 日志里直接打出 `handlers=[...]`，
  部署后一眼能核对"我这个 worker 到底会处理哪些任务"。

代价是新增任务类型要改两处（写 handler + 这里注册）。
这个代价可接受：任务类型数量少且稳定（解析/切分/向量化/索引/清理），
而"漏注册导致任务静默进死信"的代价更大。

## 为什么 handler 在这里构造而不是每次调用时构造

handler 通常持有重资源（存储后端、embedding 模型）。
每个任务重建一次会：
- 不断新建 S3 client / 重载 2GB 模型（几十秒）；
- 泄漏连接（旧 client 未 close）。

所以 worker 启动时构造一次，进程生命周期内复用。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rag_core.logging import get_logger
from rag_worker.handlers.chunk import ChunkDocumentHandler
from rag_worker.handlers.embed import EmbedDocumentHandler
from rag_worker.handlers.parse import ParseDocumentHandler

if TYPE_CHECKING:
    from rag_adapters.providers.embedding import EmbeddingProvider
    from rag_adapters.storages import StorageBackend
    from rag_core.settings.embedding import EmbeddingSettings
    from rag_worker.engine import TaskHandler

log = get_logger(__name__)


def build_handlers(
    *,
    storages: dict[str, StorageBackend],
    embedding_provider: EmbeddingProvider,
    embedding_settings: EmbeddingSettings,
    tracer: Any = None,
) -> dict[str, TaskHandler]:
    """构造"任务类型 → handler"注册表。

    键用 `TaskType` 枚举的**值**（字符串），与数据库里存的一致。
    用枚举成员本身做键会在 `TaskType.PARSE_DOCUMENT` 与
    `"parse_document"` 之间产生比较陷阱（StrEnum 能过，普通 str 不行）。
    """
    from rag_contracts.enums import TaskType

    handlers: dict[str, TaskHandler] = {
        TaskType.PARSE_DOCUMENT.value: ParseDocumentHandler(storages=storages),
        TaskType.CHUNK_DOCUMENT.value: ChunkDocumentHandler(storages=storages),
        TaskType.EMBED_DOCUMENT.value: EmbedDocumentHandler(
            provider=embedding_provider, settings=embedding_settings, tracer=tracer
        ),
    }

    # 尚未实现的任务类型在这里显式列出，启动日志会打出"缺哪些"。
    # 不做成静默跳过：否则 A9 上线后忘了注册，表现是
    # "切分任务全部进死信"而不是"启动时就告诉你缺 handler"。
    implemented = set(handlers)
    pending = [t.value for t in TaskType if t.value not in implemented]
    if pending:
        log.warning(
            "handlers_not_implemented",
            pending=pending,
            note="这些任务类型进队后会直接进死信（不是静默失败）",
        )

    return handlers


__all__ = ["build_handlers"]
