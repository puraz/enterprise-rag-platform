"""分块与检索域枚举。

对应模型：`rag_db.models.chunk`。
这些枚举同时被 A9（切分）与 A11（检索）使用，所以放在 contracts 里共享。
"""

from __future__ import annotations

from enum import StrEnum


class ChunkType(StrEnum):
    """chunk 的内容类型。

    整块不切的类型（TABLE / CODE / FORMULA）在切分时会被保护，
    不参与递归合并 —— 表格被切断会让行列错位，代码被切断会失去语义。
    """

    TEXT = "text"
    """普通正文。"""

    TITLE = "title"
    """标题（结构化切分产生，可作为上下文前缀）。"""

    TABLE = "table"
    """表格。整块保留，不按长度切。"""

    CODE = "code"
    """代码块。整块保留。"""

    FORMULA = "formula"
    """公式。整块保留。"""

    IMAGE_CAPTION = "image_caption"
    """图片说明（OCR 或版面分析产生）。"""

    LIST = "list"
    """列表项。"""


class ChunkingStrategy(StrEnum):
    """切分策略标识。

    值必须与 `rag_api.chunkers` 注册表里的名字一致。
    新增策略时要同时更新这里与注册表——不一致会导致 worker 找不到实现，
    报错信息是"未知策略"，但实际是名字拼写不同。
    """

    RECURSIVE = "recursive"
    """递归字符切分（baseline）。按分隔符优先级递归，最后按长度硬切。"""

    STRUCTURED = "structured"
    """标题层级结构化切分。按 Markdown/HTML 标题树切，保留章节路径。"""

    SEMANTIC = "semantic"
    """语义切分。按相邻句子向量相似度突变点切。"""

    SENTENCE_WINDOW = "sentence_window"
    """句子窗口。以句子为检索单元，附带上下文窗口（小块检索大块回填）。"""

    PARENT_CHILD = "parent_child"
    """父子文档。小块用于检索，大块（父块）用于回填。"""

    FIXED = "fixed"
    """固定长度滑窗。最简单，作为对照基线。"""


class IndexType(StrEnum):
    """向量索引类型。

    HNSW vs IVFFlat 的取舍：
    - HNSW：查询快、召回高、无需训练，但构建慢、内存占用大；
    - IVFFlat：构建快、内存小，但需要数据量足够才能训练好，
      且召回受 probes 影响明显。
    本项目默认 HNSW：知识库规模通常是十万到百万级 chunk，
    HNSW 的查询延迟优势更重要。
    """

    HNSW = "hnsw"
    IVFFLAT = "ivfflat"
    NONE = "none"
    """无向量索引（小数据集或对照实验）。"""


class DistanceMetric(StrEnum):
    """向量距离度量。

    **必须与 embedding 模型的训练目标一致**：
    - bge 系列、多数字段中文模型：用余弦（cosine）；
    - OpenAI text-embedding-3：用余弦；
    - 少数模型（如某些 e5 变体）训练时用内积。
    用错度量不会报错，只会让召回悄悄变差——这是最难发现的检索问题之一。
    """

    COSINE = "cosine"
    """余弦距离，`<=>` 操作符，值越小越相似。"""

    L2 = "l2"
    """欧氏距离，`<->` 操作符。"""

    INNER_PRODUCT = "inner_product"
    """负内积，`<#>` 操作符。pgvector 返回负内积以保持"越小越相似"。"""


class RetrievalMode(StrEnum):
    """检索模式（A11 用）。"""

    VECTOR = "vector"
    """纯向量检索。"""

    KEYWORD = "keyword"
    """纯关键词/BM25 检索。"""

    HYBRID = "hybrid"
    """混合检索 + RRF 融合。"""


class IndexVersionStatus(StrEnum):
    """索引版本状态（A10 零停机重建用）。

    双版本 + alias 切换的完整生命周期：

        BUILDING → READY → ACTIVE → RETIRED
                     │                 ▲
                     └──(切换前可回滚)──┘

    重建期间双写两个版本，切换瞬间只改 alias 指向。
    """

    BUILDING = "building"
    """构建中，不可查询（或仅影子查询）。"""

    READY = "ready"
    """构建完成，等待切换。"""

    ACTIVE = "active"
    """当前服务查询的版本。"""

    RETIRED = "retired"
    """已退役，保留一段时间供回滚，之后回收。"""

    FAILED = "failed"
    """构建失败。"""
