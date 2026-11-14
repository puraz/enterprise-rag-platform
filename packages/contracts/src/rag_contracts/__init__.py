"""rag_contracts —— 四端唯一真相来源。

前端 TS 类型由后端 OpenAPI 生成，队列消息与错误码也在这里定义；
任何一端都不许"各写一份"。
"""

from importlib.metadata import version

__version__ = version("rag-contracts")
