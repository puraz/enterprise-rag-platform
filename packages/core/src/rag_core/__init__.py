"""rag_core —— api 与 worker 共用的横切基础设施，不含任何业务概念。

依赖方向：只被 api / worker 依赖，本包不反向依赖任何上层。
"""

from importlib.metadata import version

# 版本号从安装元数据读取，避免和 pyproject.toml 手写两份后漂移
__version__ = version("rag-core")
