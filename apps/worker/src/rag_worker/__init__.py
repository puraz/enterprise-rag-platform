"""rag_worker —— 异步管线与定时任务进程。

端边界：禁止 import rag_api（依赖声明 + tests/test_workspace.py 双层守卫）。
与 api 的唯一通信方式：API 写队列消息 / DB 状态，worker 消费并回写。
"""

from importlib.metadata import version

__version__ = version("rag-worker")
