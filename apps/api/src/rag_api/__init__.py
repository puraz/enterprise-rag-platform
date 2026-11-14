"""rag_api —— 唯一对外入口。

分层：api(路由) / schemas / services / repositories / pipelines / retrievers /
parsers / chunkers / storages / providers / identity / evaluators / workers / models / deps
规则：路由不写业务，service 不写 SQL，SQL 只在 repository。
"""

from importlib.metadata import version

__version__ = version("rag-api")
