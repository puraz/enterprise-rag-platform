"""仓储层：所有 SQL 的唯一落点。

约定：
- **service 不写 SQL**，只调用 repository 方法；
- **路由不写业务逻辑**，只做参数校验与响应组装；
- repository 接收 `AsyncSession`，不自己建连接、不开事务
  （事务边界由 service 或依赖注入决定）。

A3 起按聚合逐个加文件：TenantRepository / DocumentRepository / …
"""
