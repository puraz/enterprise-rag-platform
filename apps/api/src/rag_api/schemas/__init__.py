"""请求/响应实体的统一出口。

## 这个目录的职责

1. **汇总每个域用到的请求/响应模型**：形状的唯一真相仍在
   `packages/contracts`（四端共享），这里只做面向 API 层的 re-export，
   让路由不必到处 import 具体契约模块，也便于一眼看清某个端点用了哪些实体。
2. **承担 domain → response 的映射**：ORM 对象、分页结果、入库结果等
   不能直接当响应返回；映射集中在这里，service 不感知 HTTP 契约。

## 为什么不在 service 里做映射

service 是业务层，应该对"响应长什么样"无感。把映射放在 service 里会让
worker / 定时任务等她方复用时被迫依赖 API 契约；放进控制器又会让路由
膨胀并且容易复制出第二份映射。`schemas` 正好是这两者之间的边界。

## 为什么可以 re-export

`packages/contracts` 已经是跨端契约的单一真相。这里**不重复定义字段**，
只导入并导出——字段增删仍然只需改 contracts 并重跑 `make gen-contracts`。
"""
