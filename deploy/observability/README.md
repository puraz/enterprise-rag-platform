# 可观测性部署（A14）

## 内容

| 文件 | 用途 |
|---|---|
| `prometheus/prometheus.yml` | 抓取配置（15s 间隔，只抓 API） |
| `prometheus/alerts.yml` | 10 条告警规则，分 4 组 |
| `grafana/dashboard.json` | 22 个面板，六层结构，可直接导入 |

## 快速验证（无需 Docker）

本机没有 Docker，但可以验证指标端点本身：

```bash
# 1. 起 API
uv run --package rag-api uvicorn rag_api.main:app --port 8000

# 2. 打几个请求
curl -s localhost:8000/health/live > /dev/null
curl -s localhost:8000/api/v1/nope > /dev/null

# 3. 看指标
curl -s localhost:8000/health/metrics | grep rag_http_requests_total
```

预期看到：

```
rag_http_requests_total{method="GET",route="/health/live",status="200"} 1.0
rag_http_requests_total{method="GET",route="__unmatched__",status="404"} 1.0
```

`__unmatched__` 是 404 的统一 label——**不用实际路径**，因为扫描器
（`/.env`、`/wp-admin`）会让时间序列数量爆炸。

## 用 Docker 起完整栈

```bash
docker run -d --name prometheus -p 9090:9090 \
  -v "$PWD/deploy/observability/prometheus:/etc/prometheus:ro" \
  prom/prometheus:latest

# Grafana 导入 dashboard.json，数据源指向 http://localhost:9090
docker run -d --name grafana -p 3000:3000 grafana/grafana:latest
```

## 设计要点

### label 基数必须受控

**绝不把 document_id / user_id / query 放进 label**。十万文档就是十万条
时间序列，Prometheus 内存会被打爆。需要按 ID 排查时用日志/trace。

路由 label 用**路由模板**（`/api/v1/documents/{doc_id}`）而不是实际路径。
404 归入 `__unmatched__`。

### 为什么 worker 指标暂不抓取

Worker 没有 HTTP 端点。两种方案各有代价：

- **多进程模式**（`PROMETHEUS_MULTIPROC_DIR`）：需要 sidecar 暴露共享目录；
- **Pushgateway**：适合批处理，但"进程消失后指标残留"会让
  `tasks_inflight` 之类的 Gauge 永久显示假值。

留到 A15 压测时按实际部署形态决定。

### 告警阈值的依据

每条规则的注释里都写了依据（例如"P95 > 2s：非 LLM 接口应在 500ms 内"）。
没有依据的阈值会在正常波动时反复触发，最终被静音——**比不设告警更糟**。
