"""A15 收口测试：压测器、对账、回归门禁。

## 为什么这些也要测

- **压测器**：如果分位数算错，容量规划就基于假数据；
- **对账**：如果检查逻辑有 bug（比如自己把证据过滤掉），
  它会永远返回"通过"——这是最危险的假阳性；
- **回归门禁**：如果它把非检索报告当成检索报告，
  会"假通过"（实测踩到过）。

三者都是"看起来在工作，实际没工作"的高危组件。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT / "scripts"))

from loadtest import LatencyStats, LoadTestReport, StageResult, format_report  # noqa: E402

# ==================================================== 压测器：分位数


def test_percentile_uses_nearest_rank() -> None:
    """分位数用"最近秩"，不是插值。

    插值会造出实际不存在的延迟值（如 12.5ms 而样本里只有 10 和 15），
    而容量规划要看"真实存在的最差 N%"。
    """
    stats = LatencyStats(samples=[10.0, 20.0, 30.0, 40.0, 50.0])

    assert stats.percentile(50) == 30.0
    assert stats.percentile(0) == 10.0
    assert stats.percentile(100) == 50.0
    # P90 应落在样本值上
    assert stats.percentile(90) in {40.0, 50.0}


def test_percentile_handles_empty() -> None:
    """空样本返回 0（不能让压测因无数据崩溃）。"""
    stats = LatencyStats(samples=[])
    assert stats.percentile(95) == 0.0
    assert stats.mean == 0.0
    assert stats.max == 0.0


def test_percentile_single_sample() -> None:
    stats = LatencyStats(samples=[42.0])
    assert stats.percentile(50) == 42.0
    assert stats.percentile(99) == 42.0


# ==================================================== 压测器：饱和点判定


def _stage(concurrency: int, qps: float, p99: float, *, success: int, total: int) -> StageResult:
    stats = LatencyStats(samples=[p99])
    return StageResult(
        concurrency=concurrency,
        duration_s=1.0,
        total_requests=total,
        success=success,
        latency=stats,
    )


def test_saturation_detected_when_qps_plateaus() -> None:
    """QPS 不再增长时判为饱和。

    这是容量规划的核心：加并发不再换来吞吐，说明瓶颈不在并发数。
    """
    report = LoadTestReport(
        stages=[
            _stage(5, qps=100, p99=20, success=100, total=100),
            _stage(10, qps=200, p99=30, success=100, total=200),
            # QPS 只增长 5%（< 20% 阈值）→ 饱和
            _stage(20, qps=210, p99=200, success=210, total=210),
        ]
    )
    saturation = report.saturation_stage
    assert saturation is not None
    assert saturation.concurrency == 20


def test_saturation_detected_on_error_rate() -> None:
    """错误率超过 1% 时判为饱和（即使 QPS 还在涨）。"""
    report = LoadTestReport(
        stages=[
            _stage(5, qps=100, p99=20, success=100, total=100),
            # 90/100 成功 = 90%
            StageResult(
                concurrency=10,
                duration_s=1.0,
                total_requests=100,
                success=90,
                errors={"PoolTimeout": 10},
                latency=LatencyStats(samples=[50.0]),
            ),
        ]
    )
    saturation = report.saturation_stage
    assert saturation is not None
    assert saturation.concurrency == 10
    assert saturation.success_rate == 0.9


def test_no_saturation_when_all_healthy() -> None:
    """所有档位 QPS 增长且无错误 → 未饱和（不该硬报一个饱和点）。"""
    report = LoadTestReport(
        stages=[
            _stage(5, qps=100, p99=20, success=100, total=100),
            _stage(10, qps=200, p99=30, success=200, total=200),
            _stage(20, qps=400, p99=45, success=400, total=400),
        ]
    )
    assert report.saturation_stage is None


def test_format_report_produces_markdown_table() -> None:
    report = LoadTestReport(
        stages=[_stage(5, qps=100, p99=20, success=100, total=100)],
        target="test",
    )
    markdown = format_report(report)
    assert "| 并发 | QPS |" in markdown
    assert "| 5 |" in markdown


# ==================================================== 对账：检查逻辑本身要正确


def test_cross_tenant_check_filters_by_parent_not_child() -> None:
    """**实测发现的漏洞**：跨租户检查必须按父表归属过滤。

    错误写法：

        WHERE c.tenant_id <> d.tenant_id AND c.tenant_id = :tid

    如果把 chunk 的 tenant_id 改成别的租户，它正好不满足
    `c.tenant_id = :tid`，被 WHERE 过滤掉——检查通过，漏洞漏报。

    正确写法按父表过滤（`d.tenant_id = :tid`），父表归属是业务事实。
    """
    # 静态验证：源码里必须用 d.tenant_id 过滤，不能用 c.tenant_id
    source = (REPO_ROOT / "scripts" / "audit_integrity.py").read_text(encoding="utf-8")
    start = source.index("async def _check_cross_tenant_leak")
    end = source.index("async def _check_document_count_consistency")
    section = source[start:end]

    assert "d.tenant_id = CAST(:tid AS uuid)" in section, (
        "跨租户检查必须按父表（documents）归属过滤"
    )
    # 不该出现"用子表 tenant_id 做过滤"的写法
    assert "AND c.tenant_id = CAST(:tid AS uuid)" not in section


def test_chunk_count_check_uses_chunk_side_aggregation() -> None:
    """**实测发现的盲区**：chunk_count 检查必须以"实际有 chunk 的文档"为全集。

    原版查 `status='ready'` 或 `chunk_count > 0`，都会漏掉
    "有 chunk 但记录为 0"的情况——而那正是最该抓的状态机 bug。
    """
    source = (REPO_ROOT / "scripts" / "audit_integrity.py").read_text(encoding="utf-8")
    start = source.index("async def _check_document_count_consistency")
    end = source.index("async def _check_orphan_embeddings")
    section = source[start:end]

    # 必须从 chunks 侧聚合
    assert "FROM chunks" in section
    assert "GROUP BY document_id" in section
    # 不该只查 status='ready'
    assert "status = 'ready'" not in section


def test_all_audit_checks_documented() -> None:
    """每个检查都要有 docstring 说明"这条抓什么"。

    对账的价值在于"知道自己在检查什么"。没有 docstring 的检查
    半年后没人敢动（不知道删了会漏什么）。
    """
    source = (REPO_ROOT / "scripts" / "audit_integrity.py").read_text(encoding="utf-8")
    for name in (
        "_check_chunk_embedding_parity",
        "_check_duplicate_chunk_index",
        "_check_cross_tenant_leak",
        "_check_document_count_consistency",
        "_check_orphan_embeddings",
        "_check_cost_reconciliation",
    ):
        index = source.index(f"async def {name}")
        # 函数定义后紧跟 docstring
        section = source[index : index + 2000]
        assert '"""' in section[:300], f"{name} 缺 docstring"


# ==================================================== 回归门禁


def test_regression_gate_skips_non_retrieval_reports() -> None:
    """门禁必须跳过非检索报告（否则"找不到指标"会被当成通过）。

    实测踩到：Ragas 报告没有 `metrics` 字段，被当成检索报告读入后，
    所有指标都"缺失→跳过"，门禁**假通过**。假通过比失败更危险。
    """
    source = (REPO_ROOT / "scripts" / "check_regression.py").read_text(encoding="utf-8")
    assert "require_metrics" in source
    assert '"recall" not in metrics' in source or "'recall' not in metrics" in source


def test_regression_gate_fails_when_nothing_compared() -> None:
    """没有任何指标可比对时必须失败（不能算通过）。"""
    source = (REPO_ROOT / "scripts" / "check_regression.py").read_text(encoding="utf-8")
    assert "if not compared:" in source
    assert "回归检查无法执行" in source


def test_regression_gate_exit_codes() -> None:
    """门禁的真实退出码：退化=1，正常=0。

    用子进程跑（不是 import 后调函数）：CI 里看的是退出码，
    必须验证真实进程行为。
    """
    script = REPO_ROOT / "scripts" / "check_regression.py"
    # S603：命令由 sys.executable + 仓库内固定路径拼成，不含外部输入。
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    # 当前仓库的指标在容差内 → 退出码 0
    assert result.returncode == 0, f"门禁意外失败：{result.stdout}{result.stderr}"


def test_tolerances_are_nonzero() -> None:
    """容差不能是 0：LLM 评判与并发时序会抖动，0 容差会让门禁被绕过。"""
    source = (REPO_ROOT / "scripts" / "check_regression.py").read_text(encoding="utf-8")
    assert '"recall": -0.02' in source
    assert '"faithfulness": -0.05' in source


# ==================================================== CI 配置


def test_ci_workflow_is_valid() -> None:
    """CI 配置必须可解析且覆盖四层门禁。"""
    yaml = pytest.importorskip("yaml")
    path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    jobs = data["jobs"]
    assert {"static", "unit", "integration", "regression"} <= set(jobs)

    # static 必须查契约漂移（前端类型漂移的唯一自动防线）
    static_steps = " ".join(
        str(step.get("run", "")) + str(step.get("name", "")) for step in jobs["static"]["steps"]
    )
    assert "check_contracts_fresh" in static_steps

    # integration 必须用带 pgvector 的镜像
    assert "pgvector" in jobs["integration"]["services"]["postgres"]["image"]


# ==================================================== 评估报告的对比环节


def test_compare_with_previous_skips_non_retrieval_reports(tmp_path: Path) -> None:
    """`compare_with_previous` 必须跳过非检索报告，且不崩。

    ## 实测踩到的坑

    `tests/eval/reports/` 里混着三种文件：

    | 文件 | 形状 |
    |---|---|
    | `{timestamp}-{label}.json` | 检索报告（有 `label`/`k`/`metrics.recall`） |
    | `ragas-*.json` | Ragas 报告（有 `label`/`faithfulness`，**没有** `k`/`metrics`） |
    | `ragas-samples-*.json` | 样本导出，是 **list** |

    初版只判断 `data.get("label") != ... and data.get("k") == ...`，
    遇到 list 直接 `AttributeError: 'list' object has no attribute 'get'`。
    表现为：跑完 `run_eval.py --setup` 在**最后的对比环节**崩掉，
    看起来像 setup 失败（实际语料已建好）。

    这里用真实的临时目录 + 三种真实形状做出验证，而不是断言源码字符串
    ——源码断言挡不住"逻辑还在但判据写错"。
    """
    import json
    import sys

    sys.path.insert(0, str(REPO_ROOT / "tests" / "eval"))
    from run_eval import Report, compare_with_previous

    # 三种真实形状的报告
    (tmp_path / "20260101T000000Z-old.json").write_text(
        json.dumps(
            {
                "label": "old",
                "k": 10,
                "metrics": {
                    "recall": 0.9,
                    "mrr": 0.8,
                    "hit_rate": 0.9,
                    "false_positive_rate": 0.1,
                    "avg_latency_ms": 100.0,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "ragas-trial.json").write_text(
        json.dumps({"label": "ragas", "faithfulness": 0.7}), encoding="utf-8"
    )
    (tmp_path / "ragas-samples.json").write_text(
        json.dumps([{"q": "x"}, {"q": "y"}]), encoding="utf-8"
    )

    # 把模块的 REPORTS_DIR 临时指过去（不污染真实报告目录）
    import run_eval

    original = run_eval.REPORTS_DIR
    run_eval.REPORTS_DIR = tmp_path
    try:
        report = Report(label="new", k=10, results=[], elapsed_s=0.1)
        # **不抛异常**就是主要断言：遇到 list 或 Ragas 报告必须安静跳过
        compare_with_previous(report)
    finally:
        run_eval.REPORTS_DIR = original


def test_compare_with_previous_handles_corrupt_json(tmp_path: Path) -> None:
    """单个损坏的报告文件不能让整个评估流程崩掉。

    报告目录是运行产物累积地，出现半截文件（进程被 kill）是常态。
    "跳过它"比"整个 eval 失败"合理——它只是少一个对比基准。
    """
    import json
    import sys

    sys.path.insert(0, str(REPO_ROOT / "tests" / "eval"))
    import run_eval
    from run_eval import Report, compare_with_previous

    (tmp_path / "broken.json").write_text("{not valid json", encoding="utf-8")

    # **必须放两份**：`compare_with_previous` 在 `len(reports) < 2` 时
    # 直接返回，只有一份的话根本走不到解析循环——测试会"通过"而
    # 实际什么都没验证（这正是初版测试的漏洞，破坏性验证时发现的）。
    # 这份必须是**完整**的检索报告：对比环节会读 metrics 的全部五个键，
    # 只给 recall 会 KeyError（测试数据不完整，不是被测代码的问题）。
    (tmp_path / "20260101T000000Z-old.json").write_text(
        json.dumps(
            {
                "label": "old",
                "k": 10,
                "metrics": {
                    "recall": 0.9,
                    "mrr": 0.8,
                    "hit_rate": 0.9,
                    "false_positive_rate": 0.1,
                    "avg_latency_ms": 100.0,
                },
            }
        ),
        encoding="utf-8",
    )

    original = run_eval.REPORTS_DIR
    run_eval.REPORTS_DIR = tmp_path
    try:
        compare_with_previous(Report(label="x", k=10, results=[], elapsed_s=0.1))
    finally:
        run_eval.REPORTS_DIR = original
