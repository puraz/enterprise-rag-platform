"""离线回归门禁：与基线对比，指标退化则失败。

## 为什么需要门禁（而不只是"能跑通测试"）

单元测试保证"逻辑没写错"，但保证不了"效果没变差"：

- 改了分块策略 → 所有测试通过，但 Recall 从 100% 掉到 80%；
- 调了 prompt → 测试通过，但 Faithfulness 从 0.95 掉到 0.7；
- 换了 embedding 模型 → 测试通过，但索引语义漂移。

这些只有**跑同一批评估用例并对比基线**才能发现。

## 阈值怎么定

不同指标有不同容差：

| 指标 | 容差 | 理由 |
|---|---|---|
| Recall@k | -2% | 检索是下限，掉一点都严重 |
| MRR | -0.03 | 排序微调会带来小幅波动 |
| Faithfulness | -0.05 | LLM 评判本身有波动（同输入不同次可能差 0.03） |

**容差不能设 0**：LLM 评判、并发时序都会带来抖动，
设 0 会让门禁天天误报，最终被绕过。

## 用法

    # 用最新报告与基线对比
    uv run python scripts/check_regression.py

    # 指定基线
    uv run python scripts/check_regression.py --baseline baseline-30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_REPORTS = REPO_ROOT / "tests" / "eval" / "reports"

# 每个指标的"可接受退化"上限。
#
# **负数表示允许下降的量**。超过就失败。
# 这些数字不是拍的，见模块 docstring 的说明。
TOLERANCES: dict[str, float] = {
    "recall": -0.02,
    "mrr": -0.03,
    "hit_rate": -0.02,
    "faithfulness": -0.05,
    # 误报率是"越低越好"，所以容差是**正数**（允许上升）
    "false_positive_rate": 0.05,
}


def _load_latest(
    pattern: str, *, exclude: str | None = None, require_metrics: bool = True
) -> dict[str, object] | None:
    """取最新的一份**同类型**报告。

    ## 必须校验报告类型（实测踩到的坑）

    第一版只按"有没有 label 字段"筛选，结果把 Ragas 的
    `ragas-*.json` 也当成检索报告读进来——它没有 `metrics` 字段，
    于是所有指标都被"找不到就跳过"，门禁**假通过**。

    假通过比失败更危险：它会让人以为"回归检查做了"，
    而实际上什么都没检查。

    所以：
    - 检索报告必须有 `metrics` 字典且含 `recall`；
    - 不满足的直接跳过并**打印原因**（不静默）。
    """
    candidates = []
    skipped: list[str] = []
    for path in sorted(EVAL_REPORTS.glob(pattern)):
        if exclude and exclude in path.name:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "label" not in data:
            continue
        metrics = data.get("metrics")
        if require_metrics and (not isinstance(metrics, dict) or "recall" not in metrics):
            skipped.append(path.name)
            continue
        candidates.append((path.stat().st_mtime, data))

    if skipped:
        print(f"（跳过 {len(skipped)} 份非检索报告：{', '.join(skipped[:3])}）")
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _find_baseline(label: str) -> dict[str, object] | None:
    """按 label 找基线报告。"""
    for path in sorted(EVAL_REPORTS.glob("*baseline*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("label") != label:
            continue
        metrics = data.get("metrics")
        if not isinstance(metrics, dict) or "recall" not in metrics:
            continue
        return data
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线回归门禁")
    parser.add_argument("--baseline", default="baseline-30", help="基线报告标签")
    args = parser.parse_args(argv)

    baseline = _find_baseline(args.baseline)
    if baseline is None:
        print(f"⚠ 找不到基线报告 {args.baseline!r}，跳过回归检查")
        print("  （首次运行时正常：先跑 run_eval.py --label baseline-30 建立基线）")
        return 0

    latest = _load_latest("*.json", exclude="baseline")
    if latest is None:
        print("⚠ 找不到当前报告，跳过回归检查")
        return 0

    baseline_metrics = baseline.get("metrics", {})
    latest_metrics = latest.get("metrics", {})
    assert isinstance(baseline_metrics, dict)
    assert isinstance(latest_metrics, dict)

    print(f"基线: {baseline.get('label')}  当前: {latest.get('label')}")
    print(f"{'指标':<22} {'基线':>10} {'当前':>10} {'变化':>10}  判定")
    print("-" * 70)

    failures: list[str] = []
    compared = 0
    for key, tolerance in TOLERANCES.items():
        old = baseline_metrics.get(key)
        new = latest_metrics.get(key)
        if old is None or new is None:
            continue

        old_value, new_value = float(old), float(new)  # type: ignore[arg-type]
        delta = new_value - old_value
        compared += 1

        # 误报率：上升是变差；其它：下降是变差。
        #
        # 这里刻意用 if/else 而不是三元表达式：两种指标的**比较方向
        # 是相反的**，写成三元会把"方向翻转"这个关键信息压进一行，
        # 读代码的人容易看反（而看反的后果是门禁失效）。
        if key == "false_positive_rate":  # noqa: SIM108 - 见上方说明
            regression = delta > tolerance
        else:
            regression = delta < tolerance

        mark = "✗ 退化" if regression else "✓"
        if key in {"recall", "hit_rate", "false_positive_rate"}:
            print(f"{key:<22} {old_value:>9.1%} {new_value:>9.1%} {delta:>+9.1%}  {mark}")
        else:
            print(f"{key:<22} {old_value:>10.3f} {new_value:>10.3f} {delta:>+10.3f}  {mark}")

        if regression:
            failures.append(
                f"{key} 从 {old_value:.3f} 退化到 {new_value:.3f}（容差 {tolerance:+.3f}）"
            )

    print()
    if not compared:
        print("✗ 回归检查无法执行：没有任何指标可比对")
        print("  可能原因：当前报告不是检索报告，或基线缺 metrics 字段。")
        print("  **这种情况不能算通过**——它意味着回归检查实际没做。")
        return 1

    if failures:
        print("✗ 回归检查失败：")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("✓ 回归检查通过：所有指标在容差内")
    return 0


if __name__ == "__main__":
    sys.exit(main())
