"""打印当前进程实际生效的配置（密钥已脱敏）。

用途：线上排查"这个实例到底连的哪个库"时，比对着 manifest 猜要快得多。
用法：uv run python scripts/show_config.py
"""

from __future__ import annotations

import json

from rag_core.settings import get_settings, load_env_files


def main() -> None:
    loaded = load_env_files()
    settings = get_settings()

    print(f"加载的配置文件: {[str(path) for path in loaded] or '无（只用进程环境变量）'}")
    print(json.dumps(settings.safe_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
