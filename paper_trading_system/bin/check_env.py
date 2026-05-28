from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def check_import(name: str, import_name: str | None = None) -> bool:
    module_name = import_name or name
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        location = getattr(module, "__file__", "built-in")
        print(f"OK   {name}: version={version} location={location}")
        return True
    except Exception as exc:
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    qlib_src = os.environ.get("QLIB_SRC", "")
    if qlib_src:
        sys.path.insert(0, qlib_src)

    print(f"python: {sys.executable}")
    print(f"QLIB_SRC: {qlib_src or '<pip package>'}")
    print(f"QLIB_PROVIDER_URI: {os.environ.get('QLIB_PROVIDER_URI', '<unset>')}")
    print()

    checks = [
        ("pandas", None),
        ("numpy", None),
        ("qlib", None),
        ("torch", None),
        ("xgboost", None),
        ("lightgbm", None),
        ("ruamel.yaml", None),
        ("yaml", None),
    ]
    ok = True
    for name, import_name in checks:
        ok = check_import(name, import_name) and ok

    provider = os.environ.get("QLIB_PROVIDER_URI")
    if provider:
        path = Path(provider).expanduser()
        if path.exists():
            print(f"OK   qlib data path exists: {path}")
            calendar = path / "calendars" / "day.txt"
            if calendar.exists():
                lines = [line.strip() for line in calendar.read_text(encoding="utf-8").splitlines() if line.strip()]
                if lines:
                    print(f"OK   qlib calendar range: {lines[0]} -> {lines[-1]} ({len(lines)} days)")
        else:
            print(f"FAIL qlib data path missing: {path}")
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
