#!/usr/bin/env bash
set -Eeuo pipefail

echo "======================================"
echo "Running pre-PR checks"
echo "======================================"

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT_DIR"

TARGET="src"

fail=0

run() {
name="$1"
shift

```
echo
echo "---- $name ----"
if "$@"; then
    echo "✓ $name passed"
else
    echo "✗ $name failed"
    fail=1
fi
```

}

# ---------- Auto-fix phase ----------

echo
echo "Applying automatic formatting..."

black "$TARGET"
ruff check --fix "$TARGET"

# ---------- Verification phase ----------

run "Ruff (lint)" ruff check "$TARGET"
run "Black (format check)" black --check "$TARGET"
run "Pyrefly (types)" pyrefly "$TARGET"

# ---------- Result ----------

echo
echo "======================================"

if [[ $fail -eq 0 ]]; then
echo "All checks passed ✅"
exit 0
else
echo "Some checks failed ❌"
exit 1
fi
