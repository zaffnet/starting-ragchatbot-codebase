#\!/usr/bin/env bash
set -euo pipefail

# Run black code formatter on the Python codebase.
# Usage:
#   ./scripts/format.sh          # Format all files in place
#   ./scripts/format.sh --check  # Check only (exit 1 if changes needed)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TARGETS=("$ROOT_DIR/backend" "$ROOT_DIR/main.py")

if [[ "${1:-}" == "--check" ]]; then
    echo "Checking formatting with black..."
    uv run black --check --diff "${TARGETS[@]}"
    echo "All files formatted correctly."
else
    echo "Formatting with black..."
    uv run black "${TARGETS[@]}"
    echo "Done."
fi
