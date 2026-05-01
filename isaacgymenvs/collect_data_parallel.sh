#!/usr/bin/env bash
set -euo pipefail

ROOT="${SCFIELDS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export SCFIELDS_ROOT="$ROOT"

echo "Delegating to scripts/collect_contact_field_data_parallel.sh" >&2
exec "$ROOT/scripts/collect_contact_field_data_parallel.sh" "$@"
