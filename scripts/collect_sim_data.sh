#!/usr/bin/env bash
set -euo pipefail

ROOT="${SCFIELDS_ROOT:-$(pwd)}"
export SCFIELDS_ROOT="$ROOT"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"

bash "$ROOT/scripts/collect_contact_field_data.sh" "${1:-scraper_8}"
