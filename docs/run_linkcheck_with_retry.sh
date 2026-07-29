#!/usr/bin/env bash
# Check external documentation links with bounded retries for remote outages.

set -euo pipefail

attempts=3
delay_seconds=30

for attempt in $(seq 1 "${attempts}"); do
    if "$(dirname "${BASH_SOURCE[0]}")/build.sh" linkcheck; then
        exit 0
    fi

    if [[ "${attempt}" -eq "${attempts}" ]]; then
        echo "External linkcheck failed after ${attempts} attempts." >&2
        echo "Internal references are enforced separately by the strict HTML job." >&2
        exit 1
    fi

    echo "External linkcheck failed; retrying in ${delay_seconds}s (attempt ${attempt}/${attempts})." >&2
    sleep "${delay_seconds}"
done
