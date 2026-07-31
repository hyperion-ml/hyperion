#!/usr/bin/env bash
# Build Hyperion's Sphinx documentation from any working directory.

set -euo pipefail

docs_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${docs_dir}/_build"
target="${1:-html}"

case "${target}" in
    clean)
        rm -rf "${build_dir}"
        exit 0
        ;;
    html|linkcheck|doctest|spelling)
        ;;
    *)
        echo "Usage: $0 [html|linkcheck|doctest|spelling|clean]" >&2
        exit 2
        ;;
esac

# Link checking necessarily requires network access; enable intersphinx only
# for that target. HTML and doctest builds remain reliable offline by default.
if [[ "${target}" == "linkcheck" ]]; then
    export HYPERION_DOCS_ONLINE=1
fi

if [[ -n "${HYPERION_PYTHON:-}" ]]; then
    python_bin="${HYPERION_PYTHON}"
elif command -v python >/dev/null 2>&1; then
    python_bin="python"
elif command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
else
    echo "Python 3 is required to build the documentation." >&2
    exit 1
fi

if ! "${python_bin}" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
    echo "Hyperion documentation requires Python 3.10 or newer." >&2
    echo "Set HYPERION_PYTHON to a compatible interpreter if necessary." >&2
    exit 1
fi

if [[ "${target}" == "spelling" ]]; then
    if codespell_bin="$(command -v codespell)"; then
        :
    else
        codespell_bin="$(dirname "${python_bin}")/codespell"
    fi
    if [[ ! -x "${codespell_bin}" ]]; then
        echo "codespell is not installed. Install documentation dependencies with:" >&2
        echo "  ${python_bin} -m pip install -r ${docs_dir}/requirements.txt" >&2
        exit 1
    fi

    rst_files=()
    while IFS= read -r -d '' rst_file; do
        rst_files+=("${rst_file}")
    done < <(
        find "${docs_dir}" -type f -name "*.rst" \
            ! -path "${docs_dir}/generated/*" -print0
    )
    "${codespell_bin}" -I "${docs_dir}/spelling_wordlist.txt" "${rst_files[@]}"
    exit $?
fi

if ! "${python_bin}" -c "import sphinx" >/dev/null 2>&1; then
    echo "Sphinx is not installed. Install documentation dependencies with:" >&2
    echo "  ${python_bin} -m pip install -r ${docs_dir}/requirements.txt" >&2
    exit 1
fi

mkdir -p "${build_dir}/.matplotlib"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${build_dir}/.matplotlib}"

"${python_bin}" -m sphinx -M "${target}" "${docs_dir}" "${build_dir}" \
    -W --keep-going
