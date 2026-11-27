#!/usr/bin/env bash
# Clone TS_Predictability at the commit recorded in UPSTREAM.md.
# Re-running this script checks out that same commit again.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="${ROOT}/third_party/TS_Predictability"
URL="https://github.com/WanjinVon/TS_Predictability.git"
PIN="3bd624602c95506b6f9beedfc43c041fdb6afb36"

mkdir -p "${ROOT}/third_party"

if [[ -d "${DEST}" && ! -d "${DEST}/.git" ]]; then
  echo "error: ${DEST} exists but is not a git checkout. Remove it and re-run." >&2
  exit 1
fi

if [[ ! -d "${DEST}/.git" ]]; then
  git init "${DEST}"
  git -C "${DEST}" remote add origin "${URL}"
fi

git -C "${DEST}" fetch --depth 1 origin "${PIN}"
git -C "${DEST}" checkout --detach FETCH_HEAD

HEAD="$(git -C "${DEST}" rev-parse HEAD)"
if [[ "${HEAD}" != "${PIN}" ]]; then
  echo "error: checkout is ${HEAD}, expected ${PIN}" >&2
  exit 1
fi

echo "TS_Predictability ${HEAD}"
echo "tree: ${DEST}"
