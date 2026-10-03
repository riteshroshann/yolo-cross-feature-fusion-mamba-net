#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
T="${TECTONIC:-tectonic}"
python make_refs.py
"$T" -X compile main.tex > build.log 2>&1 || { grep -E "^error" -A3 build.log | head -30; exit 1; }
mv main.pdf CFFM-Net_Architecture_Atlas.pdf
grep -E "undefined|Undefined" build.log | sort -u || true
echo "Built CFFM-Net_Architecture_Atlas.pdf"
