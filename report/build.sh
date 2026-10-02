#!/usr/bin/env bash
# Build the research dossier PDF with Tectonic (https://tectonic-typesetting.github.io).
# Usage: ./build.sh            (expects `tectonic` on PATH)
#        TECTONIC=/path/to/tectonic ./build.sh
set -e
cd "$(dirname "$0")"
T="${TECTONIC:-tectonic}"
"$T" -X compile main.tex > build.log 2>&1 || { grep -E "^error" -A3 build.log | head -30; exit 1; }
cp main.pdf CFFM-Net_Research_Dossier.pdf
grep -E "undefined|Undefined" build.log | sort -u || true
echo "Built CFFM-Net_Research_Dossier.pdf"
