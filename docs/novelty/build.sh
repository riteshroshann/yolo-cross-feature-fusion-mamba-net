#!/usr/bin/env bash
# Build the pilot novelty note with Tectonic (https://tectonic-typesetting.github.io).
# Usage: ./build.sh            (expects `tectonic` on PATH)
#        TECTONIC=/path/to/tectonic ./build.sh
set -e
cd "$(dirname "$0")"
T="${TECTONIC:-tectonic}"
python make_refs.py
"$T" -X compile novelty.tex > build.log 2>&1 || { grep -E "^error" -A3 build.log | head -30; exit 1; }
mv novelty.pdf CFFM-Net_Pilot_Novelty.pdf
echo "Built CFFM-Net_Pilot_Novelty.pdf"
