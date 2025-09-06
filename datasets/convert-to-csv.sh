#!/usr/bin/env bash

set -euo pipefail
shopt -s extglob nullglob

: "${FILTER:?FILTER must be a valid nfdump filter}"
DEFAULT_DATASET_DIR="./datasets"
DATASET_DIR="${DATASET_DIR:-$DEFAULT_DATASET_DIR}"
DEFAULT_OUTPUT_DIR="$DEFAULT_DATASET_DIR/tmp"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"

mkdir -p "$OUTPUT_DIR"

for dir in "$DATASET_DIR"/*; do
  [[ -d "$dir" && "$dir" != "$OUTPUT_DIR" ]] || continue
  echo "entering folder: $(basename "$dir")"

  for f in "$dir"/nfcapd.*; do
    base=$(basename "$f" .nfcapd)
    out="$OUTPUT_DIR/$base.csv"

    if [[ -f "$out" ]]; then
      echo "skipping, file already exists: $out"
      continue
    fi

    echo "converting: $f to $out"
    nfdump -r "$f" -o "csv:%ts,%td,%pr,%sa,%sp,%da,%dp,%pkt,%byt,%fl,%sas,%das,%sc,%dc,%flg" "${FILTER}" > "$out"
  done
done

