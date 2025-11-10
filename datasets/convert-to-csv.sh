#!/usr/bin/env bash

set -euo pipefail
shopt -s extglob nullglob

: "${FILTER:?FILTER must be a valid nfdump filter}"
DEFAULT_DATASET_DIR="./datasets"
DATASET_DIR="${DATASET_DIR:-$DEFAULT_DATASET_DIR}"
DEFAULT_OUTPUT_DIR="$DEFAULT_DATASET_DIR/tmp"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
PARALLEL_JOBS="${PARALLEL_JOBS:-8}"
DIR_PATTERN="${DIR_PATTERN:-*}"

mkdir -p "$OUTPUT_DIR/raw"

# Function to convert a single file
convert_file() {
    local f="$1"
    local base=$(basename "$f" .nfcapd)
    local out="$OUTPUT_DIR/raw/$base.csv"

    if [[ -f "$out" ]]; then
        echo "skipping, file already exists: $out"
        return 0
    fi

    echo "converting: $f to $out"
    nfdump -r "$f" -o "csv:%trg,%td,%pr,%sa,%sp,%da,%dp,%pkt,%byt,%fl,%sas,%das,%sc,%dc,%flg" "${FILTER}" > "$out"
}

export -f convert_file
export OUTPUT_DIR FILTER

for dir in "$DATASET_DIR"/$DIR_PATTERN; do
  [[ -d "$dir" && "$dir" != "$OUTPUT_DIR" ]] || continue
  echo "entering folder: $(basename "$dir")"

  files=()
  for f in "$dir"/nfcapd.*; do
    files+=("$f")
  done

  if [[ ${#files[@]} -eq 0 ]]; then
    echo "no nfcapd files found in $(basename "$dir")"
    continue
  fi

  echo "found ${#files[@]} files to process in parallel (max $PARALLEL_JOBS jobs)"
  printf '%s\n' "${files[@]}" | xargs -n 1 -P "$PARALLEL_JOBS" -I {} bash -c 'convert_file "$@"' _ {}

  echo "completed processing folder: $(basename "$dir")"
done

