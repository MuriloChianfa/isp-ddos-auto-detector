#!/usr/bin/env bash

set -euo pipefail
shopt -s extglob nullglob

: "${TARGET_ASN:?TARGET_ASN must be set and a valid 4-byte ASNumber}"
DEFAULT_DATASET_DIR="./datasets"
DATASET_DIR="${DATASET_DIR:-$DEFAULT_DATASET_DIR}"
OUTPUT_DIR="$DEFAULT_DATASET_DIR/ramfs"

mkdir -p "$OUTPUT_DIR"

if ! mountpoint -q "$OUTPUT_DIR"; then
  echo "mounting $OUTPUT_DIR"
  mount -t tmpfs -o size=5G tmpfs "$OUTPUT_DIR"
else
  echo "output dir already mounted"
fi

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
    nfdump -r "$f" -o csv "dst as ${TARGET_ASN}" > "$out"
  done
done

