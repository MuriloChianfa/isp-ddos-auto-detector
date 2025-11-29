#!/usr/bin/env bash

set -euo pipefail
shopt -s nullglob

# Detect if we're already in the datasets directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(basename "$SCRIPT_DIR")" == "datasets" ]]; then
  DEFAULT_DATASET_DIR="$SCRIPT_DIR"
else
  DEFAULT_DATASET_DIR="$SCRIPT_DIR/datasets"
fi

DATASET_DIR="${DATASET_DIR:-$DEFAULT_DATASET_DIR}"
DIR_PATTERN="${DIR_PATTERN:-*}"

echo "Calculating dataset sizes (CSV format, uncompressed)..."
echo "========================================================"
echo

for dir in "$DATASET_DIR"/$DIR_PATTERN; do
  [[ -d "$dir" && -d "$dir/raw" ]] || continue
  
  dataset_name=$(basename "$dir")
  
  # Count files
  csv_files=("$dir/raw"/*.csv)
  file_count=${#csv_files[@]}
  
  if [[ $file_count -eq 0 ]]; then
    echo "[$dataset_name] No CSV files found"
    echo
    continue
  fi
  
  echo "[$dataset_name]"
  
  # Calculate total size
  cd "$dir/raw"
  total_size_bytes=$(ls *.csv 2>/dev/null | xargs du -cb 2>/dev/null | awk 'END {print $1}')
  
  if [[ -n "$total_size_bytes" && "$total_size_bytes" -gt 0 ]]; then
    # Convert to human-readable format
    total_size_human=$(numfmt --to=iec-i --suffix=B "$total_size_bytes" 2>/dev/null || echo "$total_size_bytes bytes")
    
    # Also show in GB
    total_size_gb=$(awk -v bytes="$total_size_bytes" 'BEGIN {printf "%.2f GB", bytes/1024/1024/1024}')
    
    echo "  Total size: $total_size_human ($total_size_gb)"
  else
    echo "  Total size: 0"
  fi
  
  # Calculate sizes by train/validation/test splits
  echo ""
  echo "  Split breakdown:"
  
  # Define patterns based on dataset (from config.py)
  case "$dataset_name" in
    "itp-downstream-http-flood")
      train_pattern="nfcapd.20250714*.csv"
      validation_pattern="nfcapd.20250715*.csv"
      test_pattern="nfcapd.2025071[67]*.csv"
      ;;
    "itp-synack-customer-outage")
      train_pattern="nfcapd.2025100[567]*.csv"
      validation_pattern="nfcapd.2025101[01]*.csv"
      test_pattern="nfcapd.2025101[123]*.csv"
      ;;
    "itp-multivector-udp-100gbps-peak")
      train_pattern="nfcapd.20251008*.csv nfcapd.20251009*.csv nfcapd.20251010*.csv"
      validation_pattern="nfcapd.20251011*.csv"
      test_pattern="nfcapd.2025101[2-6]*.csv"
      ;;
    *)
      train_pattern=""
      validation_pattern=""
      test_pattern=""
      ;;
  esac
  
  if [[ -n "$train_pattern" ]]; then
    # Calculate train size
    train_size_bytes=$(ls $train_pattern 2>/dev/null | xargs du -cb 2>/dev/null | awk 'END {print $1}')
    if [[ -n "$train_size_bytes" && "$train_size_bytes" -gt 0 ]]; then
      train_size_human=$(numfmt --to=iec-i --suffix=B "$train_size_bytes" 2>/dev/null || echo "$train_size_bytes bytes")
      train_size_gb=$(awk -v bytes="$train_size_bytes" 'BEGIN {printf "%.2f GB", bytes/1024/1024/1024}')
      echo "    Train:      $train_size_human ($train_size_gb)"
    fi
    
    # Calculate validation size
    validation_size_bytes=$(ls $validation_pattern 2>/dev/null | xargs du -cb 2>/dev/null | awk 'END {print $1}')
    if [[ -n "$validation_size_bytes" && "$validation_size_bytes" -gt 0 ]]; then
      validation_size_human=$(numfmt --to=iec-i --suffix=B "$validation_size_bytes" 2>/dev/null || echo "$validation_size_bytes bytes")
      validation_size_gb=$(awk -v bytes="$validation_size_bytes" 'BEGIN {printf "%.2f GB", bytes/1024/1024/1024}')
      echo "    Validation: $validation_size_human ($validation_size_gb)"
    fi
    
    # Calculate test size
    test_size_bytes=$(ls $test_pattern 2>/dev/null | xargs du -cb 2>/dev/null | awk 'END {print $1}')
    if [[ -n "$test_size_bytes" && "$test_size_bytes" -gt 0 ]]; then
      test_size_human=$(numfmt --to=iec-i --suffix=B "$test_size_bytes" 2>/dev/null || echo "$test_size_bytes bytes")
      test_size_gb=$(awk -v bytes="$test_size_bytes" 'BEGIN {printf "%.2f GB", bytes/1024/1024/1024}')
      echo "    Test:       $test_size_human ($test_size_gb)"
    fi
    
    # Calculate total of splits
    train_size_bytes=${train_size_bytes:-0}
    validation_size_bytes=${validation_size_bytes:-0}
    test_size_bytes=${test_size_bytes:-0}
    split_total_bytes=$((train_size_bytes + validation_size_bytes + test_size_bytes))
    
    if [[ $split_total_bytes -gt 0 ]]; then
      split_total_human=$(numfmt --to=iec-i --suffix=B "$split_total_bytes" 2>/dev/null || echo "$split_total_bytes bytes")
      split_total_gb=$(awk -v bytes="$split_total_bytes" 'BEGIN {printf "%.2f GB", bytes/1024/1024/1024}')
      echo "    ────────────────────────────────────"
      echo "    Total:      $split_total_human ($split_total_gb)"
    fi
  fi
  
  echo
  cd - > /dev/null
done

echo "========================================================"
echo "Size calculation complete"
