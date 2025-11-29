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

echo "Counting flows in CSV files for each dataset..."
echo "================================================"
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
  echo "  Files: $file_count"

  # Count total flows (excluding header lines from wc output)
  cd "$dir/raw"
  total_flows=$(ls *.csv 2>/dev/null | xargs wc -l | awk 'END {print $1}')

  # Convert to scientific notation
  if [[ -n "$total_flows" && "$total_flows" -gt 0 ]]; then
    scientific=$(awk -v n="$total_flows" 'BEGIN {
      if (n == 0) {
        print "0"
      } else {
        exponent = int(log(n)/log(10))
        mantissa = n / (10^exponent)
        printf "%.2f×10^%d", mantissa, exponent
      }
    }')

    echo "  Total flows: $total_flows ($scientific)"
  else
    echo "  Total flows: 0"
  fi

  # Count flows by train/validation/test splits based on config.py patterns
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
    # Count train flows
    train_flows=$(ls $train_pattern 2>/dev/null | xargs wc -l 2>/dev/null | awk 'END {print $1}')
    if [[ -n "$train_flows" && "$train_flows" -gt 0 ]]; then
      train_sci=$(awk -v n="$train_flows" 'BEGIN {
        exponent = int(log(n)/log(10))
        mantissa = n / (10^exponent)
        printf "%.2f×10^%d", mantissa, exponent
      }')
      echo "    Train:      $train_flows ($train_sci)"
    fi
    
    # Count validation flows
    validation_flows=$(ls $validation_pattern 2>/dev/null | xargs wc -l 2>/dev/null | awk 'END {print $1}')
    if [[ -n "$validation_flows" && "$validation_flows" -gt 0 ]]; then
      validation_sci=$(awk -v n="$validation_flows" 'BEGIN {
        exponent = int(log(n)/log(10))
        mantissa = n / (10^exponent)
        printf "%.2f×10^%d", mantissa, exponent
      }')
      echo "    Validation: $validation_flows ($validation_sci)"
    fi
    
    # Count test flows
    test_flows=$(ls $test_pattern 2>/dev/null | xargs wc -l 2>/dev/null | awk 'END {print $1}')
    if [[ -n "$test_flows" && "$test_flows" -gt 0 ]]; then
      test_sci=$(awk -v n="$test_flows" 'BEGIN {
        exponent = int(log(n)/log(10))
        mantissa = n / (10^exponent)
        printf "%.2f×10^%d", mantissa, exponent
      }')
      echo "    Test:       $test_flows ($test_sci)"
    fi
    
    # Calculate total of splits
    train_flows=${train_flows:-0}
    validation_flows=${validation_flows:-0}
    test_flows=${test_flows:-0}
    split_total=$((train_flows + validation_flows + test_flows))
    
    if [[ $split_total -gt 0 ]]; then
      split_total_sci=$(awk -v n="$split_total" 'BEGIN {
        exponent = int(log(n)/log(10))
        mantissa = n / (10^exponent)
        printf "%.2f×10^%d", mantissa, exponent
      }')
      echo "    ────────────────────────────────────"
      echo "    Total:      $split_total ($split_total_sci)"
    fi
  fi

  echo
  cd - > /dev/null
done

echo "================================================"
echo "Flow count complete"
