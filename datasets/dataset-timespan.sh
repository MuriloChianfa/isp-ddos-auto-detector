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

# Function to extract date from filename and calculate timespan
calculate_timespan() {
  local pattern="$1"
  
  # Get all matching files and extract dates
  files=($(ls $pattern 2>/dev/null | sort))
  
  if [[ ${#files[@]} -eq 0 ]]; then
    echo ""
    return
  fi
  
  # Extract timestamp from first and last file
  # Format: nfcapd.YYYYMMDDHHmm.csv
  first_file=$(basename "${files[0]}" .csv)
  last_file=$(basename "${files[-1]}" .csv)
  
  # Extract date parts: nfcapd.202507141200 -> 202507141200
  first_timestamp=${first_file#nfcapd.}
  last_timestamp=${last_file#nfcapd.}
  
  # Parse timestamps (YYYYMMDDHHmm)
  first_year=${first_timestamp:0:4}
  first_month=${first_timestamp:4:2}
  first_day=${first_timestamp:6:2}
  first_hour=${first_timestamp:8:2}
  first_min=${first_timestamp:10:2}
  
  last_year=${last_timestamp:0:4}
  last_month=${last_timestamp:4:2}
  last_day=${last_timestamp:6:2}
  last_hour=${last_timestamp:8:2}
  last_min=${last_timestamp:10:2}
  
  # Convert to epoch seconds
  first_epoch=$(date -d "${first_year}-${first_month}-${first_day} ${first_hour}:${first_min}:00" +%s 2>/dev/null)
  last_epoch=$(date -d "${last_year}-${last_month}-${last_day} ${last_hour}:${last_min}:00" +%s 2>/dev/null)
  
  if [[ -z "$first_epoch" || -z "$last_epoch" ]]; then
    echo ""
    return
  fi
  
  # Calculate difference in seconds
  diff_seconds=$((last_epoch - first_epoch))
  
  # Convert to days, hours, minutes
  days=$((diff_seconds / 86400))
  hours=$(((diff_seconds % 86400) / 3600))
  minutes=$(((diff_seconds % 3600) / 60))
  
  # Format output
  if [[ $days -gt 0 ]]; then
    printf "%d days, %02d:%02d" "$days" "$hours" "$minutes"
  else
    printf "%02d:%02d" "$hours" "$minutes"
  fi
  
  # Also show date range
  printf " (%s-%s-%s to %s-%s-%s)" "$first_year" "$first_month" "$first_day" "$last_year" "$last_month" "$last_day"
}

echo "Calculating dataset time spans..."
echo "=================================="
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
  
  cd "$dir/raw"
  
  # Calculate total timespan
  total_timespan=$(calculate_timespan "*.csv")
  if [[ -n "$total_timespan" ]]; then
    echo "  Total timespan: $total_timespan"
  fi
  
  # Calculate timespans by train/validation/test splits
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
    # Calculate train timespan
    train_timespan=$(calculate_timespan "$train_pattern")
    if [[ -n "$train_timespan" ]]; then
      echo "    Train:      $train_timespan"
    fi
    
    # Calculate validation timespan
    validation_timespan=$(calculate_timespan "$validation_pattern")
    if [[ -n "$validation_timespan" ]]; then
      echo "    Validation: $validation_timespan"
    fi
    
    # Calculate test timespan
    test_timespan=$(calculate_timespan "$test_pattern")
    if [[ -n "$test_timespan" ]]; then
      echo "    Test:       $test_timespan"
    fi
    
    echo "    ────────────────────────────────────"
    echo "    Note: Splits may have gaps between them"
  fi
  
  echo
  cd - > /dev/null
done

echo "=================================="
echo "Timespan calculation complete"
