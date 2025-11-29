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

# Function to convert timestamp to epoch seconds
timestamp_to_epoch() {
  local ts="$1"
  date -d "$ts" +%s 2>/dev/null || echo "0"
}

# Function to calculate attack points for a given time window
calculate_attack_points() {
  local window_seconds="$1"
  shift
  local -a attack_periods=("$@")
  
  local total_attack_seconds=0
  
  # Process attack periods in pairs (start, end)
  local i=0
  while [[ $i -lt ${#attack_periods[@]} ]]; do
    local start="${attack_periods[$i]}"
    local end="${attack_periods[$((i+1))]}"
    
    local start_epoch=$(timestamp_to_epoch "$start")
    local end_epoch=$(timestamp_to_epoch "$end")
    
    if [[ "$start_epoch" != "0" && "$end_epoch" != "0" ]]; then
      local duration=$((end_epoch - start_epoch + 1))  # +1 to include both endpoints
      total_attack_seconds=$((total_attack_seconds + duration))
    fi
    
    i=$((i + 2))
  done
  
  # Convert to number of data points based on time window
  # Add window_seconds-1 to round up
  local attack_points=$(((total_attack_seconds + window_seconds - 1) / window_seconds))
  echo "$attack_points"
}

# Function to count total data points in test set based on actual file timestamps
count_test_points() {
  local test_pattern="$1"
  local window_seconds="$2"
  
  # Get first and last file to determine time range
  local files=($(ls $test_pattern 2>/dev/null | sort))
  
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "0"
    return
  fi
  
  # Extract timestamps from filenames (format: nfcapd.YYYYMMDDHHmm.csv)
  local first_file=$(basename "${files[0]}" .csv)
  local last_file=$(basename "${files[-1]}" .csv)
  
  local first_ts=${first_file#nfcapd.}
  local last_ts=${last_file#nfcapd.}
  
  # Convert to date format
  local first_date="${first_ts:0:4}-${first_ts:4:2}-${first_ts:6:2} ${first_ts:8:2}:${first_ts:10:2}:00"
  local last_date="${last_ts:0:4}-${last_ts:4:2}-${last_ts:6:2} ${last_ts:8:2}:${last_ts:10:2}:00"
  
  local first_epoch=$(timestamp_to_epoch "$first_date")
  local last_epoch=$(timestamp_to_epoch "$last_date")
  
  if [[ "$first_epoch" == "0" || "$last_epoch" == "0" ]]; then
    echo "0"
    return
  fi
  
  # Calculate time range in seconds (add 300 for the last file's duration)
  local time_range=$((last_epoch - first_epoch + 300))
  
  # Calculate total points
  local total_points=$((time_range / window_seconds))
  echo "$total_points"
}

echo "Calculating ground truth attack coverage..."
echo "============================================"
echo

for dir in "$DATASET_DIR"/$DIR_PATTERN; do
  [[ -d "$dir" && -d "$dir/raw" ]] || continue
  
  dataset_name=$(basename "$dir")
  
  echo "[$dataset_name]"
  echo ""
  
  cd "$dir/raw"
  
  # Define patterns and attack periods based on dataset (from config.py)
  case "$dataset_name" in
    "itp-downstream-http-flood")
      test_pattern="nfcapd.2025071[67]*.csv"
      
      # Attack periods for different time windows (format: "start1|end1|start2|end2|...")
      attack_1s="2025-07-16 20:27:14|2025-07-16 20:32:14|2025-07-16 22:23:07|2025-07-16 22:23:07|2025-07-16 22:24:57|2025-07-16 22:25:03|2025-07-16 22:33:05|2025-07-16 22:33:07|2025-07-17 00:09:16|2025-07-17 00:09:19|2025-07-17 00:30:15|2025-07-17 00:30:17"
      
      attack_10s="2025-07-16 20:27:10|2025-07-16 20:37:00|2025-07-16 22:24:50|2025-07-16 22:25:10|2025-07-16 22:33:00|2025-07-16 22:33:00|2025-07-17 00:09:10|2025-07-17 00:09:10|2025-07-17 00:30:10|2025-07-17 00:30:10|2025-07-17 00:38:00|2025-07-17 00:51:40"
      
      attack_60s="2025-07-16 20:27:00|2025-07-16 20:37:00|2025-07-16 22:25:00|2025-07-16 22:25:00|2025-07-16 22:33:00|2025-07-16 22:33:00|2025-07-17 00:09:00|2025-07-17 00:09:00|2025-07-17 00:30:00|2025-07-17 00:30:00|2025-07-17 00:38:00|2025-07-17 00:50:00"
      
      attack_300s="2025-07-16 20:25:00|2025-07-16 20:35:00|2025-07-16 22:30:00|2025-07-16 22:30:00|2025-07-17 00:30:00|2025-07-17 00:30:00|2025-07-17 00:40:00|2025-07-17 00:45:00"
      ;;
      
    "itp-synack-customer-outage")
      test_pattern="nfcapd.2025101[123]*.csv"
      
      attack_1s="2025-10-13 15:26:30|2025-10-13 17:16:20"
      attack_10s="$attack_1s"
      
      attack_60s="2025-10-13 15:28:00|2025-10-13 17:15:00"
      
      attack_300s="2025-10-13 15:35:00|2025-10-13 17:25:00"
      ;;
      
    "itp-multivector-udp-100gbps-peak")
      test_pattern="nfcapd.2025101[2-6]*.csv"
      
      attack_1s="2025-10-15 20:28:40|2025-10-15 20:42:10|2025-10-15 20:53:40|2025-10-15 21:06:20|2025-10-15 22:44:20|2025-10-15 22:54:40|2025-10-16 00:10:40|2025-10-16 00:19:40"
      attack_10s="$attack_1s"
      
      attack_60s="2025-10-15 20:29:00|2025-10-15 20:42:00|2025-10-15 20:54:00|2025-10-15 21:03:00|2025-10-15 22:46:00|2025-10-15 22:54:00|2025-10-16 00:11:00|2025-10-16 00:18:00"
      
      attack_300s="2025-10-15 20:25:00|2025-10-15 20:40:00|2025-10-15 20:50:00|2025-10-15 21:15:00|2025-10-15 22:45:00|2025-10-15 22:50:00|2025-10-16 00:10:00|2025-10-16 00:20:00"
      ;;
      
    *)
      cd - > /dev/null
      continue
      ;;
  esac
  
  # Calculate for each time window
  for window in 1 10 60 300; do
    window_name="${window}s"
    
    # Get attack periods for this window
    eval "attack_periods=\"\$attack_${window}s\""
    
    # Convert pipe-delimited string to array
    IFS='|' read -ra periods <<< "$attack_periods"
    
    # Calculate attack points
    attack_points=$(calculate_attack_points "$window" "${periods[@]}")
    
    # Calculate total test points
    total_points=$(count_test_points "$test_pattern" "$window")
    
    if [[ $total_points -gt 0 ]]; then
      # Calculate percentage
      percentage=$(awk -v attack="$attack_points" -v total="$total_points" 'BEGIN {printf "%.2f", (attack/total)*100}')
      
      echo "  Time window: ${window_name}"
      echo "    Attack points:     $attack_points"
      echo "    Total points:      $total_points"
      echo "    Attack coverage:   ${percentage}%"
      echo ""
    fi
  done
  
  echo ""
  cd - > /dev/null
done

echo "============================================"
echo "Attack coverage calculation complete"
