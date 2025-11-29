#!/bin/bash

# Script to count columns in CSV files in the features directories of each dataset
# Shows the column count and filename for each CSV file

echo "Counting columns in feature CSV files..."
echo "=========================================="
echo ""

# Define datasets
datasets=(
  "itp-downstream-http-flood"
  "itp-multivector-udp-100gbps-peak"
  "itp-synack-customer-outage"
)

for dataset in "${datasets[@]}"; do
  features_dir="${dataset}/features"

  if [ ! -d "$features_dir" ]; then
    echo "[$dataset]"
    echo "  Features directory not found"
    echo ""
    continue
  fi

  echo "[$dataset]"

  # Count CSV files
  csv_count=$(find "$features_dir" -maxdepth 1 -name "*.csv" -type f | wc -l)

  if [ "$csv_count" -eq 0 ]; then
    echo "  No CSV files found"
    echo ""
    continue
  fi
  
  # Get column counts and check for consistency
  declare -A col_counts

  cd "$features_dir" || continue

  for f in *.csv; do
    [ -f "$f" ] || continue
    ncols=$(awk -F, 'NR==1 { print NF; exit }' "$f")
    col_counts[$ncols]=$((${col_counts[$ncols]:-0} + 1))
  done

  # Display results
  echo "  Total CSV files: $csv_count"
  echo ""
  echo "  Column distribution:"
  for ncols in $(echo "${!col_counts[@]}" | tr ' ' '\n' | sort -nr); do
    count=${col_counts[$ncols]}
    printf "    %3d columns: %5d files\n" "$ncols" "$count"
  done

  cd - > /dev/null || exit
  echo ""
  
  unset col_counts
done

echo "=========================================="
echo "Column count complete"
