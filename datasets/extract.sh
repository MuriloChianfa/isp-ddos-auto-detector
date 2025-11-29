#!/bin/bash

set -x

export NFGEODB=/root/mmc.nf
export PARALLEL_JOBS=30

export FILTER="dst as 65550" # example filter
export DATASET_DIR=/root/datasets/ds1/2025/07/
export OUTPUT_DIR=/root/datasets/itp-downstream-http-flood

/root/datasets/convert-to-csv.sh

export FILTER="dst net 0.0.0.0/0 or dst net ::" # example filter
export DATASET_DIR=/root/datasets/ds2/2025-10-13/
export OUTPUT_DIR=/root/datasets/itp-multivector-udp-100gbps-peak

/root/datasets/convert-to-csv.sh

export FILTER="dst net 0.0.0.0/0 or dst net ::" # example filter
export DATASET_DIR=/root/datasets/ds3
export OUTPUT_DIR=/root/datasets/itp-synack-customer-outage

/root/datasets/convert-to-csv.sh

export FILTER="not dst as 65530" # example filter
export DATASET_DIR=/root/datasets/ds4/
export OUTPUT_DIR=/root/datasets/isp-synflood-multiple-days

/root/datasets/convert-to-csv.sh

export FILTER="src net 100.64.0.0/10"
export DATASET_DIR=/root/datasets/ds5/
export OUTPUT_DIR=/root/isp-ddos-auto-detector/datasets/isp-cgnat-egress-anomalies

/root/isp-ddos-auto-detector/datasets/convert-to-csv.sh

