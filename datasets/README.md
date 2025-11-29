# Datasets

This directory contains the network flow datasets used for DDoS detection research.

## Dataset Statistics

### itp-downstream-http-flood
**Description:** ITP downstream HTTP flood attack dataset

- **Files:** 1,042 CSV files
- **Total size:** 1.1 GiB (1.00 GB)
- **Total flows:** 9,426,867 (9.43×10^6)
- **Timespan:** 3 days, 14:45 (2025-07-14 to 2025-07-17)

**Split breakdown:**
- **Train:** 2,864,284 flows (2.86×10^6) | 314 MiB (0.31 GB) | 23:55
- **Validation:** 2,682,619 flows (2.68×10^6) | 292 MiB (0.28 GB) | 23:55
- **Test:** 3,879,964 flows (3.88×10^6) | 422 MiB (0.41 GB) | 1 days, 14:45

**Attack coverage:**
- 1s window: 319 attack points / 139,800 total (0.23%)
- 10s window: 144 attack points / 13,980 total (1.03%)
- 60s window: 23 attack points / 2,330 total (0.99%)
- 300s window: 4 attack points / 466 total (0.86%)
---

### itp-multivector-udp-100gbps-peak
**Description:** ITP multi-vector UDP flood attack with 100Gbps peak

- **Files:** 4,935 CSV files
- **Total size:** 21 GiB (20.76 GB)
- **Total flows:** 195,156,403 (1.95×10^8)
- **Timespan:** 18 days, 02:35 (2025-10-08 to 2025-10-26)

**Split breakdown:**
- **Train:** 59,600,377 flows (5.96×10^7) | 6.4 GiB (6.38 GB) | 2 days, 23:55
- **Validation:** 20,957,501 flows (2.10×10^7) | 2.3 GiB (2.24 GB) | 23:55
- **Test:** 114,598,525 flows (1.15×10^8) | 13 GiB (12.14 GB) | 4 days, 02:00

**Attack coverage:**
- 1s window: 2,734 attack points / 353,100 total (0.77%)
- 10s window: 274 attack points / 35,310 total (0.78%)
- 60s window: 38 attack points / 5,885 total (0.65%)
- 300s window: 12 attack points / 1,177 total (1.02%)
---

### itp-synack-customer-outage
**Description:** ITP SYN+ACK flood attack for around 2hrs causing customer outage

- **Files:** 2,813 CSV files
- **Total size:** 241 MiB (0.23 GB)
- **Total flows:** 2,254,330 (2.25×10^6)
- **Timespan:** 9 days, 18:20 (2025-10-04 to 2025-10-13)

**Split breakdown:**
- **Train:** 777,774 flows (7.78×10^5) | 84 MiB (0.08 GB) | 2 days, 23:55
- **Validation:** 124,102 flows (1.24×10^5) | 14 MiB (0.01 GB) | 1 days, 23:55
- **Test:** 1,352,454 flows (1.35×10^6) | 144 MiB (0.14 GB) | 2 days, 18:20

**Attack coverage:**
- 1s window: 6,591 attack points / 239,100 total (2.76%)
- 10s window: 660 attack points / 23,910 total (2.76%)
- 60s window: 108 attack points / 3,985 total (2.71%)
- 300s window: 23 attack points / 797 total (2.89%)
---

## Data Format

Each CSV file contains network flow records with the following fields:

```
gmt_flow_arrived_time,duration,protocol,src_addr,src_port,dst_addr,dst_port,packets,bytes,flows,src_as,dst_as,src_country,dst_country,flags
```

## Scripts

- **`convert-to-csv.sh`** - Converts nfcapd files to CSV format using nfdump
- **`count-rows.sh`** - Counts flows in all datasets with train/validation/test breakdown
- **`extract.sh`** - Filter for specific customer flows on the raw dataset

> - Train/validation/test splits are defined in `config.py`, it may be invalid if you chage there
