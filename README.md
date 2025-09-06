# isp-ddos-auto-detector
Work in progress.

## Install dependencies
```bash
conda env create -f environment.yml
conda activate nf-ae
```

## Extract features from raw datasets
```bash
# ASN65550 reserved for example purposes described by RFC5398
export FILTER="dst as 65550"
export DATASET_DIR=/media/dataset/itp-downstream-http-flood
export OUTPUT_DIR=./datasets/itp-downstream-http-flood

./datasets/convert-to-csv.sh
```

## Running the project
```bash
python main.py
```
