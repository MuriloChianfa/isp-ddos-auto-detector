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
sudo TARGET_ASN=65550 DATASET_DIR=/media/dataset/ ./datasets/convert-to-csv.sh
```

## Running the project
```bash
python main.py
```
