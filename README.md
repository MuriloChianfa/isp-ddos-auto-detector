<h2 align="center">Unsupervised DDoS Detection in High-Speed Networks:<br>An Evaluation Using Real Transit Provider Data</h2>

<div align="center">

<a href="https://doc-artefatos.github.io/sbrc2026/results.html"><img src="slides/images/SBRC25_SF_SeloD.png" alt="SBRC 2026 Artifact Available badge" height="104"></a>
<a href="https://doc-artefatos.github.io/sbrc2026/results.html"><img src="slides/images/SBRC25_SF_SeloF.png" alt="SBRC 2026 Artifact Functional badge" height="104"></a>
<a href="https://doc-artefatos.github.io/sbrc2026/results.html"><img src="slides/images/SBRC25_SF_SeloS.png" alt="SBRC 2026 Artifact Sustainable badge" height="104"></a>
<a href="https://doc-artefatos.github.io/sbrc2026/results.html"><img src="slides/images/SBRC25_SF_SeloR.png" alt="SBRC 2026 Artifact Reproducible badge" height="104"></a>

[![SBRC 2026](https://img.shields.io/badge/SBRC-2026-blue)](https://sol.sbc.org.br/index.php/sbrc/article/view/42379)
[![DOI](https://img.shields.io/badge/DOI-10.5753%2Fsbrc.2026.19291-blue)](https://doi.org/10.5753/sbrc.2026.19291)
[![Paper](https://img.shields.io/badge/Paper-SBC-green)](https://sol.sbc.org.br/index.php/sbrc/article/view/42379)
[![Slides](https://img.shields.io/badge/Slides-Beamer-orange)](slides/main.pdf)

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Platform](https://img.shields.io/badge/platform-Linux-lightgrey.svg)](https://www.linux.org/)
[![Conda](https://img.shields.io/badge/conda-env-green.svg)](https://docs.conda.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/sklearn-1.6-blue.svg)](https://scikit-learn.org/)

</div>

Distributed denial-of-service (DDoS) attack detection has been widely studied in the past decade by academia. Despite progress having been made, recent surveys show that detection in environments such as Internet Transit Providers (ITP) remains challenging due to high-speed constraints. This study evaluates four anomaly detection algorithms, namely Autoencoder, Isolation Forest, Local Outlier Factor, and One Class Support Vector Machine, using three datasets collected from operational ITPs during confirmed DDoS attacks. The evaluation considers four temporal aggregation windows and three feature selection configurations, with the objective of analyzing the predictive capacity of the algorithms under different temporal and feature selection settings. The results show that the Autoencoder detection achieved the best results when using the most aggressive feature selection configuration and the shortest temporal aggregation windows.

## README Structure

1. [**Title and Abstract**](#abstract): Research overview and objectives
2. [**README Structure**](#agenda): Description of document organization
3. [**Basic Information**](#basic-information): Hardware and execution environment requirements
4. [**Dependencies**](#dependencies): Complete list of required libraries and tools
5. [**Security Concerns**](#security-concerns): Potential risks and security procedures
6. [**Installation**](#installation): Step-by-step instructions for environment setup
7. [**Minimal Test**](#minimal-test): Simple commands to validate installation
8. [**Experiments**](#experiments): Reproduction of main results presented in the paper
9. [**Datasets**](#datasets): Description of data used in the experiments
10. [**Acknowledgments**](#acknowledgments): Thanks to collaborating institutions
11. [**LICENSE**](#license): Dual licensing (MIT for code / CC BY 4.0 for datasets)

### Repository Structure

The organization of project files and directories:

```
isp-ddos-auto-detector/
├── config.py                    # Configuration and dataset definitions
├── main.py                      # CLI entry point
├── environment.yml              # Conda environment specification
├── framework/                   # Core framework modules
│   ├── pipeline.py             # Main analysis pipeline
│   ├── models/                 # Anomaly detection models
│   │   ├── isolation_forest.py
│   │   ├── one_class_svm.py
│   │   ├── local_outlier_factor.py
│   │   └── autoencoder.py
│   ├── features.py             # Feature engineering
│   ├── evaluation.py           # Model evaluation metrics
│   ├── optimization.py         # Hyperparameter tuning
│   ├── versioning.py           # Run versioning and management
│   ├── comparison.py           # Run comparison utilities
│   ├── visualization/          # Plotting utilities
│   └── ...
├── datasets/                    # NetFlow datasets
│   ├── itp-downstream-http-flood/
│   ├── itp-multivector-udp-100gbps-peak/
│   └── itp-synack-customer-outage/
└── results/                     # Evaluation results and plots
    ├── summary/
    ├── cross_evaluation/
    ├── versions/                # Versioned runs for comparison
    ├── comparisons/             # Comparison reports
    └── runs_index.json          # Index of all saved runs
```

## Basic Information

### Hardware Requirements

Experiments were executed on a machine with the following specifications:

- **Processor**: Dual Intel Xeon E5-2683 v4 @ 2.10 GHz
- **RAM Memory**: 128 GB DDR4 2133MHz RDIMM ECC
- **GPU**: NVIDIA GeForce GTX 1050 Ti with 4 GB of VRAM
- **Storage**: At least 20 GB free space for the datasets
- **Operating System**: Linux Debian 12 Kernel 6.1.0-26-amd64

### Software Requirements

- **Python**: Latest available version 3.12.2
- **Conda**: Miniconda or Anaconda (for environment management)
- **Git and Git LFS** To clone the repository and download derived datasets

## Dependencies

The framework has well-defined dependencies, managed through Conda. All dependencies are automatically installed through the `environment.yml` file, which contains:

- Packages with pinned versions to ensure reproducibility
- Conda channel configuration (pytorch, nvidia, conda-forge, defaults)
- Additional pip dependencies for packages not available in Conda

## Security Concerns

### Potential Risks

1. **Computational Resource Consumption**:
   - Model training can consume significant amounts of RAM during hyperparameter optimization
   - Batch executions can take several hours (up to 96h for all 144 complete scenarios)
   - It is recommended to monitor CPU/GPU/RAM usage during execution using `htop`/`nvidia-smi`

2. **Large File Downloads**:
   - The `git lfs pull` command will download derived datasets that can total several GB
   - Ensure you have a stable connection and sufficient disk space
   - In bandwidth-restricted environments, consider downloading only specific datasets

### Observations

- The framework **DOES NOT** modify system files outside the project directory
- The framework **DOES NOT** collect or transmit data to external servers
- All results and trained models are saved locally in `results/` and `cache/`

## Installation

### Step 1: Install Git LFS

Git LFS is required to download derived datasets (large files).

```bash
# Ubuntu/Debian
sudo apt install git-lfs
git lfs install

# Verify installation
git lfs version
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/MuriloChianfa/isp-ddos-auto-detector.git
cd isp-ddos-auto-detector
```

### Step 3: Download Derived Datasets

```bash
# This command may take a few minutes depending on connection
git lfs pull
```

### Step 4: Install Miniconda

Follow the official installation guide for your operating system:

[https://www.anaconda.com/docs/getting-started/miniconda/install/overview](https://www.anaconda.com/docs/getting-started/miniconda/install/overview)

<details>
<summary><b>Quick reference: non-interactive install on Linux x86_64 (click to expand)</b></summary>

```bash
# Download the latest Miniconda installer for linux x86_64
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/Miniconda3-latest-Linux-x86_64.sh

# Run the installer in batch (non-interactive) mode into ~/miniconda3
bash ~/Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Make conda available in the current shell
source $HOME/miniconda3/etc/profile.d/conda.sh

# (Optional) Initialize conda for future shell sessions
conda init bash

# Accept the Anaconda Terms of Service for the default channels
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Verify installation
conda --version
```

</details>

### Step 5: Create Conda Environment

```bash
# Create environment from specification file
conda env create -f environment.yml
```

### Step 6: Activate the Environment

```bash
conda activate nf-ae
```

**Important**: Always activate the `nf-ae` environment before executing any framework commands.

### Step 7: Verify Installation

After completing the above steps, the framework will be ready to use.

Proceed to the [**Minimal Test**](#minimal-test) section to validate the installation.


---

### (Optional) Feature Extraction from Raw Data

If you have raw NetFlow data and want to extract features:

```bash
# Set environment variables
export FILTER="dst as 65550"  # ASN65550 reserved for examples (RFC5398)
export DATASET_DIR=/path/to/raw/dataset
export OUTPUT_DIR=./datasets/dataset-name

# Run conversion script
./datasets/convert-to-csv.sh
```

> [!NOTE]
> Derived datasets are already included in the repository via Git LFS, so this step is optional.

## Minimal Test

This section presents simple commands to validate that the installation was successful. The tests below execute in less than 1 minute and do not require significant computational resources.

### Step 1: Display Help

Verifies that the CLI loads and all dependencies resolve.

```bash
python main.py --help
```

### Step 2: List Available Models

Verifies that the model factory is correctly wired.

```bash
python main.py --list-models
```

### Step 3: Run a minimal end-to-end detection

This runs the **complete pipeline** for a single (dataset, model, window) combination, complete deterministic, and direct comparability with the paper: One-Class SVM on the DS2 dataset with Δt = 1s. The combination is exactly the one reported in [**Table 5**](#expected-results-table) for itp-downstream-http-flood x One-Class SVM.

Typical time on the reference hardware: **under 30 seconds**.

```bash
python main.py -d itp-downstream-http-flood -m one_class_svm -t 1 --force-retrain
```

Expected console output (last lines):

```text
Confusion Matrix:
                 Predicted
               Normal  Attack
Actual Normal   138860     331
       Attack       21     298

Basic Performance Metrics:
  Accuracy:          0.9975
  Precision:         0.4738
  Recall:            0.9342
  F1-Score:          0.6287
  F2-Score:          0.7822

Advanced Performance Metrics:
  Matthews Correlation Coefficient: 0.6643
  ROC AUC Score:     0.9659
  Miss Rate (FNR):   0.0658
  Fallout (FPR):     0.0024

Sample Distribution:
  Total Samples:     139,510
  Attack Periods:    319
  Detected Anomalies: 629
```

## Experiments

This section presents detailed instructions to reproduce the main results from the paper. Experiments are organized into claims that correspond to the presented tables and figures.

### Dataset Context

The framework was evaluated using three real DDoS attack datasets collected from operational Internet Transit Provider (ITP) networks:

| Dataset | Attack Type | Characteristics |
|---------|-------------|------------------|
| **itp-downstream-http-flood** | HTTP Flood | Layer 7 attack targeting downstream customer |
| **itp-multivector-udp-100gbps-peak** | Multi-vector UDP | Volumetric attack reaching 100+ Gbps peak |
| **itp-synack-customer-outage** | SYN-ACK Reflection | Attack causing service degradation for two hours |

Each dataset contains NetFlow v9 telemetry data with derived features including Shannon entropy, statistical moments, protocol indicators, and temporal/volumetric metrics.

### Experiment Configuration

All experiments use the following configurations:

- **Temporal window (Δt)**: 1s, 10s, 60s, 300s
- **PCC cutoff (θ)**: 0.50, 0.70, 0.90
- **Models**: Autoencoder, Isolation Forest, One-Class SVM, Local Outlier Factor
- **Metrics**: Accuracy, Precision, Recall, F₁-Score, FPR, MCC, Average Precision

---

### Claim #1: Model Performance Metrics (Δt=1s, θ=0.50)

**Objective**: Reproduce the performance metrics table for the four models on the three datasets using 1-second temporal window and PCC threshold of 0.50.

#### Execution Commands

```bash
# Run PCC feature selection analysis
python main.py --analyze-correlation --correlation-threshold 0.50

# Run batch analysis for all datasets and models
python main.py --batch --batch-time-spans 1 --force --force-retrain
```

#### Results Visualization

To generate a consolidated summary:

```bash
python main.py --summary --summary-no-optimal-params --summary-no-stats --summary-time-spans 1
```

#### Expected Results Table

| Dataset | Model | Accuracy | Precision | Recall | F₁ | FPR | MCC |
|---------|-------|----------|-----------|--------|-------|--------|--------|
| **itp-downstream-http-flood** | Autoencoder | **0.9992** | **0.7772** | 0.9404 | **0.8511** | **0.0006** | **0.8546** |
| | Isolation Forest | 0.9875 | 0.1440 | 0.9060 | 0.2485 | 0.0123 | 0.3584 |
| | Local Outlier Factor | 0.9889 | 0.1702 | **0.9906** | 0.2904 | 0.0111 | 0.4082 |
| | One-Class SVM | 0.9975 | 0.4738 | 0.9342 | 0.6287 | 0.0024 | 0.6643 |
| **itp-multivector-udp-100gbps-peak** | Autoencoder | **0.9969** | **0.9138** | 0.6628 | **0.7683** | **0.0005** | **0.7768** |
| | Isolation Forest | 0.9897 | 0.4012 | 0.6672 | 0.5011 | 0.0078 | 0.5127 |
| | Local Outlier Factor | 0.9961 | 0.8718 | 0.5768 | 0.6943 | 0.0007 | 0.7074 |
| | One-Class SVM | 0.9796 | 0.2679 | **0.9444** | 0.4174 | 0.0202 | 0.4972 |
| **itp-synack-customer-outage** | Autoencoder | **0.9901** | **0.9177** | 0.7063 | 0.7983 | 0.0018 | 0.8004 |
| | Isolation Forest | 0.9750 | 0.8029 | 0.1348 | 0.2309 | **0.0009** | 0.3229 |
| | Local Outlier Factor | 0.9792 | 0.5888 | 0.8360 | 0.6910 | 0.0167 | 0.6918 |
| | One-Class SVM | 0.9893 | 0.7897 | **0.8377** | **0.8130** | 0.0064 | **0.8079** |

*Bold values indicate best performance for each metric within each dataset.*

<!-- ### Claim #2: Precision-Recall Curves and Model Comparison

---

**Objective**: Generate Precision-Recall curves comparing all models for each dataset.

#### Execution Commands

```bash
# Generate cross-evaluation and comparisons
python main.py --cross-evaluation
``` -->

<!-- ### Additional Experiments (Optional)

---

#### Feature Correlation Analysis

```bash
python main.py --analyze-correlation --correlation-threshold 0.70
```

#### Hyperparameter Optimization

```bash
python main.py --batch --optimize --optimize-n-iter 5
```

#### PCC Threshold Comparison

To reproduce results with different thresholds (0.50, 0.70, 0.90):

```bash
python main.py --save-run "pcc_050"
python main.py --save-run "pcc_070"
python main.py --save-run "pcc_090"

# Compare results
python main.py --cross-evaluation
``` -->

- **Reproducibility**: Results may vary slightly (~1-5%) due to random initialization of weights

## Feature Visual Analysis

The following visualizations show key features extracted from each dataset during the test phase (**Δt = 300s**). These features demonstrate the distinct behavioral patterns of different attack types:

<table>
  <tr>
    <th style="text-align: center;" width="33%">itp-downstream-http-flood</th>
    <th style="text-align: center;" width="33%">itp-multivector-udp-100gbps-peak</th>
    <th style="text-align: center;" width="33%">itp-synack-customer-outage</th>
  </tr>
  <tr>
    <td><img src="./results/itp-downstream-http-flood/300seconds/features/test/src_ip_entropy.png" width="100%" /></td>
    <td><img src="./results/itp-multivector-udp-100gbps-peak/300seconds/features/test/src_ip_entropy.png" width="100%" /></td>
    <td><img src="./results/itp-synack-customer-outage/300seconds/features/test/src_ip_entropy.png" width="100%" /></td>
  </tr>
  <tr>
    <td colspan="3" align="center"><p><i>Source IP Entropy</i></p></td>
  </tr>
  <tr>
    <td><img src="./results/itp-downstream-http-flood/300seconds/features/test/dst_port_entropy.png" width="100%" /></td>
    <td><img src="./results/itp-multivector-udp-100gbps-peak/300seconds/features/test/dst_port_entropy.png" width="100%" /></td>
    <td><img src="./results/itp-synack-customer-outage/300seconds/features/test/dst_port_entropy.png" width="100%" /></td>
  </tr>
  <tr>
    <td colspan="3" align="center"><p><i>Destination Port Entropy</i></p></td>
  </tr>
  <tr>
    <td><img src="./results/itp-downstream-http-flood/300seconds/features/test/bit_rate.png" width="100%" /></td>
    <td><img src="./results/itp-multivector-udp-100gbps-peak/300seconds/features/test/bit_rate.png" width="100%" /></td>
    <td><img src="./results/itp-synack-customer-outage/300seconds/features/test/bit_rate.png" width="100%" /></td>
  </tr>
  <tr>
    <td colspan="3" align="center"><p><i>Bit Rate over Time</i></p></td>
  </tr>
  <tr>
    <td><img src="./results/itp-downstream-http-flood/300seconds/features/test/syn_flag_ratio.png" width="100%" /></td>
    <td><img src="./results/itp-multivector-udp-100gbps-peak/300seconds/features/test/syn_flag_ratio.png" width="100%" /></td>
    <td><img src="./results/itp-synack-customer-outage/300seconds/features/test/syn_flag_ratio.png" width="100%" /></td>
  </tr>
  <tr>
    <td colspan="3" align="center"><p><i>SYN Flag Ratio</i></p></td>
  </tr>
  <tr>
    <td><img src="./results/itp-downstream-http-flood/300seconds/features/test/avg_duration.png" width="100%" /></td>
    <td><img src="./results/itp-multivector-udp-100gbps-peak/300seconds/features/test/avg_duration.png" width="100%" /></td>
    <td><img src="./results/itp-synack-customer-outage/300seconds/features/test/avg_duration.png" width="100%" /></td>
  </tr>
  <tr>
    <td colspan="3" align="center"><p><i>Average Flow Duration</i></p></td>
  </tr>
  <tr>
    <td><img src="./results/itp-downstream-http-flood/300seconds/features/test/size_uniformity.png" width="100%" /></td>
    <td><img src="./results/itp-multivector-udp-100gbps-peak/300seconds/features/test/size_uniformity.png" width="100%" /></td>
    <td><img src="./results/itp-synack-customer-outage/300seconds/features/test/size_uniformity.png" width="100%" /></td>
  </tr>
  <tr>
    <td colspan="3" align="center"><p><i>Size Uniformity</i></p></td>
  </tr>
  <tr>
    <td><img src="./results/itp-downstream-http-flood/300seconds/features/test/packets_kurtosis.png" width="100%" /></td>
    <td><img src="./results/itp-multivector-udp-100gbps-peak/300seconds/features/test/packets_kurtosis.png" width="100%" /></td>
    <td><img src="./results/itp-synack-customer-outage/300seconds/features/test/packets_kurtosis.png" width="100%" /></td>
  </tr>
  <tr>
    <td colspan="3" align="center"><p><i>Packets Kurtosis</i></p></td>
  </tr>
  <tr>
    <td><img src="./results/itp-downstream-http-flood/300seconds/features/test/cross_border_ratio.png" width="100%" /></td>
    <td><img src="./results/itp-multivector-udp-100gbps-peak/300seconds/features/test/cross_border_ratio.png" width="100%" /></td>
    <td><img src="./results/itp-synack-customer-outage/300seconds/features/test/cross_border_ratio.png" width="100%" /></td>
  </tr>
  <tr>
    <td colspan="3" align="center"><p><i>Cross-Border Ratio (international traffic proportion)</i></p></td>
  </tr>
</table>

### Complementary Visualizations

<table>
  <tr>
    <th style="text-align: center;" width="33%">itp-downstream-http-flood</th>
    <th style="text-align: center;" width="33%">itp-multivector-udp-100gbps-peak</th>
    <th style="text-align: center;" width="33%">itp-synack-customer-outage</th>
  </tr>
  <tr>
    <td><img src="./results/cross_evaluation/pr_curve_itp-downstream-http-flood_1seconds.png" width="100%" /></td>
    <td><img src="./results/cross_evaluation/pr_curve_itp-multivector-udp-100gbps-peak_1seconds.png" width="100%" /></td>
    <td><img src="./results/cross_evaluation/pr_curve_itp-synack-customer-outage_1seconds.png" width="100%" /></td>
  </tr>
  <tr>
    <td colspan="3" align="center"><p><i>Precision-Recall Curves (All Models, 1s resolution)</p></i></td>
  </tr>
  <tr>
    <td><img src="./results/pcc_comparison/pr_curve_pcc_comparison_itp-downstream-http-flood_1seconds_autoencoder.png" width="100%" /></td>
    <td><img src="./results/pcc_comparison/pr_curve_pcc_comparison_itp-multivector-udp-100gbps-peak_1seconds_autoencoder.png" width="100%" /></td>
    <td><img src="./results/pcc_comparison/pr_curve_pcc_comparison_itp-synack-customer-outage_1seconds_autoencoder.png" width="100%" /></td>
  </tr>
  <tr>
    <td colspan="3" align="center"><p><i>Average Precision (AP) for Autoencoder on each of the PCC thresholds tested</p></i></td>
  </tr>
  <tr>
    <td><img src="./results/itp-downstream-http-flood/1seconds/models/autoencoder/anomaly_detection.png" width="100%" /></td>
    <td><img src="./results/itp-multivector-udp-100gbps-peak/1seconds/models/autoencoder/anomaly_detection.png" width="100%" /></td>
    <td><img src="./results/itp-synack-customer-outage/1seconds/models/autoencoder/anomaly_detection.png" width="100%" /></td>
  </tr>
  <tr>
    <td colspan="3" align="center"><p><i>Anomaly Detection Timeline (Autoencoder, 1s resolution)</p></i></td>
  </tr>
  <tr>
    <td><img src="./results/itp-downstream-http-flood/1seconds/features/correlation/correlation_heatmap.png" width="100%" /></td>
    <td><img src="./results/itp-multivector-udp-100gbps-peak/1seconds/features/correlation/correlation_heatmap.png" width="100%" /></td>
    <td><img src="./results/itp-synack-customer-outage/1seconds/features/correlation/correlation_heatmap.png" width="100%" /></td>
  </tr>
  <tr>
    <td colspan="3" align="center"><p><i>Feature Correlation Heatmaps (Pearson's Correlation, 1s resolution)</p></i></td>
  </tr>
</table>

<table align="center">
  <tr>
    <td align="center" valign="middle">
      <a href="https://sol.sbc.org.br/index.php/sbrc/article/view/42379">
        <img src="slides/images/sbrc2026paper-cover.png" alt="SBRC 2026 paper cover" height="420">
      </a>
    </td>
    <td align="center" valign="middle">
      <a href="slides/main.pdf">
        <img src="slides/images/sbrc2026paper-presentation-cover.png" alt="SBRC 2026 presentation title slide" height="205">
      </a>
      <br>
      <a href="slides/main.pdf">
        <img src="slides/images/sbrc2026paper-presentation-cover2.png" alt="SBRC 2026 presentation slide" height="205">
      </a>
    </td>
  </tr>
</table>

## Acknowledgments

A special thanks to the ITPs for granting access to operational telemetry and for their support in the collection used in this study. Without this collaboration, it would not have been possible to evaluate the proposed methods under realistic ITP traffic conditions.

## LICENSE

This project uses **dual licensing**:

- **Code** is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
- **Datasets** are licensed under the **CC BY 4.0** - see the [LICENSE-CC](LICENSE-CC) file for details.

## Citation

```bibtex
@inproceedings{chianfa2026sbrc,
  author = {Chianfa, Murilo and Miani, Rodrigo and Zarpel{\~a}o, Bruno},
  title = {Unsupervised {DDoS} Detection in High-Speed Networks: An Evaluation Using Real Transit Provider Data},
  booktitle = {Anais do XLIV Simpósio Brasileiro de Redes de Computadores e Sistemas Distribuídos},
  location = {Praia do Forte/BA},
  year = {2026},
  keywords = {DDoS, ISP, Unsupervised},
  issn = {2177-9384},
  pages = {1485--1498},
  publisher = {SBC},
  address = {Porto Alegre, RS, Brasil},
  doi = {10.5753/sbrc.2026.19291},
  url = {https://sol.sbc.org.br/index.php/sbrc/article/view/42379}
}
```

<div align="center">
  <sub>Always eager to help ISPs with the fight against DDoS attacks!</sub>
</div>
