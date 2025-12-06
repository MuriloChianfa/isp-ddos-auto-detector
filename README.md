<h1 align="center">ISP DDoS Auto Detector</h1>
<!-- <h3 align="center">A Machine Learning Framework for Unsupervised DDoS Attack Detection in ITP Networks</h3> -->

<div align="center">

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Linux-lightgrey.svg)](https://www.linux.org/)
[![Conda](https://img.shields.io/badge/conda-env-green.svg)](https://docs.conda.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/sklearn-1.6-blue.svg)](https://scikit-learn.org/)

</div>

## Abstract

This research presents a comprehensive machine learning framework for unsupervised anomaly detection in Internet Transit Provider (ITP) network traffic, specifically targeting Distributed Denial of Service (DDoS) attacks. The framework implements and evaluates distinct anomaly detection algorithms (Isolation Forest, One-Class Support Vector Machine (OCSVM), Local Outlier Factor (LOF) and Autoencoder) using NetFlow v9 data across multiple temporal resolutions (1s, 10s, 60s, 300s) and attack vectors. The system incorporates automated feature engineering with 150+ derived features including information-theoretic metrics (Shannon entropy), statistical moments, spectral analysis, and protocol-specific indicators. Our approach addresses the fundamental challenge of DDoS detection in operational ITP environments where labeled attack data is scarce and attack patterns evolve continuously.

## Getting Started

### Installation

<details open>
  <summary style="font-size: 16px;"><strong>Setup Environment</strong></summary>

  ```bash
  # Firstly, install Git LFS
  git lfs install

  # Clone the repository
  git clone https://github.com/MuriloChianfa/isp-ddos-auto-detector.git
  cd isp-ddos-auto-detector

  # Pull large files, like derived datasets
  git lfs pull

  # Create conda environment from specification
  conda env create -f environment.yml

  # Activate the environment
  conda activate nf-ae
  ```

</details>
<details>
  <summary style="font-size: 16px;"><strong>Extract Features from Raw Datasets</strong></summary>

  ```bash
  # Set environment variables
  # ASN65550 reserved for example purposes (RFC5398)
  export FILTER="dst as 65550"
  export DATASET_DIR=/media/dataset/itp-downstream-http-flood
  export OUTPUT_DIR=./datasets/itp-downstream-http-flood

  # Convert raw NetFlow data to CSV features
  ./datasets/convert-to-csv.sh
  ```

</details>

## Usage

### Quick Start

<details open>
  <summary style="font-size: 16px;"><strong>Basic Examples</strong></summary>

  ```bash
  # List available models
  python main.py --list-models

  # List available datasets
  python main.py --list-datasets

  # Show all possible commands
  python main.py --help
  ```

</details>
<details open>
  <summary style="font-size: 16px;"><strong>Complete Analysis Flow</strong></summary>

  ```bash
  # Generate all features and their respective plots
  python main.py --generate-plots

  # Analyze all generated features using Pearson's Correlation
  python main.py --analyze-correlation --correlation-threshold 0.90

  # Evaluate hyperparameters through random search
  python main.py --batch --optimize --optimize-n-iter 10

  # Generate final results using the best feature and hyperparameter set
  python main.py --batch --force --force-retrain

  # Generate figures for comparing results
  python main.py --cross-evaluation

  # Show a summary about all results
  python main.py --summary

  # Save the results for later analysis
  python main.py --save-run "example_run"
  ```

</details>


## Project Structure

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

## Datasets

The framework was evaluated using three real-world DDoS attack datasets collected from operational Internet Transit Provider (ITP) networks. All datasets consist of NetFlow v9 telemetry data captured during confirmed DDoS attack incidents:

### Dataset Characteristics

| Dataset | Attack Type | Attack Traffic |
|---------|-------------|----------------|
| **itp-downstream-http-flood** | HTTP Flood | Layer 7 application flood targeting downstream customer |
| **itp-multivector-udp-100gbps-peak** | Multi-vector UDP | Volumetric attack reaching 100+ Gbps peak bandwidth |
| **itp-synack-customer-outage** | SYN-ACK Reflection | Attack causing customer service degradation for two hours |

### Feature Engineering

Each dataset undergoes comprehensive feature extraction, generating **150+ derived features** from raw NetFlow records:

- **Information-Theoretic Metrics**: Shannon entropy for IPs, ports, ASNs, GEO Codes
- **Statistical Moments**: Mean, variance, skewness, kurtosis of packet sizes
- **Protocol-Specific Indicators**: TCP flags distribution, TCP/UDP/ICMP ratios
- **Temporal Features**: Traffic rate variations, flow duration statistics
- **Volumetric Features**: Bytes/packets per flow, packet size distributions

The feature engineering pipeline automatically adapts to different temporal aggregation windows (Δt), allowing analysis at multiple time scales from near-real-time (1s) to longer-term trends (300s).


### Performance Metrics

> **Experimental Setup:** All experiments were carried out on a dedicated machine equipped with an Intel Xeon E5-2683 v4 CPU running at 2.10 GHz, 128 GB of RAM and an NVIDIA GeForce GTX 1050 Ti GPU. The table below presents detection performance for **Δt = 1s** and **θ = 0.50** regarding basic metrics:

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

*Where bold values indicate the best performance for each metric within each dataset. θ represents the PCC threshold.*


### Feature Analysis

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
    <td colspan="3" align="center"><p><i>Average Flow Duration (mean connection lifetime)</i></p></td>
  </tr>
  <tr>
    <td><img src="./results/itp-downstream-http-flood/300seconds/features/test/size_uniformity.png" width="100%" /></td>
    <td><img src="./results/itp-multivector-udp-100gbps-peak/300seconds/features/test/size_uniformity.png" width="100%" /></td>
    <td><img src="./results/itp-synack-customer-outage/300seconds/features/test/size_uniformity.png" width="100%" /></td>
  </tr>
  <tr>
    <td colspan="3" align="center"><p><i>Size Uniformity (packet size consistency)</i></p></td>
  </tr>
  <tr>
    <td><img src="./results/itp-downstream-http-flood/300seconds/features/test/packets_kurtosis.png" width="100%" /></td>
    <td><img src="./results/itp-multivector-udp-100gbps-peak/300seconds/features/test/packets_kurtosis.png" width="100%" /></td>
    <td><img src="./results/itp-synack-customer-outage/300seconds/features/test/packets_kurtosis.png" width="100%" /></td>
  </tr>
  <tr>
    <td colspan="3" align="center"><p><i>Packets Kurtosis (tailedness of packet distribution)</i></p></td>
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


## Results

The experimental validation encompassed **144 distinct scenarios**, systematically combining:

- **3 datasets**: Real-world DDoS attacks captured from ITP border routers (itp-downstream-http-flood, itp-multivector-udp-100gbps-peak, itp-synack-customer-outage)
- **4 anomaly detection models**: Autoencoder (AE), Isolation Forest (IF), One-Class SVM (OCSVM), and Local Outlier Factor (LOF)
- **4 temporal aggregation windows** (Δt): 1s, 10s, 60s, and 300s for feature extraction
- **3 PCC thresholds** (θ): 0.50, 0.70, and 0.90 for correlation-based feature selection

This comprehensive evaluation strategy ensures robust assessment across diverse attack patterns, model architectures, temporal resolutions, and feature dimensionality reduction approaches. Some results:

<table>
  <tr>
    <th style="text-align: center;" width="33%">itp-downstream-http-flood</th>
    <th style="text-align: center;" width="33%">itp-multivector-udp-100gbps-peak</th>
    <th style="text-align: center;" width="33%">itp-synack-customer-outage</th>
  </tr>
  <tr>
    <td><img src="./results/versions/0_90_pcc_n_iter_5/itp-downstream-http-flood/1seconds/models/autoencoder/anomaly_detection.png" width="100%" /></td>
    <td><img src="./results/versions/0_90_pcc_n_iter_5/itp-multivector-udp-100gbps-peak/1seconds/models/autoencoder/anomaly_detection.png" width="100%" /></td>
    <td><img src="./results/versions/0_90_pcc_n_iter_5/itp-synack-customer-outage/1seconds/models/autoencoder/anomaly_detection.png" width="100%" /></td>
  </tr>
  <tr>
    <td colspan="3" align="center"><p><i>Anomaly Detection Timeline (Autoencoder, 1s resolution)</p></i></td>
  </tr>
  <tr>
    <td><img src="./results/versions/0_90_pcc_n_iter_5/cross_evaluation/pr_curve_itp-downstream-http-flood_1seconds.png" width="100%" /></td>
    <td><img src="./results/versions/0_90_pcc_n_iter_5/cross_evaluation/pr_curve_itp-multivector-udp-100gbps-peak_1seconds.png" width="100%" /></td>
    <td><img src="./results/versions/0_90_pcc_n_iter_5/cross_evaluation/pr_curve_itp-synack-customer-outage_1seconds.png" width="100%" /></td>
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
    <td><img src="./results/itp-downstream-http-flood/1seconds/features/correlation/correlation_heatmap.png" width="100%" /></td>
    <td><img src="./results/itp-multivector-udp-100gbps-peak/1seconds/features/correlation/correlation_heatmap.png" width="100%" /></td>
    <td><img src="./results/itp-synack-customer-outage/1seconds/features/correlation/correlation_heatmap.png" width="100%" /></td>
  </tr>
  <tr>
    <td colspan="3" align="center"><p><i>Feature Correlation Heatmaps (Pearson's Correlation, 1s resolution)</p></i></td>
  </tr>
</table>

<div align="center">
  <img src="./results/versions/0_90_pcc_n_iter_5/cross_evaluation/heatmap_models_vs_windows.png" alt="Models vs Time Windows Performance" width="80%" />
  <p><i>Cross-evaluation heatmap showing F₁-scores across model-timespan combinations</i></p>
</div>

<div align="center">
  <img src="./results/pcc_comparison/avg_ap_comparison_1seconds.png" alt="Autoencoder showing the average precision for the three datasets" width="80%" />
  <p><i>Cross-evaluation bars showing average Average-Precision (AP) across each of dataset</i></p>
</div>


## Acknowledgments

A special thanks to the ITPs for granting access to operational telemetry and for their support in the collection used in this study. Without this collaboration, it would not have been possible to evaluate the proposed methods under realistic ITP traffic conditions.


<!-- ## Citation

If you use this framework in your research, please cite:

```bibtex
@software{chianfa2025isp_ddos,
  author = {Chianfa, Murilo},
  title = {{title}},
  year = {2025},
  month = {November},
  version = {1.0.0},
  url = {https://github.com/MuriloChianfa/isp-ddos-auto-detector}
}
``` -->

---

<div align="center">
  <sub>Always eager to help ISPs with the fight against DDoS attacks!</sub>
</div>
