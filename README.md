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

This research presents a comprehensive machine learning framework for unsupervised anomaly detection in Internet Transit Provider (ITP) network traffic, specifically targeting Distributed Denial of Service (DDoS) attacks. The framework implements and evaluates distinct anomaly detection algorithms (Isolation Forest, One-Class Support Vector Machine (OC-SVM), Local Outlier Factor (LOF) and Autoencoder) using NetFlow v9 data across multiple temporal resolutions (1s, 10s, 60s, 300s) and attack vectors. The system incorporates automated feature engineering with 150+ derived features including information-theoretic metrics (Shannon entropy), statistical moments, spectral analysis, and protocol-specific indicators. Our approach addresses the fundamental challenge of DDoS detection in operational ITP environments where labeled attack data is scarce and attack patterns evolve continuously.

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


## Results

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


<!-- ### Performance Metrics

> Environment: Intel Xeon E5-2683 v4 @ 2.10GHz, 128GB RAM, Debian 12

| dataset                          | model                |   accuracy |   precision |   recall |   f1_score |   roc_auc |    fpr |    mcc |
|----------------------------------|----------------------|------------|-------------|----------|------------|-----------|--------|--------|
| itp-downstream-http-flood        | autoencoder          |     0.9993 |      0.8032 |   0.9467 |     0.8691 |    0.9731 | 0.0005 | 0.8717 |
| itp-downstream-http-flood        | isolation_forest     |     0.9989 |      0.7023 |   0.8652 |     0.7753 |    0.9322 | 0.0008 | 0.779  |
| itp-downstream-http-flood        | local_outlier_factor |     0.9945 |      0.2929 |   1      |     0.4531 |    0.9972 | 0.0055 | 0.5397 |
| itp-downstream-http-flood        | one_class_svm        |     0.9978 |      0.7308 |   0.0596 |     0.1101 |    0.5298 | 0.0001 | 0.2082 |
| itp-multivector-udp-100gbps-peak | autoencoder          |     0.9986 |      0.9268 |   0.8895 |     0.9078 |    0.9445 | 0.0005 | 0.9073 |
| itp-multivector-udp-100gbps-peak | isolation_forest     |     0.9943 |      0.7363 |   0.4166 |     0.5321 |    0.7077 | 0.0012 | 0.5514 |
| itp-multivector-udp-100gbps-peak | local_outlier_factor |     0.997  |      0.8155 |   0.7857 |     0.8003 |    0.8921 | 0.0014 | 0.7989 |
| itp-multivector-udp-100gbps-peak | one_class_svm        |     0.9951 |      0.9368 |   0.4009 |     0.5615 |    0.7003 | 0.0002 | 0.6111 |
| itp-synack-customer-outage       | autoencoder          |     0.9946 |      0.9173 |   0.8838 |     0.9002 |    0.9408 | 0.0023 | 0.8976 |
| itp-synack-customer-outage       | isolation_forest     |     0.9931 |      0.9036 |   0.8395 |     0.8704 |    0.9185 | 0.0026 | 0.8674 |
| itp-synack-customer-outage       | local_outlier_factor |     0.9738 |      0.8187 |   0.072  |     0.1323 |    0.5358 | 0.0005 | 0.2381 |
| itp-synack-customer-outage       | one_class_svm        |     0.9783 |      0.7817 |   0.3056 |     0.4395 |    0.6516 | 0.0024 | 0.4807 | -->


<!-- ## Acknowledgments

A special thanks to ... -->


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
