#!/usr/bin/env python3
"""
Generate average AP comparison chart across models and datasets for 1-second window.
Shows mean AP across PCC thresholds (0.50, 0.70, 0.90) with range as error bars.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Configuration
CSV_PATH = "results/pcc_comparison/pcc_comparison_summary.csv"
OUTPUT_PNG = "results/pcc_comparison/avg_ap_comparison_1seconds.png"
OUTPUT_CSV = "results/pcc_comparison/avg_ap_comparison_1seconds_data.csv"
OUTPUT_JSON = "results/pcc_comparison/pcc_variability_analysis_1seconds.json"

# Model colors (pastel palette for papers)
MODEL_COLORS = {
    'autoencoder': '#A8E6CF',           # Pastel Green
    'isolation_forest': '#A8D8EA',      # Pastel Blue
    'one_class_svm': '#FFAAA5',         # Pastel Red/Pink
    'local_outlier_factor': '#FFD3B6'   # Pastel Orange/Peach
}

# Dataset display names (shortened)
DATASET_NAMES = {
    'itp-downstream-http-flood': 'DS1',
    'itp-multivector-udp-100gbps-peak': 'DS2',
    'itp-synack-customer-outage': 'DS3'
}

# Model display names
MODEL_NAMES = {
    'autoencoder': 'AE',
    'isolation_forest': 'IF',
    'one_class_svm': 'OCSVM',
    'local_outlier_factor': 'LOF'
}


def load_and_process_data(csv_path, window='1seconds'):
    """Load CSV and calculate mean AP with range for each model-dataset combination."""
    df = pd.read_csv(csv_path)
    
    # Filter for 1-second window
    df_filtered = df[df['Window'] == window].copy()
    
    # Calculate mean, min, max, and range across PCC thresholds
    pcc_columns = ['PCC_0.50', 'PCC_0.70', 'PCC_0.90']
    
    results = []
    for _, row in df_filtered.iterrows():
        pcc_values = [row[col] for col in pcc_columns]
        
        results.append({
            'Dataset': row['Dataset'],
            'Model': row['Model'],
            'Mean_AP': np.mean(pcc_values),
            'Min_AP': np.min(pcc_values),
            'Max_AP': np.max(pcc_values),
            'Range': np.max(pcc_values) - np.min(pcc_values),
            'Std_AP': np.std(pcc_values)
        })
    
    return pd.DataFrame(results)


def create_grouped_bar_chart(df, output_path):
    """Create grouped bar chart with error bars."""
    
    # Get unique datasets and models
    datasets = ['itp-downstream-http-flood', 'itp-multivector-udp-100gbps-peak', 'itp-synack-customer-outage']
    models = ['autoencoder', 'isolation_forest', 'one_class_svm', 'local_outlier_factor']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bar_width = 0.18
    x_pos = np.arange(len(datasets))
    
    # Create a dummy error bar for legend
    errorbar_legend = None
    
    # Plot bars for each model
    for i, model in enumerate(models):
        means = []
        ranges = []
        
        for dataset in datasets:
            row = df[(df['Dataset'] == dataset) & (df['Model'] == model)]
            if not row.empty:
                means.append(row.iloc[0]['Mean_AP'])
                ranges.append(row.iloc[0]['Range'])
            else:
                means.append(0)
                ranges.append(0)
        
        # Calculate bar positions
        positions = x_pos + (i - 1.5) * bar_width
        
        # Create bars with error bars (showing range as +/- range/2)
        bars = ax.bar(positions, means, bar_width, 
                     label=MODEL_NAMES[model],
                     color=MODEL_COLORS[model],
                     alpha=0.85,
                     edgecolor='black',
                     linewidth=0.5)
        
        # Add error bars (range shown as half above and half below mean)
        error_bars = [r/2 for r in ranges]
        ax.errorbar(positions, means, yerr=error_bars,
                   fmt='none', 
                   ecolor='black',
                   capsize=3,
                   capthick=1.5,
                   alpha=0.7,
                   linewidth=1.5)
        
        # Add value labels on bars
        for j, (pos, mean, bar) in enumerate(zip(positions, means, bars)):
            if mean > 0:  # Only label non-zero values
                # Position text above error bar
                text_y = mean + error_bars[j] + 0.02
                ax.text(pos, text_y, f'{mean:.3f}',
                       ha='center', va='bottom',
                       fontsize=8, fontweight='bold',
                       rotation=0)
    
    ax.set_ylabel('Average Precision (AP)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Datasets', fontsize=14, fontweight='bold')
    # ax.set_title('AP Across PCC Thresholds (0.50, 0.70, 0.90)\n1-Second Time Window',
    #             fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([DATASET_NAMES[d] for d in datasets],
                       fontsize=12, fontweight='bold')
    
    ax.set_ylim([0, 1.05])
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    ax.legend(loc='upper right', fontsize=11, frameon=True,
             fancybox=True, shadow=True, ncol=2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {output_path}")
    plt.close()


def save_summary_csv(df, output_path):
    """Save the processed data to CSV."""
    # Sort by dataset and model for readability
    df_sorted = df.sort_values(['Dataset', 'Model'])
    
    # Add display names for better readability
    df_sorted['Dataset_Display'] = df_sorted['Dataset'].map(DATASET_NAMES)
    df_sorted['Model_Display'] = df_sorted['Model'].map(MODEL_NAMES)
    
    # Reorder columns
    cols = ['Dataset', 'Dataset_Display', 'Model', 'Model_Display', 
            'Mean_AP', 'Min_AP', 'Max_AP', 'Range', 'Std_AP']
    df_sorted = df_sorted[cols]
    
    df_sorted.to_csv(output_path, index=False, float_format='%.6f')
    print(f"Summary CSV saved to: {output_path}")


def generate_variability_analysis(csv_path, output_path, window='1seconds'):
    """Generate detailed JSON analysis of PCC variability for each model."""
    df = pd.read_csv(csv_path)
    df_filtered = df[df['Window'] == window].copy()
    
    pcc_columns = ['PCC_0.50', 'PCC_0.70', 'PCC_0.90']
    
    # Structure: models -> datasets -> metrics
    analysis = {
        "metadata": {
            "window": window,
            "pcc_thresholds": [0.50, 0.70, 0.90],
            "datasets": list(DATASET_NAMES.keys()),
            "models": list(MODEL_NAMES.keys())
        },
        "variability_by_model": {},
        "variability_by_dataset": {},
        "overall_statistics": {},
        "ranking": {}
    }
    
    # Analysis by model (across all datasets)
    for model in MODEL_NAMES.keys():
        model_data = df_filtered[df_filtered['Model'] == model]
        
        all_pcc_values = []
        dataset_breakdown = {}
        
        for _, row in model_data.iterrows():
            dataset = row['Dataset']
            pcc_values = [row[col] for col in pcc_columns]
            all_pcc_values.extend(pcc_values)
            
            dataset_breakdown[dataset] = {
                "pcc_0.50": float(row['PCC_0.50']),
                "pcc_0.70": float(row['PCC_0.70']),
                "pcc_0.90": float(row['PCC_0.90']),
                "mean": float(np.mean(pcc_values)),
                "std": float(np.std(pcc_values)),
                "min": float(np.min(pcc_values)),
                "max": float(np.max(pcc_values)),
                "range": float(np.max(pcc_values) - np.min(pcc_values)),
                "coefficient_of_variation": float(np.std(pcc_values) / np.mean(pcc_values)) if np.mean(pcc_values) > 0 else 0
            }
        
        analysis["variability_by_model"][model] = {
            "display_name": MODEL_NAMES[model],
            "overall_mean": float(np.mean(all_pcc_values)),
            "overall_std": float(np.std(all_pcc_values)),
            "overall_min": float(np.min(all_pcc_values)),
            "overall_max": float(np.max(all_pcc_values)),
            "overall_range": float(np.max(all_pcc_values) - np.min(all_pcc_values)),
            "coefficient_of_variation": float(np.std(all_pcc_values) / np.mean(all_pcc_values)) if np.mean(all_pcc_values) > 0 else 0,
            "dataset_breakdown": dataset_breakdown
        }
    
    # Analysis by dataset (across all models)
    for dataset in DATASET_NAMES.keys():
        dataset_data = df_filtered[df_filtered['Dataset'] == dataset]
        
        all_pcc_values = []
        model_breakdown = {}
        
        for _, row in dataset_data.iterrows():
            model = row['Model']
            pcc_values = [row[col] for col in pcc_columns]
            all_pcc_values.extend(pcc_values)
            
            model_breakdown[model] = {
                "pcc_0.50": float(row['PCC_0.50']),
                "pcc_0.70": float(row['PCC_0.70']),
                "pcc_0.90": float(row['PCC_0.90']),
                "mean": float(np.mean(pcc_values)),
                "std": float(np.std(pcc_values)),
                "min": float(np.min(pcc_values)),
                "max": float(np.max(pcc_values)),
                "range": float(np.max(pcc_values) - np.min(pcc_values)),
                "coefficient_of_variation": float(np.std(pcc_values) / np.mean(pcc_values)) if np.mean(pcc_values) > 0 else 0
            }
        
        analysis["variability_by_dataset"][dataset] = {
            "display_name": DATASET_NAMES[dataset],
            "overall_mean": float(np.mean(all_pcc_values)),
            "overall_std": float(np.std(all_pcc_values)),
            "overall_min": float(np.min(all_pcc_values)),
            "overall_max": float(np.max(all_pcc_values)),
            "overall_range": float(np.max(all_pcc_values) - np.min(all_pcc_values)),
            "coefficient_of_variation": float(np.std(all_pcc_values) / np.mean(all_pcc_values)) if np.mean(all_pcc_values) > 0 else 0,
            "model_breakdown": model_breakdown
        }
    
    # Overall statistics
    all_values = df_filtered[pcc_columns].values.flatten()
    analysis["overall_statistics"] = {
        "total_configurations": int(len(df_filtered) * 3),  # 3 PCC thresholds per config
        "global_mean": float(np.mean(all_values)),
        "global_std": float(np.std(all_values)),
        "global_min": float(np.min(all_values)),
        "global_max": float(np.max(all_values)),
        "global_range": float(np.max(all_values) - np.min(all_values)),
        "coefficient_of_variation": float(np.std(all_values) / np.mean(all_values)) if np.mean(all_values) > 0 else 0
    }
    
    # Rankings
    # Most stable models (lowest CV)
    model_cv = [(model, data['coefficient_of_variation']) 
                for model, data in analysis['variability_by_model'].items()]
    model_cv_sorted = sorted(model_cv, key=lambda x: x[1])
    
    # Best performing models (highest mean)
    model_mean = [(model, data['overall_mean']) 
                  for model, data in analysis['variability_by_model'].items()]
    model_mean_sorted = sorted(model_mean, key=lambda x: x[1], reverse=True)
    
    analysis["ranking"] = {
        "most_stable_models": [
            {
                "rank": i+1,
                "model": model,
                "display_name": MODEL_NAMES[model],
                "coefficient_of_variation": cv,
                "interpretation": "Lower is more stable across PCC thresholds"
            }
            for i, (model, cv) in enumerate(model_cv_sorted)
        ],
        "best_performing_models": [
            {
                "rank": i+1,
                "model": model,
                "display_name": MODEL_NAMES[model],
                "mean_ap": mean,
                "interpretation": "Higher mean AP across all configurations"
            }
            for i, (model, mean) in enumerate(model_mean_sorted)
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Variability analysis JSON saved to: {output_path}")
    
    return analysis


def print_summary_table(df):
    """Print a formatted summary table to console."""
    print("\n" + "="*80)
    print("AVERAGE AP COMPARISON - 1 SECOND WINDOW")
    print("="*80)
    
    datasets = ['itp-downstream-http-flood', 'itp-multivector-udp-100gbps-peak', 'itp-synack-customer-outage']
    models = ['autoencoder', 'isolation_forest', 'one_class_svm', 'local_outlier_factor']
    
    for dataset in datasets:
        print(f"\n{DATASET_NAMES[dataset]}")
        print("-" * 80)
        print(f"{'Model':<25} {'Mean AP':>12} {'Min AP':>12} {'Max AP':>12} {'Range':>12}")
        print("-" * 80)
        
        for model in models:
            row = df[(df['Dataset'] == dataset) & (df['Model'] == model)]
            if not row.empty:
                r = row.iloc[0]
                print(f"{MODEL_NAMES[model]:<25} {r['Mean_AP']:>12.4f} {r['Min_AP']:>12.4f} "
                      f"{r['Max_AP']:>12.4f} {r['Range']:>12.4f}")
    
    print("\n" + "="*80)
    
    # Find best performers
    print("\nBEST PERFORMERS BY DATASET (Highest Mean AP):")
    print("-" * 80)
    for dataset in datasets:
        df_dataset = df[df['Dataset'] == dataset]
        best = df_dataset.loc[df_dataset['Mean_AP'].idxmax()]
        print(f"{DATASET_NAMES[dataset]:<25} {MODEL_NAMES[best['Model']]:<25} "
              f"(Mean AP: {best['Mean_AP']:.4f})")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("GENERATING AVERAGE AP COMPARISON CHART")
    print("="*80 + "\n")
    
    print(f"Loading data from: {CSV_PATH}")
    df = load_and_process_data(CSV_PATH)
    print(f"Processed {len(df)} model-dataset combinations\n")
    
    print_summary_table(df)
    
    print("Generating grouped bar chart with error bars...")
    create_grouped_bar_chart(df, OUTPUT_PNG)
    
    print("\nSaving summary data...")
    save_summary_csv(df, OUTPUT_CSV)
    
    print("\nGenerating PCC variability analysis...")
    analysis = generate_variability_analysis(CSV_PATH, OUTPUT_JSON)
    
    # Print key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS FROM VARIABILITY ANALYSIS")
    print("="*80)
    print("\nMost Stable Models (Lowest Coefficient of Variation):")
    for rank_data in analysis['ranking']['most_stable_models'][:3]:
        print(f"  {rank_data['rank']}. {rank_data['display_name']:<25} CV: {rank_data['coefficient_of_variation']:.4f}")
    
    print("\nBest Performing Models (Highest Mean AP):")
    for rank_data in analysis['ranking']['best_performing_models'][:3]:
        print(f"  {rank_data['rank']}. {rank_data['display_name']:<25} Mean AP: {rank_data['mean_ap']:.4f}")
    
    print("\n" + "="*80)
    print("COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
