"""
Feature correlation analysis using Pearson's Correlation Coefficient.

This module provides functionality to analyze feature correlations in training data,
identify highly correlated feature pairs, and generate configuration snippets with
recommended feature exclusions.
"""

import os
import json
import glob
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set


class CorrelationAnalyzer:
    """
    Analyzes feature correlations in training datasets using Pearson's Correlation Coefficient.
    
    This class loads training feature CSVs, calculates correlation matrices, identifies
    highly correlated feature pairs above a specified threshold, and determines which
    features should be dropped to reduce multicollinearity.
    """
    
    def __init__(self, dataset_name: str, time_span: int):
        """
        Initialize the correlation analyzer.
        
        Args:
            dataset_name: Name of the dataset to analyze
            time_span: Time span in seconds (1, 10, 60, 300)
        """
        self.dataset_name = dataset_name
        self.time_span = time_span
        self.output_dir = f"./results/{dataset_name}/{time_span}seconds/features/correlation"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _find_training_csv(self) -> str:
        """
        Find the training features CSV file with the most features (all available features).
        
        Returns:
            Path to the training CSV file with the most columns
            
        Raises:
            FileNotFoundError: If no training CSV is found
        """
        pattern = f"./datasets/{self.dataset_name}/features/train_features_{self.time_span}s_*.csv"
        csv_files = glob.glob(pattern)
        
        if not csv_files:
            raise FileNotFoundError(
                f"No training feature CSV found for dataset '{self.dataset_name}' "
                f"with time span {self.time_span}s.\n"
                f"Expected pattern: {pattern}\n"
                f"Please run feature extraction first."
            )
        
        # If multiple CSVs exist, find the one with the most features (columns)
        if len(csv_files) > 1:
            print(f"  Found {len(csv_files)} training CSVs, selecting one with most features...")
            max_features = 0
            selected_csv = csv_files[0]
            
            for csv_file in csv_files:
                # Quick column count check
                df_sample = pd.read_csv(csv_file, nrows=0)
                num_cols = len(df_sample.columns)
                if num_cols > max_features:
                    max_features = num_cols
                    selected_csv = csv_file
            
            print(f"  Selected: {os.path.basename(selected_csv)} ({max_features} columns)")
            return selected_csv
        
        return csv_files[0]
    
    def _load_features(self, csv_path: str) -> pd.DataFrame:
        """
        Load features from CSV file.
        
        Args:
            csv_path: Path to the features CSV file
            
        Returns:
            DataFrame with features (excluding timestamp column)
        """
        df = pd.read_csv(csv_path)
        
        # Remove timestamp column if present
        if 'timestamp' in df.columns:
            df = df.drop('timestamp', axis=1)
        
        # Remove any non-numeric columns
        df = df.select_dtypes(include=[np.number])
        
        return df
    
    def _find_empty_features(self, features: pd.DataFrame) -> Dict[str, Dict]:
        """
        Identify features that are empty or have zero/near-zero variance.
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            Dictionary mapping empty feature names to their details:
            {
                'feature_name': {
                    'reason': 'All NaN values' | 'Zero variance' | 'Near-zero variance',
                    'variance': 0.0,
                    'nan_count': 100,
                    'nan_percentage': 100.0
                }
            }
        """
        empty_features = {}
        
        for col in features.columns:
            nan_count = features[col].isna().sum()
            nan_percentage = (nan_count / len(features)) * 100
            
            # Check if all values are NaN
            if nan_count == len(features):
                empty_features[col] = {
                    'reason': 'All NaN values',
                    'variance': 0.0,
                    'nan_count': int(nan_count),
                    'nan_percentage': 100.0
                }
                continue
            
            # Check for zero or near-zero variance
            variance = features[col].var()
            
            if variance == 0:
                empty_features[col] = {
                    'reason': 'Zero variance (constant values)',
                    'variance': float(variance),
                    'nan_count': int(nan_count),
                    'nan_percentage': float(nan_percentage)
                }
            elif variance < 1e-10:  # Near-zero threshold
                empty_features[col] = {
                    'reason': 'Near-zero variance',
                    'variance': float(variance),
                    'nan_count': int(nan_count),
                    'nan_percentage': float(nan_percentage)
                }
        
        return empty_features
    
    def _calculate_correlation_matrix(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Pearson correlation matrix.
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            Correlation matrix as DataFrame
        """
        return features.corr(method='pearson')
    
    def _find_correlated_pairs(
        self, 
        corr_matrix: pd.DataFrame, 
        threshold: float
    ) -> List[Tuple[str, str, float]]:
        """
        Find pairs of features with correlation above threshold.
        
        Args:
            corr_matrix: Correlation matrix
            threshold: Correlation threshold (e.g., 0.90)
            
        Returns:
            List of tuples (feature1, feature2, correlation_value)
            Only includes upper triangle (no duplicates)
        """
        correlated_pairs = []
        
        # Get upper triangle indices (excluding diagonal)
        rows, cols = np.triu_indices_from(corr_matrix, k=1)
        
        for i, j in zip(rows, cols):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                feature1 = corr_matrix.index[i]
                feature2 = corr_matrix.columns[j]
                correlated_pairs.append((feature1, feature2, corr_value))
        
        # Sort by absolute correlation value (descending)
        correlated_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return correlated_pairs
    
    def _determine_features_to_drop(
        self,
        correlated_pairs: List[Tuple[str, str, float]],
        features: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Determine which features to drop from correlated pairs.
        
        Strategy: Drop the feature with lower variance from each correlated pair.
        Use a greedy approach to handle chains of correlations.
        
        Args:
            correlated_pairs: List of (feature1, feature2, correlation) tuples
            features: Original feature DataFrame for variance calculation
            
        Returns:
            Dictionary mapping dropped feature names to their details:
            {
                'feature_name': {
                    'correlated_with': 'other_feature',
                    'correlation': 0.95,
                    'variance': 123.45,
                    'kept_variance': 234.56,
                    'reason': 'Lower variance'
                }
            }
        """
        dropped_features = {}
        kept_features = set()
        
        # Calculate variance for all features
        variances = features.var()
        
        for feat1, feat2, corr_value in correlated_pairs:
            # Skip if either feature already handled
            if feat1 in dropped_features or feat2 in dropped_features:
                continue
            
            # If one is already kept, drop the other
            if feat1 in kept_features:
                drop_feature = feat2
                keep_feature = feat1
            elif feat2 in kept_features:
                drop_feature = feat1
                keep_feature = feat2
            else:
                # Drop the feature with lower variance
                var1 = variances[feat1]
                var2 = variances[feat2]
                
                if var1 < var2:
                    drop_feature = feat1
                    keep_feature = feat2
                else:
                    drop_feature = feat2
                    keep_feature = feat1
            
            dropped_features[drop_feature] = {
                'correlated_with': keep_feature,
                'correlation': float(corr_value),
                'variance': float(variances[drop_feature]),
                'kept_variance': float(variances[keep_feature]),
                'reason': 'Lower variance than correlated feature'
            }
            kept_features.add(keep_feature)
        
        return dropped_features
    
    def _save_correlation_matrix(self, corr_matrix: pd.DataFrame) -> str:
        """
        Save correlation matrix to CSV file.
        
        Args:
            corr_matrix: Correlation matrix DataFrame
            
        Returns:
            Path to saved file
        """
        output_path = os.path.join(self.output_dir, "correlation_matrix.csv")
        corr_matrix.to_csv(output_path)
        return output_path
    
    def _save_dropped_features(
        self, 
        empty_features: Dict[str, Dict],
        correlated_features: Dict[str, Dict]
    ) -> str:
        """
        Save dropped features information to JSON file.
        
        Args:
            empty_features: Dictionary of empty/zero-variance features
            correlated_features: Dictionary of correlated feature details
            
        Returns:
            Path to saved file
        """
        output_path = os.path.join(self.output_dir, "dropped_features.json")
        
        # Combine both dictionaries with category labels
        combined = {
            'empty_features': empty_features,
            'correlated_features': correlated_features,
            'summary': {
                'total_empty': len(empty_features),
                'total_correlated': len(correlated_features),
                'total_dropped': len(empty_features) + len(correlated_features)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(combined, f, indent=2)
        return output_path
    
    def _generate_config_snippet(
        self,
        original_features: List[str],
        dropped_features: Set[str],
        empty_features: Set[str],
        correlated_features: Set[str],
        dataset_config: Dict
    ) -> str:
        """
        Generate importable Python module with feature configuration.
        
        Args:
            original_features: List of all original features
            dropped_features: Set of all features to exclude
            empty_features: Set of empty/zero-variance features
            correlated_features: Set of correlated features
            dataset_config: Dataset configuration from config.py
            
        Returns:
            Path to saved config snippet file
        """
        # Filter features and sort alphabetically
        filtered_features = sorted([f for f in original_features if f not in dropped_features])
        
        snippet = []
        snippet.append('"""')
        snippet.append(f"Optimal feature configuration for {self.dataset_name} - {self.time_span}s window")
        snippet.append("")
        snippet.append("Generated from correlation analysis:")
        snippet.append(f"- Original features: {len(original_features)}")
        snippet.append(f"- Empty/zero-variance features: {len(empty_features)}")
        snippet.append(f"- Correlated features: {len(correlated_features)}")
        snippet.append(f"- Total dropped features: {len(dropped_features)}")
        snippet.append(f"- Remaining features: {len(filtered_features)}")
        snippet.append("")
        if empty_features:
            snippet.append(f"Empty/Zero-variance features removed:")
            for feat in sorted(empty_features):
                snippet.append(f"  - {feat}")
        if correlated_features:
            snippet.append("")
            snippet.append(f"Correlated features removed:")
            for feat in sorted(correlated_features):
                snippet.append(f"  - {feat}")
        snippet.append('"""')
        snippet.append("")
        
        # Create clean variable name from dataset name
        var_name = self.dataset_name.replace('-', '_').upper()
        
        # Main feature configuration
        snippet.append(f"# Feature configuration for {self.dataset_name} at {self.time_span}s window")
        snippet.append(f"FEATURE_CONFIG = [")
        for feat in filtered_features:
            snippet.append(f"    '{feat}',")
        snippet.append("]")
        snippet.append("")
        
        # Statistics for reference
        snippet.append("# Analysis statistics")
        snippet.append(f"STATS = {{")
        snippet.append(f"    'dataset': '{self.dataset_name}',")
        snippet.append(f"    'time_span': '{self.time_span}s',")
        snippet.append(f"    'original_features': {len(original_features)},")
        snippet.append(f"    'empty_features': {len(empty_features)},")
        snippet.append(f"    'correlated_features': {len(correlated_features)},")
        snippet.append(f"    'dropped_features': {len(dropped_features)},")
        snippet.append(f"    'remaining_features': {len(filtered_features)},")
        snippet.append(f"    'empty_feature_names': {sorted(list(empty_features))},")
        snippet.append(f"    'correlated_feature_names': {sorted(list(correlated_features))},")
        snippet.append("}")
        snippet.append("")
        
        # Usage example
        snippet.append("# Usage example:")
        snippet.append("#")
        snippet.append("# In config.py, you can import and use this configuration:")
        snippet.append("#")
        snippet.append(f"#   from results.{self.dataset_name.replace('-', '_')}.{self.time_span}seconds.features.correlation.optimal_features import FEATURE_CONFIG")
        snippet.append("#")
        snippet.append(f"#   DATASETS['{self.dataset_name}']['feature_config'] = FEATURE_CONFIG")
        snippet.append("#   # OR for window-specific config:")
        snippet.append(f"#   DATASETS['{self.dataset_name}']['windows']['{self.time_span}']['feature_config'] = FEATURE_CONFIG")
        
        output_path = os.path.join(self.output_dir, "optimal_features.py")
        with open(output_path, 'w') as f:
            f.write('\n'.join(snippet))
        
        return output_path
    
    def _generate_summary(
        self,
        csv_path: str,
        total_features: int,
        empty_features: Dict[str, Dict],
        correlated_pairs: List[Tuple[str, str, float]],
        dropped_features: Dict[str, Dict],
        artifacts: Dict[str, str]
    ) -> str:
        """
        Generate analysis summary text.
        
        Args:
            csv_path: Path to analyzed CSV
            total_features: Total number of features
            empty_features: Dictionary of empty/zero-variance features
            correlated_pairs: List of correlated pairs
            dropped_features: Dictionary of dropped features
            artifacts: Dictionary of artifact file paths
            
        Returns:
            Path to saved summary file
        """
        total_dropped = len(empty_features) + len(dropped_features)
        
        lines = []
        lines.append("=" * 80)
        lines.append(f"CORRELATION ANALYSIS SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Dataset: {self.dataset_name}")
        lines.append(f"Time Span: {self.time_span} seconds")
        lines.append(f"CSV Analyzed: {os.path.basename(csv_path)}")
        lines.append("")
        lines.append(f"Total Features: {total_features}")
        lines.append(f"Empty/Zero-Variance Features: {len(empty_features)}")
        lines.append(f"Correlation Pairs Found: {len(correlated_pairs)}")
        lines.append(f"Correlated Features to Drop: {len(dropped_features)}")
        lines.append(f"Total Features to Drop: {total_dropped}")
        lines.append(f"Remaining Features: {total_features - total_dropped}")
        lines.append("")
        
        if empty_features:
            lines.append("EMPTY/ZERO-VARIANCE FEATURES:")
            lines.append("-" * 80)
            for feat, details in sorted(empty_features.items()):
                lines.append(f"  -> {feat}")
                lines.append(f"    - Reason: {details['reason']}")
                lines.append(f"    - Variance: {details['variance']:.10f}")
                lines.append(f"    - NaN count: {details['nan_count']} ({details['nan_percentage']:.2f}%)")
                lines.append("")
        
        if dropped_features:
            lines.append("CORRELATED FEATURES (DROPPED):")
            lines.append("-" * 80)
            for feat, details in sorted(dropped_features.items()):
                lines.append(f"  -> {feat}")
                lines.append(f"    - Correlated with: {details['correlated_with']}")
                lines.append(f"    - Correlation: {details['correlation']:.4f}")
                lines.append(f"    - Variance: {details['variance']:.6f} (kept: {details['kept_variance']:.6f})")
                lines.append(f"    - Reason: {details['reason']}")
                lines.append("")
        
        if not empty_features and not dropped_features:
            lines.append("No features need to be dropped.")
            lines.append("")
        
        lines.append("=" * 80)
        lines.append("ARTIFACTS GENERATED:")
        lines.append("=" * 80)
        for name, path in artifacts.items():
            lines.append(f"  {name}: {path}")
        lines.append("")
        lines.append("=" * 80)
        
        summary_text = '\n'.join(lines)
        
        # Save to file
        output_path = os.path.join(self.output_dir, "analysis_summary.txt")
        with open(output_path, 'w') as f:
            f.write(summary_text)
        
        return output_path
    
    def analyze_training_features(
        self, 
        threshold: float = 0.90,
        dataset_config: Optional[Dict] = None
    ) -> Dict:
        """
        Perform complete correlation analysis on training features.
        
        Args:
            threshold: Correlation threshold for identifying highly correlated pairs
            dataset_config: Dataset configuration from config.py (for snippet generation)
            
        Returns:
            Dictionary containing analysis results:
            {
                'csv_path': str,
                'total_features': int,
                'correlation_matrix': pd.DataFrame,
                'correlated_pairs': List[Tuple],
                'dropped_features': Dict,
                'remaining_features': List[str],
                'artifacts': Dict[str, str]
            }
        """
        # Find and load training CSV
        csv_path = self._find_training_csv()
        print(f"  Loading: {os.path.basename(csv_path)}")
        
        features = self._load_features(csv_path)
        total_features = len(features.columns)
        print(f"  Features loaded: {total_features}")
        
        # Find empty/zero-variance features
        print(f"  Checking for empty/zero-variance features...")
        empty_features = self._find_empty_features(features)
        print(f"  Empty/zero-variance features: {len(empty_features)}")
        
        # Remove empty features from analysis
        if empty_features:
            features = features.drop(columns=list(empty_features.keys()))
            print(f"  Features after removing empty ones: {len(features.columns)}")
        
        # Calculate correlation matrix
        print(f"  Calculating Pearson correlation matrix...")
        corr_matrix = self._calculate_correlation_matrix(features)
        
        # Find correlated pairs
        print(f"  Finding correlated pairs (threshold: {threshold})...")
        correlated_pairs = self._find_correlated_pairs(corr_matrix, threshold)
        print(f"  Found {len(correlated_pairs)} highly correlated pairs")
        
        # Determine features to drop
        print(f"  Determining correlated features to drop...")
        dropped_correlated = self._determine_features_to_drop(correlated_pairs, features)
        print(f"  Correlated features to drop: {len(dropped_correlated)}")
        
        # Combine all dropped features
        all_dropped = set(empty_features.keys()) | set(dropped_correlated.keys())
        
        # Get remaining features (from original feature list)
        original_features = self._load_features(csv_path).columns.tolist()
        remaining_features = [f for f in original_features if f not in all_dropped]
        
        total_dropped = len(all_dropped)
        print(f"  Total features to drop: {total_dropped}")
        print(f"  Remaining features: {len(remaining_features)}")
        
        # Save artifacts
        print(f"  Saving artifacts...")
        artifacts = {}
        artifacts['Correlation Matrix'] = self._save_correlation_matrix(corr_matrix)
        artifacts['Dropped Features JSON'] = self._save_dropped_features(
            empty_features, 
            dropped_correlated
        )
        
        # Generate optimal features Python module
        if dataset_config:
            artifacts['Optimal Features Module'] = self._generate_config_snippet(
                original_features,
                all_dropped,
                set(empty_features.keys()),
                set(dropped_correlated.keys()),
                dataset_config
            )
        
        # Generate visualizations (will be called separately)
        from framework.visualization.correlation_plots import CorrelationVisualizer
        visualizer = CorrelationVisualizer(self.output_dir)
        
        artifacts['Correlation Heatmap'] = visualizer.plot_correlation_heatmap(
            corr_matrix, 
            threshold
        )
        artifacts['Correlation Distribution'] = visualizer.plot_correlation_distribution(
            corr_matrix,
            threshold
        )
        
        # Generate summary
        summary_path = self._generate_summary(
            csv_path,
            total_features,
            empty_features,
            correlated_pairs,
            dropped_correlated,
            artifacts
        )
        artifacts['Analysis Summary'] = summary_path
        
        return {
            'csv_path': csv_path,
            'total_features': total_features,
            'empty_features': empty_features,
            'correlation_matrix': corr_matrix,
            'correlated_pairs': correlated_pairs,
            'dropped_correlated': dropped_correlated,
            'all_dropped_features': all_dropped,
            'remaining_features': remaining_features,
            'artifacts': artifacts
        }


def run_correlation_analysis(
    datasets: Optional[List[str]] = None,
    time_spans: Optional[List[int]] = None,
    threshold: float = 0.90
):
    """
    Run correlation analysis across multiple datasets and time spans.
    
    Args:
        datasets: List of dataset names to analyze (None = all)
        time_spans: List of time spans to analyze (None = all)
        threshold: Correlation threshold
    """
    from config import DATASETS
    from framework.constants import SUPPORTED_TIME_SPANS
    
    # Default to all datasets and time spans
    if datasets is None:
        datasets = list(DATASETS.keys())
    if time_spans is None:
        time_spans = SUPPORTED_TIME_SPANS
    
    print("\n" + "=" * 80)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 80)
    print(f"Correlation Threshold: {threshold}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Time Spans: {', '.join(map(str, time_spans))} seconds")
    print("=" * 80)
    
    total_analyses = len(datasets) * len(time_spans)
    current = 0
    
    for dataset in datasets:
        dataset_config = DATASETS.get(dataset)
        if not dataset_config:
            print(f"\n[WARNING] Dataset '{dataset}' not found in config, skipping...")
            continue
        
        for time_span in time_spans:
            current += 1
            print(f"\n[{current}/{total_analyses}] Analyzing: {dataset} - {time_span}s window")
            print("-" * 80)
            
            try:
                analyzer = CorrelationAnalyzer(dataset, time_span)
                results = analyzer.analyze_training_features(
                    threshold=threshold,
                    dataset_config=dataset_config
                )
                
                print(f"  Analysis complete!")
                print(f"    - Total features: {results['total_features']}")
                print(f"    - Empty/zero-variance: {len(results['empty_features'])}")
                print(f"    - Correlated features: {len(results['dropped_correlated'])}")
                print(f"    - Total dropped: {len(results['all_dropped_features'])}")
                print(f"    - Remaining features: {len(results['remaining_features'])}")
                print(f"    - Output directory: {analyzer.output_dir}")
            except FileNotFoundError as e:
                print(f"  Error: {str(e)}")
            except Exception as e:
                print(f"  Unexpected error: {str(e)}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS COMPLETE")
    print("=" * 80)
