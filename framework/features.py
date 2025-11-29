import numpy as np
import pandas as pd
import os
import hashlib
from scipy.stats import entropy, skew, kurtosis
from .cache import DataCache
from .utils import get_time_span_frequency, get_time_span_floor, get_time_span_detailed_description
from multiprocessing import Pool, cpu_count
import functools


class NetworkFeatureExtractor:
    def __init__(self, time_span=300, use_cache=True, dataset_name=None, max_processes=None, feature_config=None, skip_ema=False):
        """
        Initialize NetworkFeatureExtractor
        
        Args:
            time_span (int): Time window in seconds for aggregation and rate calculations 
                           - 60 for 1-minute windows (uses received timestamps)
                           - 300 for 5-minute windows (uses file_timestamp) [default]
            use_cache (bool): Whether to use caching for feature extraction
            dataset_name (str): Name of the dataset being processed (for result organization)
            max_processes (int): Maximum number of processes to use for parallel processing.
            feature_config (dict): Configuration for feature selection and customization
            skip_ema (bool): If True, skip EMA smoothing (used during chunk processing in parallel mode)
        """
        self.time_span = time_span
        self.use_cache = use_cache
        self.dataset_name = dataset_name
        self.max_processes = max_processes
        self.feature_config = feature_config or {}
        self.cache = DataCache() if use_cache else None
        self.skip_ema = skip_ema
        
        # Get EMA alpha from config (can be overridden by window config)
        # Handle both dict and wrapped format
        if isinstance(self.feature_config, dict):
            self.ema_alpha = self.feature_config.get('ema_alpha', None)
        else:
            self.ema_alpha = None
    
    def _get_features_csv_path(self, split_name, feature_config=None):
        """Get the path for features CSV file
        
        Args:
            split_name (str): Name of the split (train, validation, test, etc.)
            feature_config (dict): Optional feature config to use instead of self.feature_config
        """
        base_dir = f"./datasets/{self.dataset_name}/features"
        os.makedirs(base_dir, exist_ok=True)
        
        # Use provided config or fall back to instance config
        config_to_use = feature_config if feature_config is not None else self.feature_config
        
        # Include feature config hash in filename to ensure cache invalidation when config changes
        config_suffix = ""
        if config_to_use:
            import hashlib
            # Convert config to hashable string (handle both list and dict formats)
            if isinstance(config_to_use, list):
                config_str = str(sorted(config_to_use))
            elif isinstance(config_to_use, dict):
                config_str = str(sorted(config_to_use.items()))
            else:
                config_str = str(config_to_use)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
            config_suffix = f"_cfg{config_hash}"
        
        return os.path.join(base_dir, f"{split_name}_features_{self.time_span}s{config_suffix}.csv")
    
    def _should_include_feature(self, feature_name, group_name=None):
        """Check if a feature should be included based on configuration"""
        if not self.feature_config:
            return True
        
        # Handle new list format
        if isinstance(self.feature_config, list):
            # If it's a list, just check if feature is in the list
            return feature_name in self.feature_config
        
        # Handle wrapped format with 'features' key
        if isinstance(self.feature_config, dict) and 'features' in self.feature_config:
            return feature_name in self.feature_config['features']
            
        # Handle old dict format with include_groups/exclude_features
        if isinstance(self.feature_config, dict):
            # Check if feature is explicitly excluded
            excluded_features = self.feature_config.get('exclude_features', [])
            if feature_name in excluded_features:
                return False
                
            # Check if feature group is included
            include_groups = self.feature_config.get('include_groups', [])
            if include_groups and group_name:
                return group_name in include_groups
            
        # Default to include if no specific configuration
        return True
    
    def _filter_features(self, features_df):
        """Filter features based on dataset configuration
        
        Supports multiple formats:
        1. New format (list): ['feature1', 'feature2', ...]
        2. New format with EMA (dict): {'features': [...], 'ema_alpha': 0.1}
        3. Old format (dict): {'include_groups': [...], 'exclude_features': [...]}
        """
        if not self.feature_config:
            return features_df
            
        from .constants import FEATURE_GROUPS
        
        # Determine which features to keep
        features_to_keep = ['timestamp']  # Always keep timestamp
        
        # Extract feature list if wrapped in dict with 'features' key
        feature_list = None
        if isinstance(self.feature_config, dict) and 'features' in self.feature_config:
            feature_list = self.feature_config['features']
        elif isinstance(self.feature_config, list):
            feature_list = self.feature_config
        
        # Check if new format (list of features)
        if feature_list is not None:
            # New format: direct list of feature names
            for feature in feature_list:
                if feature in features_df.columns:
                    features_to_keep.append(feature)
        
        # Check if old format (dict with include_groups)
        elif isinstance(self.feature_config, dict):
            include_groups = self.feature_config.get('include_groups', [])
            exclude_features = self.feature_config.get('exclude_features', [])
            
            if include_groups:
                # Old format: include features from specified groups
                for group_name in include_groups:
                    if group_name in FEATURE_GROUPS:
                        group_features = FEATURE_GROUPS[group_name]
                        for feature in group_features:
                            if feature in features_df.columns and feature not in exclude_features:
                                features_to_keep.append(feature)
            else:
                # Include all features except excluded ones
                for col in features_df.columns:
                    if col != 'timestamp' and col not in exclude_features:
                        features_to_keep.append(col)
        
        # Remove duplicates while preserving order
        features_to_keep = list(dict.fromkeys(features_to_keep))
        
        # Filter the dataframe
        available_features = [f for f in features_to_keep if f in features_df.columns]
        filtered_df = features_df[available_features].copy()
        
        # print(f"  Feature selection: {len(available_features)-1} features selected from {len(features_df.columns)-1} available")
        # if include_groups:
        #     print(f"  Included groups: {', '.join(include_groups)}")
        # if exclude_features:
        #     print(f"  Excluded features: {', '.join(exclude_features)}")
            
        return filtered_df
    
    def _apply_ema_if_configured(self, features_df):
        """
        Apply EMA smoothing to configured features if requested
        
        Args:
            features_df (pd.DataFrame): DataFrame with extracted features
            
        Returns:
            pd.DataFrame: DataFrame with EMA features added (if configured)
        """
        # Skip EMA if flag is set (used during parallel chunk processing)
        if self.skip_ema:
            return features_df
        
        from config import DEFAULT_EMA_CONFIG
        from .constants import FEATURE_GROUPS
        
        # Determine which EMA features are requested in feature_config
        requested_ema_features = []
        
        # Extract feature list based on config format
        feature_list = None
        if isinstance(self.feature_config, dict) and 'features' in self.feature_config:
            feature_list = self.feature_config['features']
        elif isinstance(self.feature_config, list):
            feature_list = self.feature_config
        
        # Check if new format (explicit list of features)
        if feature_list:
            # Find all features ending with '_ema' in the feature config
            requested_ema_features = [f for f in feature_list if f.endswith('_ema')]
        # Check if old format with include_groups
        elif isinstance(self.feature_config, dict) and 'include_groups' in self.feature_config:
            include_groups = self.feature_config.get('include_groups', [])
            if 'ema_smoothed' in include_groups:
                # Get all EMA features from the FEATURE_GROUPS constant
                requested_ema_features = FEATURE_GROUPS.get('ema_smoothed', [])
        
        # If no EMA features are requested, return unchanged
        if not requested_ema_features:
            return features_df
        
        # Get alpha from config (window-specific override or default)
        alpha = self.ema_alpha if self.ema_alpha is not None else DEFAULT_EMA_CONFIG['alpha']
        
        # Extract base feature names from requested EMA features
        # For example: 'packet_rate_ema' -> 'packet_rate'
        base_features_to_smooth = []
        for ema_feature in requested_ema_features:
            base_feature = ema_feature.replace('_ema', '')
            if base_feature in features_df.columns:
                base_features_to_smooth.append(base_feature)
        
        # Apply EMA smoothing only to requested features
        if base_features_to_smooth:
            features_df = self.apply_ema_smoothing(features_df, base_features_to_smooth, alpha=alpha)
        
        return features_df
    
    def _features_csv_exists(self, split_name):
        """Check if features CSV already exists"""
        csv_path = self._get_features_csv_path(split_name)
        return os.path.exists(csv_path)
    
    def load_features_from_csv(self, split_name):
        """Load features from CSV if it exists"""
        csv_path = self._get_features_csv_path(split_name)
        if os.path.exists(csv_path):
            print(f"Loading existing features for {split_name} from {csv_path}")
            df = pd.read_csv(csv_path)
            # Ensure timestamp column is properly parsed as datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return None
    
    def _load_from_all_features(self, split_name):
        """
        Try to load features by subsetting from an 'all_features' dataset if it exists.
        This avoids reprocessing raw data when only the feature selection has changed.
        
        Args:
            split_name (str): Name of the split (train, validation, test, etc.)
            
        Returns:
            pd.DataFrame or None: Subset of features if all_features CSV exists, None otherwise
        """
        from .constants import FEATURES_BY_ATTACK_TYPE
        
        # Get the path to the all_features CSV
        all_features_config = FEATURES_BY_ATTACK_TYPE.get('all_features')
        if not all_features_config:
            return None
        
        all_features_path = self._get_features_csv_path(split_name, feature_config=all_features_config)
        
        # Check if all_features CSV exists
        if not os.path.exists(all_features_path):
            return None
        
        print(f"  Found all_features dataset at {all_features_path}")
        print(f"  Loading and subsetting features instead of reprocessing raw data...")
        
        # Load the complete feature set
        all_features_df = pd.read_csv(all_features_path)
        
        # Ensure timestamp is properly parsed
        if 'timestamp' in all_features_df.columns:
            all_features_df['timestamp'] = pd.to_datetime(all_features_df['timestamp'])
        
        # Apply feature filtering to get only the columns we need
        filtered_df = self._filter_features(all_features_df)
        
        print(f"  Loaded {len(filtered_df)} records with {len(filtered_df.columns)-1} features (from {len(all_features_df.columns)-1} total features)")
        
        return filtered_df
    
    def extract_features_to_csv(self, loader, force_regenerate=False, parallel=True):
        """
        Extract features on-demand and save to CSV files with parallel processing
        
        Args:
            loader: NetworkDataLoader instance
            force_regenerate (bool): Force regeneration even if CSV files exist
            parallel (bool): Whether to use parallel processing for feature extraction
            
        Returns:
            dict: Dictionary with split_name -> features_df
        """
        features_dict = {}
        
        for split_name in loader.patterns.keys():
            csv_path = self._get_features_csv_path(split_name)
            
            # Check if we should use existing CSV
            if not force_regenerate and self._features_csv_exists(split_name):
                features_df = self.load_features_from_csv(split_name)
                if features_df is not None:
                    print(f"Using existing features for {split_name} from CSV")
                    features_dict[split_name] = features_df
                    continue
            
            # Try to load from all_features dataset if available (optimization)
            if not force_regenerate:
                features_df = self._load_from_all_features(split_name)
                if features_df is not None:
                    # Save the subset to its own CSV for future use
                    print(f"  Saving subset to {csv_path}")
                    features_df.to_csv(csv_path, index=False)
                    features_dict[split_name] = features_df
                    print(f"  {split_name.capitalize()} features: {len(features_df)} records")
                    continue
            
            print(f"Extracting features for {split_name} split...")
            
            # Show time window information once
            time_desc = get_time_span_detailed_description(self.time_span)
            print(f"  Using {time_desc} time windows ({self.time_span} seconds)")
            
            # Collect all data chunks first
            data_chunks = []
            for i, data_chunk in enumerate(loader.get_data_generator(split_name, parallel=True)):
                data_chunks.append(data_chunk)
                if i % 20 == 0 and i > 0:
                    print(f"  Collected {i} chunks for {split_name}...")
            
            print(f"  Processing {len(data_chunks)} chunks for {split_name}...")
            
            if parallel and len(data_chunks) > 1:
                # Use parallel processing for feature extraction
                if self.max_processes is not None:
                    num_processes = min(cpu_count(), len(data_chunks), self.max_processes)
                else:
                    num_processes = min(cpu_count(), len(data_chunks), 48)

                print(f"  Using {num_processes} parallel processes for feature extraction")
                
                # Create a partial function with the time_span and feature_config
                extract_func = functools.partial(self._extract_features_from_chunk, self.time_span, feature_config=self.feature_config)
                
                with Pool(num_processes) as pool:
                    all_features = pool.map(extract_func, data_chunks)
                
                # Filter out None results
                all_features = [f for f in all_features if f is not None]
            else:
                # Sequential processing
                all_features = []
                for i, chunk in enumerate(data_chunks):
                    # if i % 10 == 0 and i > 0:
                        # print(f"  Processing chunk {i}/{len(data_chunks)} for {split_name}...")
                    
                    chunk_features = self._extract_features_from_chunk(self.time_span, chunk, feature_config=self.feature_config)
                    if chunk_features is not None:
                        all_features.append(chunk_features)
            
            if all_features:
                # Combine all features
                combined_features = pd.concat(all_features, ignore_index=True)
                
                # Remove duplicate timestamps that occur from overlapping file processing
                original_count = len(combined_features)
                # Keep the first occurrence of each timestamp
                combined_features = combined_features.drop_duplicates(subset=['timestamp'], keep='first')
                dedup_count = len(combined_features)
                
                if original_count != dedup_count:
                    print(f"  Removed {original_count - dedup_count} duplicate timestamp records")
                
                # Sort by timestamp to ensure proper chronological order
                combined_features = combined_features.sort_values('timestamp').reset_index(drop=True)
                
                # Apply EMA smoothing AFTER combining all chunks (EMA requires sequential processing)
                combined_features = self._apply_ema_if_configured(combined_features)
                
                print(f"Saving {len(combined_features)} feature records to {csv_path}")
                combined_features.to_csv(csv_path, index=False)
                
                features_dict[split_name] = combined_features
                print(f"  {split_name.capitalize()} features: {len(combined_features)} records")
            else:
                print(f"  Warning: No features extracted for {split_name}")
        
        return features_dict
    
    @staticmethod
    def _extract_features_from_chunk(time_span, data_chunk, feature_config=None):
        """Static method to extract features from a data chunk - used for parallel processing"""
        try:
            # Create a temporary feature extractor for this chunk
            # Skip EMA during chunk processing - it will be applied after combining all chunks
            temp_extractor = NetworkFeatureExtractor(time_span=time_span, use_cache=False, feature_config=feature_config, skip_ema=True)
            return temp_extractor.prepare_advanced_features(data_chunk)
        except Exception as e:
            print(f"    Error extracting features from chunk: {str(e)}")
            return None
    
    @staticmethod
    def calculate_port_entropy(ports):
        """Calculate entropy of port distribution"""
        _, counts = np.unique(ports, return_counts=True)
        return entropy(counts, base=2)
    
    @staticmethod
    def calculate_as_entropy(as_numbers):
        """Calculate entropy of AS (Autonomous System) distribution"""
        # Filter out NaN values and convert to string for consistent handling
        valid_as = [str(x) for x in as_numbers if pd.notna(x)]
        if len(valid_as) == 0:
            return 0
        _, counts = np.unique(valid_as, return_counts=True)
        return entropy(counts, base=2)
    
    @staticmethod
    def calculate_geo_entropy(geo_codes):
        """Calculate entropy of geographic location distribution"""
        # Filter out NaN values and empty strings
        valid_geo = [str(x) for x in geo_codes if pd.notna(x) and str(x).strip() != '']
        if len(valid_geo) == 0:
            return 0
        _, counts = np.unique(valid_geo, return_counts=True)
        return entropy(counts, base=2)
    
    @staticmethod
    def calculate_simpson_index(values):
        """
        Calculate Simpson's Diversity Index
        Measures the probability that two randomly selected items belong to different categories
        Returns value between 0 (no diversity) and 1 (high diversity)
        """
        if len(values) == 0:
            return 0
        
        # Filter out NaN values
        valid_values = [x for x in values if pd.notna(x)]
        if len(valid_values) == 0:
            return 0
        
        _, counts = np.unique(valid_values, return_counts=True)
        proportions = counts / len(valid_values)
        
        # Simpson's Index: 1 - sum(p_i^2)
        simpson = 1 - np.sum(proportions ** 2)
        return simpson
    
    @staticmethod
    def calculate_gini_index(values):
        """
        Calculate Gini Coefficient (inequality measure)
        Returns value between 0 (perfect equality) and 1 (perfect inequality)
        """
        if len(values) == 0:
            return 0
        
        # Filter NaN and convert to array
        valid_values = np.array([x for x in values if pd.notna(x) and x >= 0])
        if len(valid_values) == 0:
            return 0
        
        # Gini coefficient calculation
        sorted_values = np.sort(valid_values)
        n = len(sorted_values)
        index = np.arange(1, n + 1)
        
        gini = (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
        return max(0, min(gini, 1))  # Clamp between 0 and 1
    
    @staticmethod
    def calculate_statistical_features(series, prefix=""):
        """
        Calculate comprehensive statistical features for a series
        Returns: mean, variance, std, min, max, skewness, kurtosis
        """
        features = {}
        
        if len(series) == 0:
            return {
                f'{prefix}mean': 0,
                f'{prefix}variance': 0,
                f'{prefix}std': 0,
                f'{prefix}min': 0,
                f'{prefix}max': 0,
                f'{prefix}skewness': 0,
                f'{prefix}kurtosis': 0,
                f'{prefix}range': 0,
                f'{prefix}cv': 0
            }
        
        # Filter NaN values
        valid_series = series.dropna()
        if len(valid_series) == 0:
            return {
                f'{prefix}mean': 0,
                f'{prefix}variance': 0,
                f'{prefix}std': 0,
                f'{prefix}min': 0,
                f'{prefix}max': 0,
                f'{prefix}skewness': 0,
                f'{prefix}kurtosis': 0,
                f'{prefix}range': 0,
                f'{prefix}cv': 0
            }
        
        features[f'{prefix}mean'] = valid_series.mean()
        features[f'{prefix}variance'] = valid_series.var()
        features[f'{prefix}std'] = valid_series.std()
        features[f'{prefix}min'] = valid_series.min()
        features[f'{prefix}max'] = valid_series.max()
        features[f'{prefix}range'] = features[f'{prefix}max'] - features[f'{prefix}min']
        
        # Coefficient of variation
        if features[f'{prefix}mean'] > 0:
            features[f'{prefix}cv'] = features[f'{prefix}std'] / features[f'{prefix}mean']
        else:
            features[f'{prefix}cv'] = 0
        
        # Requires at least 3 values and sufficient variance
        if len(valid_series) >= 3:
            # Check if data has sufficient variance to avoid precision loss
            if features[f'{prefix}std'] > 1e-10: # nearly identical values
                features[f'{prefix}skewness'] = skew(valid_series)
                features[f'{prefix}kurtosis'] = kurtosis(valid_series)
            else:
                # Data is too uniform, let's set default values so
                features[f'{prefix}skewness'] = 0
                features[f'{prefix}kurtosis'] = 0
        else:
            features[f'{prefix}skewness'] = 0
            features[f'{prefix}kurtosis'] = 0
        
        return features
    
    @staticmethod
    def apply_ema_smoothing(features_df, ema_features_list, alpha):
        """
        Apply Exponential Moving Average smoothing to selected features
        Creates new features with _ema suffix
        
        Args:
            features_df (pd.DataFrame): DataFrame with features and timestamp
            ema_features_list (list): List of feature names to apply EMA smoothing
            alpha (float): Smoothing factor (0 < alpha < 1). Lower = more smoothing
        
        Returns:
            pd.DataFrame: DataFrame with added EMA features (original_name + '_ema')
        """
        df_smoothed = features_df.copy()
        
        for feature in ema_features_list:
            if feature in df_smoothed.columns:
                # Calculate EMA using pandas ewm (exponential weighted moving average)
                # adjust=False means we use the recursive formula: ema_t = alpha * x_t + (1-alpha) * ema_{t-1}
                ema_values = df_smoothed[feature].ewm(alpha=alpha, adjust=False).mean()
                df_smoothed[f'{feature}_ema'] = ema_values
        
        return df_smoothed
    
    @staticmethod
    def calculate_uniqueness_features(values, prefix=""):
        """
        Calculate uniqueness-related features
        Returns: unique count, uniqueness ratio, collision probability
        """
        features = {}
        
        if len(values) == 0:
            return {
                f'{prefix}unique_count': 0,
                f'{prefix}uniqueness_ratio': 0,
                f'{prefix}collision_probability': 1.0
            }
        
        # Filter NaN values
        valid_values = [x for x in values if pd.notna(x)]
        if len(valid_values) == 0:
            return {
                f'{prefix}unique_count': 0,
                f'{prefix}uniqueness_ratio': 0,
                f'{prefix}collision_probability': 1.0
            }
        
        unique_count = len(np.unique(valid_values))
        features[f'{prefix}unique_count'] = unique_count
        features[f'{prefix}uniqueness_ratio'] = unique_count / len(valid_values)
        
        # Collision probability = 1 - Simpson's Index
        _, counts = np.unique(valid_values, return_counts=True)
        proportions = counts / len(valid_values)
        simpson = 1 - np.sum(proportions ** 2)
        features[f'{prefix}collision_probability'] = 1 - simpson
        
        return features
    
    @staticmethod
    def calculate_fan_in_out_features(group):
        """
        Calculate fan-in and fan-out patterns
        Fan-in: many sources -> one destination
        Fan-out: one source -> many destinations
        """
        features = {}
        
        if len(group) == 0:
            return {
                'max_fan_in': 0,
                'avg_fan_in': 0,
                'max_fan_out': 0,
                'avg_fan_out': 0,
                'fan_in_std': 0,
                'fan_out_std': 0,
                'fan_in_out_ratio': 0
            }
        
        # Fan-in: count sources per destination
        fan_in = group.groupby('dstAddr')['srcAddr'].nunique()
        features['max_fan_in'] = fan_in.max() if len(fan_in) > 0 else 0
        features['avg_fan_in'] = fan_in.mean() if len(fan_in) > 0 else 0
        features['fan_in_std'] = fan_in.std() if len(fan_in) > 0 else 0
        
        # Fan-out: count destinations per source
        fan_out = group.groupby('srcAddr')['dstAddr'].nunique()
        features['max_fan_out'] = fan_out.max() if len(fan_out) > 0 else 0
        features['avg_fan_out'] = fan_out.mean() if len(fan_out) > 0 else 0
        features['fan_out_std'] = fan_out.std() if len(fan_out) > 0 else 0
        
        # Fan-in to fan-out ratio
        if features['avg_fan_out'] > 0:
            features['fan_in_out_ratio'] = features['avg_fan_in'] / features['avg_fan_out']
        else:
            features['fan_in_out_ratio'] = 0
        
        return features
    
    @staticmethod
    def calculate_asymmetry_features(group):
        """
        Calculate asymmetry features between source and destination
        """
        features = {}
        
        if len(group) == 0:
            return {
                'src_dst_port_entropy_diff': 0,
                'src_dst_ip_entropy_diff': 0,
                'port_entropy_abs_diff': 0,
                'ip_entropy_abs_diff': 0
            }
        
        # Port entropy difference
        src_port_entropy = NetworkFeatureExtractor.calculate_port_entropy(group['srcPort'].values)
        dst_port_entropy = NetworkFeatureExtractor.calculate_port_entropy(group['dstPort'].values)
        features['src_dst_port_entropy_diff'] = src_port_entropy - dst_port_entropy
        features['port_entropy_abs_diff'] = abs(src_port_entropy - dst_port_entropy)
        
        # IP entropy difference
        src_ip_entropy = NetworkFeatureExtractor.calculate_port_entropy(group['srcAddr'].values)
        dst_ip_entropy = NetworkFeatureExtractor.calculate_port_entropy(group['dstAddr'].values)
        features['src_dst_ip_entropy_diff'] = src_ip_entropy - dst_ip_entropy
        features['ip_entropy_abs_diff'] = abs(src_ip_entropy - dst_ip_entropy)
        
        return features
    
    @staticmethod
    def calculate_amplification_features(group):
        """
        Calculate features for detecting amplification attacks (DNS, NTP, etc.)
        """
        features = {}
        
        if len(group) == 0:
            return {
                'avg_request_reply_ratio': 0,
                'small_request_large_reply_ratio': 0,
                'amplification_score': 0,
                'udp_amplification_potential': 0
            }
        
        # Focus on UDP traffic (common for amplification attacks)
        udp_flows = group[group['proto'] == 17]
        
        if len(udp_flows) > 0:
            # Identify potential request/reply patterns by packet size
            # Small requests: <= 100 bytes, Large replies: >= 500 bytes
            small_flows = udp_flows[udp_flows['bytes'] <= 100]
            large_flows = udp_flows[udp_flows['bytes'] >= 500]
            
            features['small_request_large_reply_ratio'] = len(large_flows) / max(len(small_flows), 1)
            
            # Average size ratio
            if len(udp_flows) > 0:
                avg_size = udp_flows['bytes'].mean()
                max_size = udp_flows['bytes'].max()
                if avg_size > 0:
                    features['avg_request_reply_ratio'] = max_size / avg_size
                else:
                    features['avg_request_reply_ratio'] = 0
            else:
                features['avg_request_reply_ratio'] = 0
            
            # Amplification score (high ratio of large to small packets)
            if len(small_flows) > 0:
                features['amplification_score'] = len(large_flows) / len(udp_flows)
            else:
                features['amplification_score'] = 0
            
            # UDP amplification potential (common ports: DNS 53, NTP 123, etc.)
            amp_ports = [53, 123, 161, 389, 1900]
            amp_flows = udp_flows[udp_flows['dstPort'].isin(amp_ports) | udp_flows['srcPort'].isin(amp_ports)]
            features['udp_amplification_potential'] = len(amp_flows) / len(udp_flows) if len(udp_flows) > 0 else 0
        else:
            features['avg_request_reply_ratio'] = 0
            features['small_request_large_reply_ratio'] = 0
            features['amplification_score'] = 0
            features['udp_amplification_potential'] = 0
        
        return features
    
    @staticmethod
    def calculate_concentration_features(values, top_k=[1, 5, 10], prefix=""):
        """
        Calculate top-K concentration and heavy-tail metrics
        """
        features = {}
        
        if len(values) == 0:
            for k in top_k:
                features[f'{prefix}top_{k}_ratio'] = 0
            features[f'{prefix}heavy_tail_index'] = 0
            features[f'{prefix}concentration_score'] = 0
            return features
        
        # Count occurrences
        value_counts = pd.Series(values).value_counts()
        total = len(values)
        
        # Top-K concentration
        for k in top_k:
            if len(value_counts) >= k:
                top_k_sum = value_counts.head(k).sum()
                features[f'{prefix}top_{k}_ratio'] = top_k_sum / total
            else:
                features[f'{prefix}top_{k}_ratio'] = value_counts.sum() / total if len(value_counts) > 0 else 0
        
        # Heavy-tail index (Hill estimator) - simplified version
        # Higher values indicate heavier tails
        if len(value_counts) > 1:
            sorted_counts = np.sort(value_counts.values)[::-1]
            k_hill = min(len(sorted_counts) // 4, 100)  # Use top 25% or 100 samples
            if k_hill > 1:
                log_ratios = np.log(sorted_counts[:k_hill] / sorted_counts[k_hill])
                features[f'{prefix}heavy_tail_index'] = np.mean(log_ratios)
            else:
                features[f'{prefix}heavy_tail_index'] = 0
        else:
            features[f'{prefix}heavy_tail_index'] = 0
        
        # Concentration score (Herfindahl-Hirschman Index)
        proportions = value_counts / total
        features[f'{prefix}concentration_score'] = np.sum(proportions ** 2)
        
        return features
    
    @staticmethod
    def calculate_port_usage_features(group):
        """
        Calculate port usage peculiarities
        """
        features = {}
        
        if len(group) == 0:
            return {
                'ephemeral_port_ratio': 0,
                'well_known_port_ratio': 0,
                'dst_port_concentration': 0,
                'src_port_concentration': 0,
                'registered_port_ratio': 0
            }
        
        # Port ranges: 0-1023 (well-known), 1024-49151 (registered), 49152-65535 (ephemeral)
        well_known_dst = group[group['dstPort'] < 1024]
        registered_dst = group[(group['dstPort'] >= 1024) & (group['dstPort'] < 49152)]
        ephemeral_dst = group[group['dstPort'] >= 49152]
        
        features['well_known_port_ratio'] = len(well_known_dst) / len(group)
        features['registered_port_ratio'] = len(registered_dst) / len(group)
        features['ephemeral_port_ratio'] = len(ephemeral_dst) / len(group)
        
        # Destination port concentration (Herfindahl index)
        dst_port_counts = group['dstPort'].value_counts()
        dst_port_proportions = dst_port_counts / len(group)
        features['dst_port_concentration'] = np.sum(dst_port_proportions ** 2)
        
        # Source port concentration
        src_port_counts = group['srcPort'].value_counts()
        src_port_proportions = src_port_counts / len(group)
        features['src_port_concentration'] = np.sum(src_port_proportions ** 2)
        
        return features
    
    @staticmethod
    def calculate_entropy_differences(group):
        """
        Calculate delta entropy between various dimensions
        """
        features = {}
        
        if len(group) == 0:
            return {
                'entropy_delta_src_dst_ip': 0,
                'entropy_delta_src_dst_port': 0,
                'entropy_balance_score': 0
            }
        
        # IP entropy difference
        src_ip_entropy = NetworkFeatureExtractor.calculate_port_entropy(group['srcAddr'].values)
        dst_ip_entropy = NetworkFeatureExtractor.calculate_port_entropy(group['dstAddr'].values)
        features['entropy_delta_src_dst_ip'] = src_ip_entropy - dst_ip_entropy
        
        # Port entropy difference
        src_port_entropy = NetworkFeatureExtractor.calculate_port_entropy(group['srcPort'].values)
        dst_port_entropy = NetworkFeatureExtractor.calculate_port_entropy(group['dstPort'].values)
        features['entropy_delta_src_dst_port'] = src_port_entropy - dst_port_entropy
        
        # Entropy balance score (closer to 0 means more balanced)
        features['entropy_balance_score'] = abs(features['entropy_delta_src_dst_ip']) + abs(features['entropy_delta_src_dst_port'])
        
        return features
    
    @staticmethod
    def calculate_spectral_entropy(group, time_span):
        """
        Calculate spectral entropy and traffic energy features
        """
        features = {}
        
        if len(group) < 2:
            return {
                'spectral_entropy': 0,
                'traffic_energy': 0,
                'traffic_burstiness': 0
            }
        
        # Get packet arrival times within window
        try:
            timestamps = pd.to_datetime(group['received'], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
            timestamps = timestamps.dropna().sort_values()
            
            if len(timestamps) < 2:
                return {
                    'spectral_entropy': 0,
                    'traffic_energy': 0,
                    'traffic_burstiness': 0
                }
            
            # Create time series of packet counts in sub-intervals
            num_bins = min(time_span, 100)  # Limit bins for efficiency
            
            # Check if timestamps have sufficient range to avoid division by zero
            timestamp_range = timestamps.iloc[-1] - timestamps.iloc[0]
            if timestamp_range.total_seconds() > 0:
                time_series, _ = np.histogram(timestamps.astype(np.int64), bins=num_bins)
            else:
                # All timestamps are identical - create single bin with all packets
                time_series = np.array([len(timestamps)])
            
            # Traffic energy (variance of packet counts)
            features['traffic_energy'] = np.var(time_series)
            
            # Burstiness (ratio of std to mean)
            mean_packets = np.mean(time_series)
            if mean_packets > 0:
                features['traffic_burstiness'] = np.std(time_series) / mean_packets
            else:
                features['traffic_burstiness'] = 0
            
            # Spectral entropy (entropy of power spectrum)
            if len(time_series) > 1:
                # Simple frequency domain representation
                fft = np.fft.fft(time_series)
                power_spectrum = np.abs(fft) ** 2
                power_spectrum = power_spectrum[power_spectrum > 0]
                
                if len(power_spectrum) > 0:
                    # Normalize to probability distribution
                    power_spectrum = power_spectrum / np.sum(power_spectrum)
                    features['spectral_entropy'] = entropy(power_spectrum, base=2)
                else:
                    features['spectral_entropy'] = 0
            else:
                features['spectral_entropy'] = 0
                
        except Exception:
            features['spectral_entropy'] = 0
            features['traffic_energy'] = 0
            features['traffic_burstiness'] = 0
        
        return features
    
    @staticmethod
    def calculate_flag_combination_entropy(flags_series):
        """
        Calculate Shannon entropy of TCP flag combinations
        """
        if len(flags_series) == 0:
            return 0
        
        # Filter out NaN values
        valid_flags = [str(x) for x in flags_series if pd.notna(x)]
        
        if len(valid_flags) == 0:
            return 0
        
        # Calculate entropy of flag combinations
        _, counts = np.unique(valid_flags, return_counts=True)
        return entropy(counts, base=2)
    
    @staticmethod
    def calculate_diversity_features(group):
        """
        Calculate pure diversity metrics (counts and ratios) for AS and Geo
        """
        features = {}
        
        if len(group) == 0:
            return {
                'as_diversity_count': 0,
                'as_diversity_ratio': 0,
                'geo_diversity_count': 0,
                'geo_diversity_ratio': 0,
                'cross_as_flow_ratio': 0,
                'cross_geo_flow_ratio': 0
            }
        
        # AS diversity
        unique_src_as = group['srcAS'].nunique()
        unique_dst_as = group['dstAS'].nunique() if 'dstAS' in group.columns else 0
        total_unique_as = len(pd.concat([group['srcAS'], group['dstAS']]).unique()) if 'dstAS' in group.columns else unique_src_as
        
        features['as_diversity_count'] = total_unique_as
        features['as_diversity_ratio'] = unique_src_as / len(group) if len(group) > 0 else 0
        
        # Geo diversity
        unique_src_geo = group['srcGeo'].nunique()
        unique_dst_geo = group['dstGeo'].nunique() if 'dstGeo' in group.columns else 0
        total_unique_geo = len(pd.concat([group['srcGeo'], group['dstGeo']]).unique()) if 'dstGeo' in group.columns else unique_src_geo
        
        features['geo_diversity_count'] = total_unique_geo
        features['geo_diversity_ratio'] = unique_src_geo / len(group) if len(group) > 0 else 0
        
        # Cross-AS and Cross-Geo flows
        if 'dstAS' in group.columns:
            cross_as_flows = group[group['srcAS'] != group['dstAS']]
            features['cross_as_flow_ratio'] = len(cross_as_flows) / len(group)
        else:
            features['cross_as_flow_ratio'] = 0
        
        if 'dstGeo' in group.columns:
            cross_geo_flows = group[group['srcGeo'] != group['dstGeo']]
            features['cross_geo_flow_ratio'] = len(cross_geo_flows) / len(group)
        else:
            features['cross_geo_flow_ratio'] = 0
        
        return features
    
    @staticmethod
    def extract_flag_features(flags_series):
        """Extract features from TCP flags"""
        flag_features = {}
        
        if len(flags_series) == 0:
            return {
                'syn_flag_ratio': 0,
                'ack_flag_ratio': 0,
                'fin_flag_ratio': 0,
                'rst_flag_ratio': 0,
                'psh_flag_ratio': 0
            }
        
        # Convert to string and handle NaN values
        valid_flags = [str(x) for x in flags_series if pd.notna(x)]
        total_flows = len(valid_flags)
        
        if total_flows == 0:
            return {
                'syn_flag_ratio': 0,
                'ack_flag_ratio': 0,
                'fin_flag_ratio': 0,
                'rst_flag_ratio': 0,
                'psh_flag_ratio': 0
            }
        
        # Count individual flags (assuming typical flag representation)
        syn_count = sum(1 for f in valid_flags if 'S' in f)
        ack_count = sum(1 for f in valid_flags if 'A' in f)
        fin_count = sum(1 for f in valid_flags if 'F' in f)
        rst_count = sum(1 for f in valid_flags if 'R' in f)
        psh_count = sum(1 for f in valid_flags if 'P' in f)
        
        flag_features['syn_flag_ratio'] = syn_count / total_flows
        flag_features['ack_flag_ratio'] = ack_count / total_flows
        flag_features['fin_flag_ratio'] = fin_count / total_flows
        flag_features['rst_flag_ratio'] = rst_count / total_flows
        flag_features['psh_flag_ratio'] = psh_count / total_flows
        
        return flag_features

    @staticmethod
    def extract_syn_flood_features(group):
        """Extract SYN flood specific features"""
        syn_flood_features = {}
        
        # SYN flood specific patterns
        tcp_flows = group[group['proto'] == 6]  # TCP flows only
        
        if len(tcp_flows) > 0:
            # SYN flag analysis
            syn_flows = tcp_flows[tcp_flows['flags'].str.contains('S', na=False)]
            syn_flood_features['syn_flood_ratio'] = len(syn_flows) / len(tcp_flows)
            
            # Half-open connection indicators
            ack_flows = tcp_flows[tcp_flows['flags'].str.contains('A', na=False)]
            syn_flood_features['syn_ack_ratio'] = len(syn_flows) / max(len(ack_flows), 1)
            
            # Small packet size indicator (SYN packets are typically small)
            small_packets = tcp_flows[tcp_flows['bytes'] <= 64]
            syn_flood_features['small_packet_ratio'] = len(small_packets) / len(tcp_flows)
            
            # Connection duration patterns (SYN floods often have very short durations)
            syn_flood_features['avg_tcp_duration'] = tcp_flows['duration'].mean()
            syn_flood_features['zero_duration_ratio'] = (tcp_flows['duration'] == 0).sum() / len(tcp_flows)
            
            # Port scanning patterns
            unique_dst_ports_per_src = tcp_flows.groupby('srcAddr')['dstPort'].nunique()
            syn_flood_features['avg_ports_per_src'] = unique_dst_ports_per_src.mean()
            syn_flood_features['max_ports_per_src'] = unique_dst_ports_per_src.max()
            
        else:
            syn_flood_features = {
                'syn_flood_ratio': 0,
                'syn_ack_ratio': 0,
                'small_packet_ratio': 0,
                'avg_tcp_duration': 0,
                'zero_duration_ratio': 0,
                'avg_ports_per_src': 0,
                'max_ports_per_src': 0
            }
        
        return syn_flood_features

    @staticmethod
    def calculate_traffic_anomaly_features(group, time_span):
        """Calculate traffic volume anomaly features"""
        anomaly_features = {}
        
        # Traffic spike detection
        total_traffic = group['bytes'].sum()
        total_packets = group['packets'].sum()
        
        # Packet rate variations
        if len(group) > 1:
            packet_rates = group['packets'] / group['duration'].replace(0, 0.001)  # Avoid division by zero
            anomaly_features['packet_rate_std'] = packet_rates.std()
            anomaly_features['packet_rate_cv'] = packet_rates.std() / max(packet_rates.mean(), 0.001)
        else:
            anomaly_features['packet_rate_std'] = 0
            anomaly_features['packet_rate_cv'] = 0
        
        # Connection establishment patterns
        anomaly_features['flows_per_second'] = len(group) / time_span
        
        return anomaly_features

    @staticmethod
    def calculate_inter_arrival_features(group):
        """Calculate inter-arrival time features for temporal analysis"""
        inter_arrival_features = {}
        
        if len(group) < 2:
            return {
                'avg_inter_arrival_time': 0,
                'std_inter_arrival_time': 0,
                'min_inter_arrival_time': 0,
                'max_inter_arrival_time': 0,
                'inter_arrival_cv': 0,
                'burst_ratio': 0,
                'periodic_pattern_score': 0
            }
        
        # Sort by received timestamp
        sorted_group = group.sort_values('received')
        timestamps = pd.to_datetime(sorted_group['received'], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        
        # Calculate inter-arrival times (in seconds)
        inter_arrivals = timestamps.diff().dt.total_seconds().dropna()
        
        if len(inter_arrivals) == 0:
            return {
                'avg_inter_arrival_time': 0,
                'std_inter_arrival_time': 0,
                'min_inter_arrival_time': 0,
                'max_inter_arrival_time': 0,
                'inter_arrival_cv': 0,
                'burst_ratio': 0,
                'periodic_pattern_score': 0
            }
        
        # Basic statistics
        inter_arrival_features['avg_inter_arrival_time'] = inter_arrivals.mean()
        inter_arrival_features['std_inter_arrival_time'] = inter_arrivals.std()
        inter_arrival_features['min_inter_arrival_time'] = inter_arrivals.min()
        inter_arrival_features['max_inter_arrival_time'] = inter_arrivals.max()
        
        # Coefficient of variation (regularity indicator)
        mean_iat = inter_arrivals.mean()
        if mean_iat > 0:
            inter_arrival_features['inter_arrival_cv'] = inter_arrivals.std() / mean_iat
        else:
            inter_arrival_features['inter_arrival_cv'] = 0
        
        # Burst detection (very short inter-arrival times)
        burst_threshold = 0.1  # 100ms
        bursts = inter_arrivals[inter_arrivals < burst_threshold]
        inter_arrival_features['burst_ratio'] = len(bursts) / len(inter_arrivals)
        
        # Periodic pattern detection (DDoS tools often have regular timing)
        # Check for common intervals (e.g., every 1s, 0.5s, 0.1s)
        common_intervals = [1.0, 0.5, 0.1, 0.01]
        max_periodic_score = 0
        
        for interval in common_intervals:
            # Count how many inter-arrivals are close to this interval (±10%)
            tolerance = interval * 0.1
            near_interval = inter_arrivals[
                (inter_arrivals >= interval - tolerance) & 
                (inter_arrivals <= interval + tolerance)
            ]
            periodic_score = len(near_interval) / len(inter_arrivals)
            max_periodic_score = max(max_periodic_score, periodic_score)
        
        inter_arrival_features['periodic_pattern_score'] = max_periodic_score
        
        return inter_arrival_features

    @staticmethod
    def calculate_connection_patterns(group):
        """Calculate connection establishment and teardown patterns"""
        connection_features = {}
        
        # Connection state analysis
        tcp_flows = group[group['proto'] == 6]
        
        if len(tcp_flows) > 0:
            # Analyze TCP flags for connection states
            syn_flows = tcp_flows[tcp_flows['flags'].str.contains('S', na=False)]
            fin_flows = tcp_flows[tcp_flows['flags'].str.contains('F', na=False)]
            rst_flows = tcp_flows[tcp_flows['flags'].str.contains('R', na=False)]
            
            # Connection establishment vs teardown ratio
            connection_features['connection_establishment_ratio'] = len(syn_flows) / len(tcp_flows)
            connection_features['connection_teardown_ratio'] = (len(fin_flows) + len(rst_flows)) / len(tcp_flows)
            
            # Incomplete connections (SYN without corresponding FIN/RST)
            if len(syn_flows) > 0:
                incomplete_ratio = (len(syn_flows) - len(fin_flows) - len(rst_flows)) / len(syn_flows)
                connection_features['incomplete_connection_ratio'] = max(0, incomplete_ratio)
            else:
                connection_features['incomplete_connection_ratio'] = 0
                
            # Connection duration anomalies
            connection_features['zero_duration_connections'] = (tcp_flows['duration'] == 0).sum() / len(tcp_flows)
            connection_features['very_short_connections'] = (tcp_flows['duration'] < 1).sum() / len(tcp_flows)
            
        else:
            connection_features = {
                'connection_establishment_ratio': 0,
                'connection_teardown_ratio': 0,
                'incomplete_connection_ratio': 0,
                'zero_duration_connections': 0,
                'very_short_connections': 0
            }
        
        return connection_features

    @staticmethod
    def calculate_flow_size_patterns(group):
        """Analyze packet and byte size distributions for attack patterns"""
        size_features = {}
        
        # Packet size analysis
        sizes = group['bytes'].values
        packet_counts = group['packets'].values
        
        if len(sizes) > 0:
            # Small packet attacks (common in DDoS)
            small_packets = sizes[sizes <= 64]  # Typical SYN packet size
            size_features['small_packet_flow_ratio'] = len(small_packets) / len(sizes)
            
            # Large packet attacks
            large_packets = sizes[sizes >= 1500]  # Near MTU size
            size_features['large_packet_flow_ratio'] = len(large_packets) / len(sizes)
            
            # Uniform size patterns (bot-generated traffic)
            unique_sizes = len(np.unique(sizes))
            size_features['size_uniformity'] = 1 - (unique_sizes / len(sizes))
            
            # Packet count patterns
            single_packet_flows = packet_counts[packet_counts == 1]
            size_features['single_packet_flow_ratio'] = len(single_packet_flows) / len(packet_counts)
            
            # Flow size entropy
            _, counts = np.unique(sizes, return_counts=True)
            size_features['flow_size_entropy'] = entropy(counts, base=2)
            
        else:
            size_features = {
                'small_packet_flow_ratio': 0,
                'large_packet_flow_ratio': 0,
                'size_uniformity': 0,
                'single_packet_flow_ratio': 0,
                'flow_size_entropy': 0
            }
        
        return size_features

    @staticmethod
    def calculate_ip_behavior_patterns(group):
        """Analyze IP address behavior patterns"""
        ip_features = {}
        
        # Source IP behavior
        src_ip_flows = group.groupby('srcAddr').size()
        
        # High-volume sources (potential attack sources)
        flow_threshold = src_ip_flows.quantile(0.95) if len(src_ip_flows) > 0 else 0
        high_volume_ips = src_ip_flows[src_ip_flows > flow_threshold]
        
        ip_features['high_volume_src_ratio'] = len(high_volume_ips) / max(len(src_ip_flows), 1)
        ip_features['max_flows_per_src'] = src_ip_flows.max() if len(src_ip_flows) > 0 else 0
        ip_features['avg_flows_per_src'] = src_ip_flows.mean() if len(src_ip_flows) > 0 else 0
        
        # IP scanning patterns
        src_port_diversity = group.groupby('srcAddr')['srcPort'].nunique()
        dst_port_diversity = group.groupby('srcAddr')['dstPort'].nunique()
        
        ip_features['avg_src_port_diversity'] = src_port_diversity.mean() if len(src_port_diversity) > 0 else 0
        ip_features['avg_dst_port_diversity'] = dst_port_diversity.mean() if len(dst_port_diversity) > 0 else 0
        
        # Port scanning indicators
        if len(dst_port_diversity) > 0:
            scanning_ips = dst_port_diversity[dst_port_diversity > 10]  # IPs targeting >10 ports
            ip_features['port_scanning_ratio'] = len(scanning_ips) / len(dst_port_diversity)
        else:
            ip_features['port_scanning_ratio'] = 0
        
        return ip_features

    def prepare_advanced_features(self, df):
        """Extract comprehensive features for network anomaly detection"""

        # Check if dataframe is empty
        if df is None or len(df) == 0:
            return []

        if self.time_span == 1:
            # 1 SECOND WINDOW - Real-time resolution time windows
            df['received'] = pd.to_datetime(df['received'], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
            nan_count = df['received'].isna().sum()
            if nan_count > 0:
                print(f"    Warning: {nan_count} out of {len(df)} timestamps failed to parse")
                df = df.dropna(subset=['received'])
            
            start_time = df['received'].min().floor(get_time_span_floor(self.time_span))
            end_time = df['received'].max().floor(get_time_span_floor(self.time_span))
            complete_time_range = pd.date_range(start=start_time, end=end_time, freq=get_time_span_frequency(self.time_span))

            df['time_window'] = df['received'].dt.floor(get_time_span_floor(self.time_span))
            time_grouped = df.groupby('time_window')
            
            existing_windows = set(time_grouped.groups.keys())
        elif self.time_span == 10:
            # 10 SECOND WINDOW - Ultra high-resolution time windows
            df['received'] = pd.to_datetime(df['received'], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
            nan_count = df['received'].isna().sum()
            if nan_count > 0:
                print(f"    Warning: {nan_count} out of {len(df)} timestamps failed to parse")
                df = df.dropna(subset=['received'])
            
            start_time = df['received'].min().floor(get_time_span_floor(self.time_span))
            end_time = df['received'].max().floor(get_time_span_floor(self.time_span))
            complete_time_range = pd.date_range(start=start_time, end=end_time, freq=get_time_span_frequency(self.time_span))

            df['time_window'] = df['received'].dt.floor(get_time_span_floor(self.time_span))
            time_grouped = df.groupby('time_window')
            
            existing_windows = set(time_grouped.groups.keys())
        elif self.time_span == 60:
            # 1 MINUTE WINDOW - Group by flow timestamps
            df['received'] = pd.to_datetime(df['received'], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
            nan_count = df['received'].isna().sum()
            if nan_count > 0:
                print(f"    Warning: {nan_count} out of {len(df)} timestamps failed to parse")
                # Remove rows with invalid timestamps to prevent issues
                df = df.dropna(subset=['received'])
            
            start_time = df['received'].min().floor('min')
            end_time = df['received'].max().floor('min')
            complete_time_range = pd.date_range(start=start_time, end=end_time, freq='1min')

            df['minute_window'] = df['received'].dt.floor('min')
            time_grouped = df.groupby('minute_window')
            
            # Create a set to track which minutes have data
            existing_windows = set(time_grouped.groups.keys())
        else:
            # 5 MINUTE WINDOW - Group by file timestamps (timezone-aware, UTC-3)
            time_grouped = df.groupby('file_timestamp')
            complete_time_range = None
            existing_windows = None
        
        features_list = []
        timestamps = []
        
        # Process existing time windows
        for timestamp, group in time_grouped:
            feature_row = {}
            timestamps.append(timestamp)
            
            # Basic flow statistics
            if self._should_include_feature('total_flows', 'basic'):
                feature_row['total_flows'] = len(group)
            if self._should_include_feature('total_packets', 'basic'):
                feature_row['total_packets'] = group['packets'].sum()
            if self._should_include_feature('total_bytes', 'basic'):
                feature_row['total_bytes'] = group['bytes'].sum()
            if self._should_include_feature('avg_duration', 'basic'):
                feature_row['avg_duration'] = group['duration'].mean()
            
            # Traffic rate features
            if self._should_include_feature('packet_rate', 'traffic_rates'):
                feature_row['packet_rate'] = feature_row.get('total_packets', group['packets'].sum()) / self.time_span
            if self._should_include_feature('bit_rate', 'traffic_rates'):
                feature_row['bit_rate'] = (feature_row.get('total_bytes', group['bytes'].sum()) * 8) / self.time_span
            if self._should_include_feature('flow_rate', 'traffic_rates'):
                feature_row['flow_rate'] = feature_row.get('total_flows', len(group)) / self.time_span
            
            # Packet size statistics
            total_packets = feature_row.get('total_packets', group['packets'].sum())
            total_bytes = feature_row.get('total_bytes', group['bytes'].sum())
            
            if self._should_include_feature('avg_packet_size', 'traffic_rates'):
                if total_packets > 0:
                    feature_row['avg_packet_size'] = total_bytes / total_packets
                else:
                    feature_row['avg_packet_size'] = 0
            if self._should_include_feature('packets_per_flow', 'traffic_rates'):
                feature_row['packets_per_flow'] = group['packets'].mean()
            if self._should_include_feature('bytes_per_flow', 'traffic_rates'):
                feature_row['bytes_per_flow'] = group['bytes'].mean()
            
            # Port entropy
            if self._should_include_feature('src_port_entropy', 'entropy'):
                feature_row['src_port_entropy'] = self.calculate_port_entropy(group['srcPort'].values)
            if self._should_include_feature('dst_port_entropy', 'entropy'):
                feature_row['dst_port_entropy'] = self.calculate_port_entropy(group['dstPort'].values)
            
            # Protocol distribution
            if any(self._should_include_feature(f, 'protocol') for f in ['tcp_ratio', 'udp_ratio', 'icmp_ratio']):
                proto_counts = group['proto'].value_counts()
                if self._should_include_feature('tcp_ratio', 'protocol'):
                    feature_row['tcp_ratio'] = proto_counts.get(6, 0) / len(group)
                if self._should_include_feature('udp_ratio', 'protocol'):
                    feature_row['udp_ratio'] = proto_counts.get(17, 0) / len(group)
                if self._should_include_feature('icmp_ratio', 'protocol'):
                    feature_row['icmp_ratio'] = proto_counts.get(1, 0) / len(group)
            
            # IP diversity
            if self._should_include_feature('unique_src_ips', 'ip_diversity'):
                feature_row['unique_src_ips'] = group['srcAddr'].nunique()
            if self._should_include_feature('src_ip_entropy', 'entropy'):
                feature_row['src_ip_entropy'] = self.calculate_port_entropy(group['srcAddr'].values)
            
            # AS (Autonomous System) features
            if self._should_include_feature('unique_src_as', 'ip_diversity'):
                feature_row['unique_src_as'] = group['srcAS'].nunique()
            if self._should_include_feature('src_as_entropy', 'entropy'):
                feature_row['src_as_entropy'] = self.calculate_as_entropy(group['srcAS'].values)
            
            # AS diversity ratio
            if self._should_include_feature('as_diversity_ratio', 'ip_diversity'):
                unique_src_as = feature_row.get('unique_src_as', group['srcAS'].nunique())
                unique_src_ips = feature_row.get('unique_src_ips', group['srcAddr'].nunique())
                feature_row['as_diversity_ratio'] = unique_src_as / max(unique_src_ips, 1)
            
            # Geographic features
            if self._should_include_feature('unique_src_geo', 'ip_diversity'):
                feature_row['unique_src_geo'] = group['srcGeo'].nunique()
            if self._should_include_feature('src_geo_entropy', 'entropy'):
                feature_row['src_geo_entropy'] = self.calculate_geo_entropy(group['srcGeo'].values)
            
            # Cross-border traffic ratio
            if self._should_include_feature('cross_border_ratio', 'ip_diversity'):
                cross_border = group[group['srcGeo'] != group['dstGeo']]
                feature_row['cross_border_ratio'] = len(cross_border) / len(group) if len(group) > 0 else 0
            
            # TCP Flag features
            if any(self._should_include_feature(f, 'tcp_flags') for f in ['syn_flag_ratio', 'ack_flag_ratio', 'fin_flag_ratio', 'rst_flag_ratio', 'psh_flag_ratio']):
                flag_features = self.extract_flag_features(group['flags'])
                for flag_feature, value in flag_features.items():
                    if self._should_include_feature(flag_feature, 'tcp_flags'):
                        feature_row[flag_feature] = value
            
            # SYN flood specific features
            if any(self._should_include_feature(f, 'syn_flood_specific') for f in ['syn_flood_ratio', 'syn_ack_ratio', 'small_packet_ratio', 'avg_tcp_duration', 'zero_duration_ratio', 'avg_ports_per_src', 'max_ports_per_src']):
                syn_flood_features = self.extract_syn_flood_features(group)
                for sf_feature, value in syn_flood_features.items():
                    if self._should_include_feature(sf_feature, 'syn_flood_specific'):
                        feature_row[sf_feature] = value
            
            # Traffic anomaly features
            if any(self._should_include_feature(f, 'traffic_anomaly') for f in ['packet_rate_std', 'packet_rate_cv', 'flows_per_second']):
                traffic_anomaly_features = self.calculate_traffic_anomaly_features(group, self.time_span)
                for ta_feature, value in traffic_anomaly_features.items():
                    if self._should_include_feature(ta_feature, 'traffic_anomaly'):
                        feature_row[ta_feature] = value
            
            # Inter-arrival time features
            if any(self._should_include_feature(f, 'inter_arrival') for f in ['avg_inter_arrival_time', 'std_inter_arrival_time', 'min_inter_arrival_time', 'max_inter_arrival_time', 'inter_arrival_cv', 'burst_ratio', 'periodic_pattern_score']):
                inter_arrival_features = self.calculate_inter_arrival_features(group)
                for ia_feature, value in inter_arrival_features.items():
                    if self._should_include_feature(ia_feature, 'inter_arrival'):
                        feature_row[ia_feature] = value
            
            # Connection pattern features
            if any(self._should_include_feature(f, 'connection_patterns') for f in ['connection_establishment_ratio', 'connection_teardown_ratio', 'incomplete_connection_ratio', 'zero_duration_connections', 'very_short_connections']):
                connection_features = self.calculate_connection_patterns(group)
                for conn_feature, value in connection_features.items():
                    if self._should_include_feature(conn_feature, 'connection_patterns'):
                        feature_row[conn_feature] = value
            
            # Flow size pattern features
            if any(self._should_include_feature(f, 'flow_patterns') for f in ['small_packet_flow_ratio', 'large_packet_flow_ratio', 'size_uniformity', 'single_packet_flow_ratio', 'flow_size_entropy']):
                size_features = self.calculate_flow_size_patterns(group)
                for size_feature, value in size_features.items():
                    if self._should_include_feature(size_feature, 'flow_patterns'):
                        feature_row[size_feature] = value
            
            # IP behavior pattern features
            if any(self._should_include_feature(f, 'ip_behavior') for f in ['high_volume_src_ratio', 'max_flows_per_src', 'avg_flows_per_src', 'avg_src_port_diversity', 'avg_dst_port_diversity', 'port_scanning_ratio']):
                ip_behavior_features = self.calculate_ip_behavior_patterns(group)
                for ip_feature, value in ip_behavior_features.items():
                    if self._should_include_feature(ip_feature, 'ip_behavior'):
                        feature_row[ip_feature] = value
            
            # Connection patterns
            if self._should_include_feature('avg_src_ports_per_ip', 'connection_diversity'):
                feature_row['avg_src_ports_per_ip'] = group.groupby('srcAddr')['srcPort'].nunique().mean()
            if self._should_include_feature('avg_dst_ports_per_ip', 'connection_diversity'):
                feature_row['avg_dst_ports_per_ip'] = group.groupby('dstAddr')['dstPort'].nunique().mean()
            
            # Traffic volume distribution
            if any(self._should_include_feature(f, 'flow_patterns') for f in ['max_bytes_per_flow', 'std_bytes_per_flow', 'max_packets_per_flow', 'std_packets_per_flow']):
                if self._should_include_feature('max_bytes_per_flow', 'flow_patterns'):
                    feature_row['max_bytes_per_flow'] = group['bytes'].max()
                if self._should_include_feature('std_bytes_per_flow', 'flow_patterns'):
                    feature_row['std_bytes_per_flow'] = group['bytes'].std()
                if self._should_include_feature('max_packets_per_flow', 'flow_patterns'):
                    feature_row['max_packets_per_flow'] = group['packets'].max()
                if self._should_include_feature('std_packets_per_flow', 'flow_patterns'):
                    feature_row['std_packets_per_flow'] = group['packets'].std()
            
            # NEW ADVANCED FEATURES
            
            # Statistical features for bytes, packets, and duration
            if any(self._should_include_feature(f, 'statistical') for f in ['bytes_mean', 'bytes_variance', 'bytes_skewness', 'bytes_kurtosis']):
                bytes_stats = self.calculate_statistical_features(group['bytes'], prefix='bytes_')
                for stat_name, stat_value in bytes_stats.items():
                    if self._should_include_feature(stat_name, 'statistical'):
                        feature_row[stat_name] = stat_value
            
            if any(self._should_include_feature(f, 'statistical') for f in ['packets_mean', 'packets_variance', 'packets_skewness', 'packets_kurtosis']):
                packets_stats = self.calculate_statistical_features(group['packets'], prefix='packets_')
                for stat_name, stat_value in packets_stats.items():
                    if self._should_include_feature(stat_name, 'statistical'):
                        feature_row[stat_name] = stat_value
            
            if any(self._should_include_feature(f, 'statistical') for f in ['duration_mean', 'duration_variance', 'duration_skewness', 'duration_kurtosis']):
                duration_stats = self.calculate_statistical_features(group['duration'], prefix='duration_')
                for stat_name, stat_value in duration_stats.items():
                    if self._should_include_feature(stat_name, 'statistical'):
                        feature_row[stat_name] = stat_value
            
            # Simpson's and Gini indices
            if self._should_include_feature('src_ip_simpson_index', 'diversity_indices'):
                feature_row['src_ip_simpson_index'] = self.calculate_simpson_index(group['srcAddr'].values)
            if self._should_include_feature('dst_ip_simpson_index', 'diversity_indices'):
                feature_row['dst_ip_simpson_index'] = self.calculate_simpson_index(group['dstAddr'].values)
            if self._should_include_feature('src_port_simpson_index', 'diversity_indices'):
                feature_row['src_port_simpson_index'] = self.calculate_simpson_index(group['srcPort'].values)
            if self._should_include_feature('dst_port_simpson_index', 'diversity_indices'):
                feature_row['dst_port_simpson_index'] = self.calculate_simpson_index(group['dstPort'].values)
            
            if self._should_include_feature('bytes_gini_index', 'diversity_indices'):
                feature_row['bytes_gini_index'] = self.calculate_gini_index(group['bytes'].values)
            if self._should_include_feature('packets_gini_index', 'diversity_indices'):
                feature_row['packets_gini_index'] = self.calculate_gini_index(group['packets'].values)
            
            # Uniqueness features for IPs and ports
            if any(self._should_include_feature(f, 'uniqueness') for f in ['src_ip_unique_count', 'src_ip_uniqueness_ratio', 'src_ip_collision_probability']):
                src_ip_uniqueness = self.calculate_uniqueness_features(group['srcAddr'].values, prefix='src_ip_')
                for uniq_name, uniq_value in src_ip_uniqueness.items():
                    if self._should_include_feature(uniq_name, 'uniqueness'):
                        feature_row[uniq_name] = uniq_value
            
            if any(self._should_include_feature(f, 'uniqueness') for f in ['dst_ip_unique_count', 'dst_ip_uniqueness_ratio', 'dst_ip_collision_probability']):
                dst_ip_uniqueness = self.calculate_uniqueness_features(group['dstAddr'].values, prefix='dst_ip_')
                for uniq_name, uniq_value in dst_ip_uniqueness.items():
                    if self._should_include_feature(uniq_name, 'uniqueness'):
                        feature_row[uniq_name] = uniq_value
            
            # Fan-in/Fan-out features
            if any(self._should_include_feature(f, 'fan_in_out') for f in ['max_fan_in', 'avg_fan_in', 'max_fan_out', 'avg_fan_out', 'fan_in_out_ratio']):
                fan_features = self.calculate_fan_in_out_features(group)
                for fan_name, fan_value in fan_features.items():
                    if self._should_include_feature(fan_name, 'fan_in_out'):
                        feature_row[fan_name] = fan_value
            
            # Asymmetry features
            if any(self._should_include_feature(f, 'asymmetry') for f in ['src_dst_port_entropy_diff', 'src_dst_ip_entropy_diff', 'port_entropy_abs_diff', 'ip_entropy_abs_diff']):
                asymmetry_features = self.calculate_asymmetry_features(group)
                for asym_name, asym_value in asymmetry_features.items():
                    if self._should_include_feature(asym_name, 'asymmetry'):
                        feature_row[asym_name] = asym_value
            
            # Amplification attack features
            if any(self._should_include_feature(f, 'amplification') for f in ['avg_request_reply_ratio', 'small_request_large_reply_ratio', 'amplification_score', 'udp_amplification_potential']):
                amplification_features = self.calculate_amplification_features(group)
                for amp_name, amp_value in amplification_features.items():
                    if self._should_include_feature(amp_name, 'amplification'):
                        feature_row[amp_name] = amp_value
            
            # Top-K concentration features for destinations
            if any(self._should_include_feature(f, 'concentration') for f in ['dst_top_1_ratio', 'dst_top_5_ratio', 'dst_top_10_ratio', 'dst_heavy_tail_index', 'dst_concentration_score']):
                dst_concentration = self.calculate_concentration_features(group['dstAddr'].values, prefix='dst_')
                for conc_name, conc_value in dst_concentration.items():
                    if self._should_include_feature(conc_name, 'concentration'):
                        feature_row[conc_name] = conc_value
            
            # Top-K concentration for ports
            if any(self._should_include_feature(f, 'concentration') for f in ['dst_port_top_1_ratio', 'dst_port_top_5_ratio', 'dst_port_top_10_ratio']):
                dst_port_concentration = self.calculate_concentration_features(group['dstPort'].values, prefix='dst_port_')
                for conc_name, conc_value in dst_port_concentration.items():
                    if self._should_include_feature(conc_name, 'concentration'):
                        feature_row[conc_name] = conc_value
            
            # Port usage features
            if any(self._should_include_feature(f, 'port_usage') for f in ['ephemeral_port_ratio', 'well_known_port_ratio', 'dst_port_concentration', 'src_port_concentration']):
                port_usage_features = self.calculate_port_usage_features(group)
                for port_name, port_value in port_usage_features.items():
                    if self._should_include_feature(port_name, 'port_usage'):
                        feature_row[port_name] = port_value
            
            # Entropy differences
            if any(self._should_include_feature(f, 'entropy_diff') for f in ['entropy_delta_src_dst_ip', 'entropy_delta_src_dst_port', 'entropy_balance_score']):
                entropy_diff_features = self.calculate_entropy_differences(group)
                for ent_diff_name, ent_diff_value in entropy_diff_features.items():
                    if self._should_include_feature(ent_diff_name, 'entropy_diff'):
                        feature_row[ent_diff_name] = ent_diff_value
            
            # Spectral entropy and traffic energy
            if any(self._should_include_feature(f, 'spectral') for f in ['spectral_entropy', 'traffic_energy', 'traffic_burstiness']):
                spectral_features = self.calculate_spectral_entropy(group, self.time_span)
                for spec_name, spec_value in spectral_features.items():
                    if self._should_include_feature(spec_name, 'spectral'):
                        feature_row[spec_name] = spec_value
            
            # Flag combination entropy
            if self._should_include_feature('flag_combination_entropy', 'entropy'):
                feature_row['flag_combination_entropy'] = self.calculate_flag_combination_entropy(group['flags'])
            
            # Pure diversity features
            if any(self._should_include_feature(f, 'pure_diversity') for f in ['as_diversity_count', 'as_diversity_ratio', 'geo_diversity_count', 'geo_diversity_ratio']):
                diversity_features = self.calculate_diversity_features(group)
                for div_name, div_value in diversity_features.items():
                    if self._should_include_feature(div_name, 'pure_diversity'):
                        feature_row[div_name] = div_value
            
            # Inter-arrival time variance (extend existing inter-arrival features)
            if self._should_include_feature('inter_arrival_variance', 'inter_arrival'):
                if 'std_inter_arrival_time' in feature_row:
                    feature_row['inter_arrival_variance'] = feature_row['std_inter_arrival_time'] ** 2
                else:
                    inter_arrival_features = self.calculate_inter_arrival_features(group)
                    if 'std_inter_arrival_time' in inter_arrival_features:
                        feature_row['inter_arrival_variance'] = inter_arrival_features['std_inter_arrival_time'] ** 2
            
            features_list.append(feature_row)
        
        # Fill missing time windows with zero values for 1-second, 10-second and 1-minute windows
        if self.time_span in [1, 10, 60] and complete_time_range is not None:
            missing_windows = [t for t in complete_time_range if t not in existing_windows]
            
            if missing_windows:
                time_desc = get_time_span_detailed_description(self.time_span)
                # print(f"    Filling {len(missing_windows)} missing {time_desc} windows with zeros")
                
                # Create zero feature template from the first feature row if available
                zero_feature_template = {}
                if features_list:
                    zero_feature_template = {k: 0 for k in features_list[0].keys()}
                
                # Add zero rows for missing time windows (avoid duplicates)
                existing_timestamps = set(timestamps)
                for missing_window in missing_windows:
                    if missing_window not in existing_timestamps:
                        zero_row = zero_feature_template.copy()
                        features_list.append(zero_row)
                        timestamps.append(missing_window)
        
        features_df = pd.DataFrame(features_list)
        features_df['timestamp'] = timestamps
        
        # Sort by timestamp to ensure proper chronological order
        features_df = features_df.sort_values('timestamp').reset_index(drop=True)
        
        # Apply feature filtering based on configuration
        features_df = self._filter_features(features_df)
        
        # Apply EMA smoothing to create new EMA features
        features_df = self._apply_ema_if_configured(features_df)
        
        # Feature scaling and outlier handling
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_cols] = features_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0)
        
        return features_df

    def process_datasets(self, datasets):
        """
        Process all datasets and extract features (backward compatibility)
        Use extract_features_to_csv() for memory-efficient processing
        """
        
        # Try to load from cache first
        if self.use_cache and self.cache:
            datasets_hash = DataCache.hash_dataframes(datasets)
            cached_features = self.cache.get_features_cache(datasets_hash, self.time_span)
            if cached_features is not None:
                print("Using cached features!")
                return cached_features
        
        print("Extracting features from datasets...")
        features_dict = {}
        
        for split_name, data in datasets.items():
            print(f"Processing {split_name} dataset...")
            features_dict[split_name] = self.prepare_advanced_features(data)
        
        # Save to cache for next time
        if self.use_cache and self.cache:
            datasets_hash = DataCache.hash_dataframes(datasets)
            self.cache.save_features_cache(features_dict, datasets_hash, self.time_span)
        
        return features_dict

    def prepare_training_data(self, features_dict):
        """Prepare and clean feature matrices for training"""
        
        # Try to load from cache first
        if self.use_cache and self.cache:
            features_hash = DataCache.hash_dataframes(features_dict)
            cached_processed = self.cache.get_processed_features_cache(features_hash)
            if cached_processed is not None:
                print("Using cached processed features!")
                return cached_processed
        
        print("Processing features for training...")
        processed_features = {}
        
        for split_name, features_df in features_dict.items():
            feature_cols = [col for col in features_df.columns if col != 'timestamp']
            training_features = features_df[feature_cols].copy()
            training_features = training_features.replace([np.inf, -np.inf], np.nan).fillna(0)
            processed_features[split_name] = {
                'features': training_features,
                'timestamps': features_df['timestamp']
            }
        
        # Save to cache for next time
        if self.use_cache and self.cache:
            features_hash = DataCache.hash_dataframes(features_dict)
            self.cache.save_processed_features_cache(processed_features, features_hash)
        
        return processed_features
