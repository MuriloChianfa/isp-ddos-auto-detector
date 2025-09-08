import numpy as np
import pandas as pd
import os
from scipy.stats import entropy
from .cache import DataCache
from multiprocessing import Pool, cpu_count
import functools


class NetworkFeatureExtractor:
    def __init__(self, time_span=300, use_cache=True, dataset_name=None, max_processes=None, feature_config=None):
        """
        Initialize NetworkFeatureExtractor
        
        Args:
            time_span (int): Time window in seconds for aggregation and rate calculations 
                           - 60 for 1-minute windows (uses firstSeen timestamps)
                           - 300 for 5-minute windows (uses file_timestamp) [default]
            use_cache (bool): Whether to use caching for feature extraction
            dataset_name (str): Name of the dataset being processed (for result organization)
            max_processes (int): Maximum number of processes to use for parallel processing.
            feature_config (dict): Configuration for feature selection and customization
        """
        self.time_span = time_span
        self.use_cache = use_cache
        self.dataset_name = dataset_name
        self.max_processes = max_processes
        self.feature_config = feature_config or {}
        self.cache = DataCache() if use_cache else None
    
    def _get_features_csv_path(self, split_name):
        """Get the path for features CSV file"""
        base_dir = f"./datasets/{self.dataset_name}/features"
        os.makedirs(base_dir, exist_ok=True)
        
        # Include feature config hash in filename to ensure cache invalidation when config changes
        config_suffix = ""
        if self.feature_config:
            import hashlib
            config_str = str(sorted(self.feature_config.items()))
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
            config_suffix = f"_cfg{config_hash}"
        
        return os.path.join(base_dir, f"{split_name}_features_{self.time_span}s{config_suffix}.csv")
    
    def _should_include_feature(self, feature_name, group_name=None):
        """Check if a feature should be included based on configuration"""
        if not self.feature_config:
            return True
            
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
        """Filter features based on dataset configuration"""
        if not self.feature_config:
            return features_df
            
        from .constants import FEATURE_GROUPS
        
        # Determine which features to keep
        features_to_keep = ['timestamp']  # Always keep timestamp
        
        include_groups = self.feature_config.get('include_groups', [])
        exclude_features = self.feature_config.get('exclude_features', [])
        
        if include_groups:
            # Include features from specified groups
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
        
        print(f"  Feature selection: {len(available_features)-1} features selected from {len(features_df.columns)-1} available")
        if include_groups:
            print(f"  Included groups: {', '.join(include_groups)}")
        if exclude_features:
            print(f"  Excluded features: {', '.join(exclude_features)}")
            
        return filtered_df
    
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
            
            print(f"Extracting features for {split_name} split...")
            
            # Show time window information once
            if self.time_span == 60:
                print(f"  Using 1-minute time windows (60 seconds)")
            else:
                print(f"  Using 5-minute time windows ({self.time_span} seconds)")
            
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
                    num_processes = min(cpu_count(), len(data_chunks), 16)

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
                    if i % 10 == 0 and i > 0:
                        print(f"  Processing chunk {i}/{len(data_chunks)} for {split_name}...")
                    
                    chunk_features = self._extract_features_from_chunk(self.time_span, chunk, feature_config=self.feature_config)
                    if chunk_features is not None:
                        all_features.append(chunk_features)
            
            if all_features:
                # Combine all features
                combined_features = pd.concat(all_features, ignore_index=True)
                
                # Save to CSV
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
            temp_extractor = NetworkFeatureExtractor(time_span=time_span, use_cache=False, feature_config=feature_config)
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
        
        # Sort by firstSeen timestamp
        sorted_group = group.sort_values('firstSeen')
        timestamps = pd.to_datetime(sorted_group['firstSeen'], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        
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

        if self.time_span == 60:
            # 1 MINUTE WINDOW - Group by flow timestamps
            df['firstSeen'] = pd.to_datetime(df['firstSeen'], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
            nan_count = df['firstSeen'].isna().sum()
            if nan_count > 0:
                print(f"    Warning: {nan_count} out of {len(df)} timestamps failed to parse")
            df['minute_window'] = df['firstSeen'].dt.floor('min')
            time_grouped = df.groupby('minute_window')
        else:
            # 5 MINUTE WINDOW - Group by file timestamps
            time_grouped = df.groupby('file_timestamp')
        
        features_list = []
        timestamps = []
        
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
            
            features_list.append(feature_row)
        
        features_df = pd.DataFrame(features_list)
        features_df['timestamp'] = timestamps
        
        # Apply feature filtering based on configuration
        features_df = self._filter_features(features_df)
        
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
