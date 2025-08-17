import numpy as np
import pandas as pd
from scipy.stats import entropy


class NetworkFeatureExtractor:
    def __init__(self, time_span=10):
        """
        Initialize NetworkFeatureExtractor
        
        Args:
            time_span (int): Time window in seconds for rate calculations (default: 10 for real-time processing)
        """
        self.time_span = time_span
    
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
                'psh_flag_ratio': 0,
                'urg_flag_ratio': 0,
                'unique_flag_combinations': 0
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
                'psh_flag_ratio': 0,
                'urg_flag_ratio': 0,
                'unique_flag_combinations': 0
            }
        
        # Count individual flags (assuming typical flag representation)
        syn_count = sum(1 for f in valid_flags if 'S' in f)
        ack_count = sum(1 for f in valid_flags if 'A' in f)
        fin_count = sum(1 for f in valid_flags if 'F' in f)
        rst_count = sum(1 for f in valid_flags if 'R' in f)
        psh_count = sum(1 for f in valid_flags if 'P' in f)
        urg_count = sum(1 for f in valid_flags if 'U' in f)
        
        flag_features['syn_flag_ratio'] = syn_count / total_flows
        flag_features['ack_flag_ratio'] = ack_count / total_flows
        flag_features['fin_flag_ratio'] = fin_count / total_flows
        flag_features['rst_flag_ratio'] = rst_count / total_flows
        flag_features['psh_flag_ratio'] = psh_count / total_flows
        flag_features['urg_flag_ratio'] = urg_count / total_flows
        flag_features['unique_flag_combinations'] = len(set(valid_flags))
        
        return flag_features

    def prepare_advanced_features(self, df):
        """Extract comprehensive features for network anomaly detection"""
        time_grouped = df.groupby('file_timestamp')
        
        features_list = []
        timestamps = []
        
        for timestamp, group in time_grouped:
            feature_row = {}
            timestamps.append(timestamp)
            
            # Basic flow statistics
            feature_row['total_flows'] = len(group)
            feature_row['total_packets'] = group['packets'].sum()
            feature_row['total_bytes'] = group['bytes'].sum()
            feature_row['avg_duration'] = group['duration'].mean()
            
            # Traffic rate features
            feature_row['packet_rate'] = feature_row['total_packets'] / self.time_span
            feature_row['bit_rate'] = (feature_row['total_bytes'] * 8) / self.time_span
            feature_row['flow_rate'] = feature_row['total_flows'] / self.time_span
            
            # Packet size statistics
            feature_row['avg_packet_size'] = group['bytes'].sum() / group['packets'].sum()
            feature_row['packets_per_flow'] = group['packets'].mean()
            feature_row['bytes_per_flow'] = group['bytes'].mean()
            
            # Port entropy
            feature_row['src_port_entropy'] = self.calculate_port_entropy(group['srcPort'].values)
            feature_row['dst_port_entropy'] = self.calculate_port_entropy(group['dstPort'].values)
            
            # Protocol distribution
            proto_counts = group['proto'].value_counts()
            feature_row['tcp_ratio'] = proto_counts.get(6, 0) / len(group)
            feature_row['udp_ratio'] = proto_counts.get(17, 0) / len(group)
            feature_row['icmp_ratio'] = proto_counts.get(1, 0) / len(group)
            
            # IP diversity
            feature_row['unique_src_ips'] = group['srcAddr'].nunique()
            feature_row['unique_dst_ips'] = group['dstAddr'].nunique()
            feature_row['src_ip_entropy'] = self.calculate_port_entropy(group['srcAddr'].values)
            feature_row['dst_ip_entropy'] = self.calculate_port_entropy(group['dstAddr'].values)
            
            # AS (Autonomous System) features
            feature_row['unique_src_as'] = group['srcAS'].nunique()
            feature_row['unique_dst_as'] = group['dstAS'].nunique()
            feature_row['src_as_entropy'] = self.calculate_as_entropy(group['srcAS'].values)
            feature_row['dst_as_entropy'] = self.calculate_as_entropy(group['dstAS'].values)
            
            # AS diversity ratio
            feature_row['as_diversity_ratio'] = feature_row['unique_src_as'] / max(feature_row['unique_src_ips'], 1)
            
            # Geographic features
            feature_row['unique_src_geo'] = group['srcGeo'].nunique()
            feature_row['unique_dst_geo'] = group['dstGeo'].nunique()
            feature_row['src_geo_entropy'] = self.calculate_geo_entropy(group['srcGeo'].values)
            feature_row['dst_geo_entropy'] = self.calculate_geo_entropy(group['dstGeo'].values)
            
            # Cross-border traffic ratio
            cross_border = group[group['srcGeo'] != group['dstGeo']]
            feature_row['cross_border_ratio'] = len(cross_border) / len(group) if len(group) > 0 else 0
            
            # TCP Flag features
            flag_features = self.extract_flag_features(group['flags'])
            feature_row.update(flag_features)
            
            # Connection patterns
            feature_row['avg_src_ports_per_ip'] = group.groupby('srcAddr')['srcPort'].nunique().mean()
            feature_row['avg_dst_ports_per_ip'] = group.groupby('dstAddr')['dstPort'].nunique().mean()
            
            # Traffic volume distribution
            feature_row['max_bytes_per_flow'] = group['bytes'].max()
            feature_row['std_bytes_per_flow'] = group['bytes'].std()
            feature_row['max_packets_per_flow'] = group['packets'].max()
            feature_row['std_packets_per_flow'] = group['packets'].std()
            
            features_list.append(feature_row)
        
        features_df = pd.DataFrame(features_list)
        features_df['timestamp'] = timestamps
        
        return features_df

    def process_datasets(self, datasets):
        """Process all datasets and extract features"""
        features_dict = {}
        
        for split_name, data in datasets.items():
            features_dict[split_name] = self.prepare_advanced_features(data)
        
        return features_dict

    def prepare_training_data(self, features_dict):
        """Prepare and clean feature matrices for training"""
        processed_features = {}
        
        for split_name, features_df in features_dict.items():
            feature_cols = [col for col in features_df.columns if col != 'timestamp']
            training_features = features_df[feature_cols].copy()
            training_features = training_features.replace([np.inf, -np.inf], np.nan).fillna(0)
            processed_features[split_name] = {
                'features': training_features,
                'timestamps': features_df['timestamp']
            }
        
        return processed_features
