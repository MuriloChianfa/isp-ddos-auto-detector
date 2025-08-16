import numpy as np
import pandas as pd
from scipy.stats import entropy


class NetworkFeatureExtractor:
    @staticmethod
    def calculate_port_entropy(ports):
        """Calculate entropy of port distribution"""
        _, counts = np.unique(ports, return_counts=True)
        return entropy(counts, base=2)

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
            time_span = 300
            feature_row['packet_rate'] = feature_row['total_packets'] / time_span
            feature_row['bit_rate'] = (feature_row['total_bytes'] * 8) / time_span
            feature_row['flow_rate'] = feature_row['total_flows'] / time_span
            
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
