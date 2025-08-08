#!/usr/bin/env python3
"""
Feature Engineering module for DDoS Detection System
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class AdvancedFeatureEngineer:
    """Enhanced feature engineering for DDoS detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_features(self, df):
        """Extract comprehensive features from netflow data"""
        # Basic aggregations
        agg = df.groupby('ts_bin').agg({
            'bytes': 'sum',
            'packets': 'sum', 
            'flows': 'sum',
            'srcAddr': 'nunique',
            'dstAddr': 'nunique',
            'srcPort': 'nunique',
            'dstPort': 'nunique',
            'proto': 'nunique',
            'duration': ['mean', 'std', 'min', 'max']
        })
        
        agg.columns = ['_'.join(col).strip() for col in agg.columns]
        
        # Advanced features
        agg['bytes_per_flow'] = agg['bytes_sum'] / agg['flows_sum']
        agg['pkts_per_flow'] = agg['packets_sum'] / agg['flows_sum']
        agg['bytes_per_pkt'] = agg['bytes_sum'] / agg['packets_sum']
        
        # Rate features (per second)
        # TODO: adapt to others aggregate time periods
        agg['flow_rate'] = agg['flows_sum'] / 60
        agg['byte_rate'] = agg['bytes_sum'] / 60
        agg['pkt_rate'] = agg['packets_sum'] / 60
        
        # Diversity ratios
        agg['src_dst_ratio'] = agg['srcAddr_nunique'] / agg['dstAddr_nunique']
        agg['port_ratio'] = agg['srcPort_nunique'] / agg['dstPort_nunique']
        agg['proto_diversity'] = agg['proto_nunique']
        
        # Entropy features
        agg['addr_entropy'] = self._calculate_entropy(df, 'srcAddr')
        agg['port_entropy'] = self._calculate_entropy(df, 'srcPort')
        
        agg = agg.replace([np.inf, -np.inf], np.nan)
        agg = agg.fillna(0)
        
        return agg.reset_index()
    
    def _calculate_entropy(self, df, column):
        """Calculate entropy for categorical columns"""
        entropy = df.groupby('ts_bin')[column].apply(
            lambda x: -np.sum((x.value_counts() / len(x)) * np.log2(x.value_counts() / len(x) + 1e-10))
        )
        return entropy
    
    def prepare_features(self, data, fit_scaler=True):
        """Prepare features for training"""
        # Exclude non-numerical columns
        exclude_cols = ['ts_bin', 'attack_type']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Only use numerical columns
        numerical_data = data[feature_cols].select_dtypes(include=[np.number])
        feature_cols = numerical_data.columns.tolist()
        X = numerical_data.values
        
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, feature_cols
