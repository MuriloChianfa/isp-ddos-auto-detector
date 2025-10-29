"""
Constants and feature group definitions for the ISP DDoS Auto Detector
"""

# Time span configurations
TIME_SPANS = {
    1: {
        'description': '1-second',
        'detailed_description': 'real-time resolution',
        'frequency': '1s',
        'pandas_floor': '1s'
    },
    10: {
        'description': '10-second',
        'detailed_description': 'ultra high-resolution',
        'frequency': '10s',
        'pandas_floor': '10s'
    },
    60: {
        'description': '1-minute', 
        'detailed_description': 'high-resolution',
        'frequency': '1min',
        'pandas_floor': 'min'
    },
    300: {
        'description': '5-minute',
        'detailed_description': 'standard resolution', 
        'frequency': '5min',
        'pandas_floor': '5min'
    }
}

# Supported time spans
SUPPORTED_TIME_SPANS = list(TIME_SPANS.keys())

# Feature group definitions
FEATURE_GROUPS = {
    'basic': [
        'total_flows', 'total_packets', 'total_bytes', 'avg_duration'
    ],
    'traffic_rates': [
        'packet_rate', 'bit_rate', 'flow_rate', 'avg_packet_size', 
        'packets_per_flow', 'bytes_per_flow'
    ],
    'entropy': [
        'src_port_entropy', 'dst_port_entropy', 'src_ip_entropy', 
        'src_as_entropy', 'src_geo_entropy', 'flow_size_entropy',
        'flag_combination_entropy'
    ],
    'protocol': [
        'tcp_ratio', 'udp_ratio', 'icmp_ratio'
    ],
    'ip_diversity': [
        'unique_src_ips', 'unique_src_as', 'unique_src_geo',
        'as_diversity_ratio', 'cross_border_ratio'
    ],
    'tcp_flags': [
        'syn_flag_ratio', 'ack_flag_ratio', 'fin_flag_ratio', 
        'rst_flag_ratio', 'psh_flag_ratio'
    ],
    'syn_flood_specific': [
        'syn_flood_ratio', 'syn_ack_ratio', 'small_packet_ratio',
        'avg_tcp_duration', 'zero_duration_ratio', 'avg_ports_per_src',
        'max_ports_per_src'
    ],
    'http_flood_specific': [
        'large_packet_flow_ratio', 'size_uniformity', 'avg_dst_port_diversity'
    ],
    'connection_patterns': [
        'connection_establishment_ratio', 'connection_teardown_ratio',
        'incomplete_connection_ratio', 'zero_duration_connections',
        'very_short_connections'
    ],
    'inter_arrival': [
        'avg_inter_arrival_time', 'std_inter_arrival_time',
        'min_inter_arrival_time', 'max_inter_arrival_time',
        'inter_arrival_cv', 'burst_ratio', 'periodic_pattern_score',
        'inter_arrival_variance'
    ],
    'flow_patterns': [
        'small_packet_flow_ratio', 'single_packet_flow_ratio',
        'max_bytes_per_flow', 'std_bytes_per_flow',
        'max_packets_per_flow', 'std_packets_per_flow'
    ],
    'ip_behavior': [
        'high_volume_src_ratio', 'max_flows_per_src', 'avg_flows_per_src',
        'avg_src_port_diversity', 'port_scanning_ratio'
    ],
    'traffic_anomaly': [
        'packet_rate_std', 'packet_rate_cv', 'flows_per_second'
    ],
    'connection_diversity': [
        'avg_src_ports_per_ip', 'avg_dst_ports_per_ip'
    ],
    # NEW FEATURE GROUPS
    'statistical': [
        'bytes_mean', 'bytes_variance', 'bytes_std', 'bytes_min', 'bytes_max',
        'bytes_range', 'bytes_cv', 'bytes_skewness', 'bytes_kurtosis',
        'packets_mean', 'packets_variance', 'packets_std', 'packets_min', 'packets_max',
        'packets_range', 'packets_cv', 'packets_skewness', 'packets_kurtosis',
        'duration_mean', 'duration_variance', 'duration_std', 'duration_min', 'duration_max',
        'duration_range', 'duration_cv', 'duration_skewness', 'duration_kurtosis'
    ],
    'diversity_indices': [
        'src_ip_simpson_index', 'dst_ip_simpson_index',
        'src_port_simpson_index', 'dst_port_simpson_index',
        'bytes_gini_index', 'packets_gini_index'
    ],
    'uniqueness': [
        'src_ip_unique_count', 'src_ip_uniqueness_ratio', 'src_ip_collision_probability',
        'dst_ip_unique_count', 'dst_ip_uniqueness_ratio', 'dst_ip_collision_probability'
    ],
    'fan_in_out': [
        'max_fan_in', 'avg_fan_in', 'fan_in_std',
        'max_fan_out', 'avg_fan_out', 'fan_out_std',
        'fan_in_out_ratio'
    ],
    'asymmetry': [
        'src_dst_port_entropy_diff', 'src_dst_ip_entropy_diff',
        'port_entropy_abs_diff', 'ip_entropy_abs_diff'
    ],
    'amplification': [
        'avg_request_reply_ratio', 'small_request_large_reply_ratio',
        'amplification_score', 'udp_amplification_potential'
    ],
    'concentration': [
        'dst_top_1_ratio', 'dst_top_5_ratio', 'dst_top_10_ratio',
        'dst_heavy_tail_index', 'dst_concentration_score',
        'dst_port_top_1_ratio', 'dst_port_top_5_ratio', 'dst_port_top_10_ratio',
        'dst_port_heavy_tail_index', 'dst_port_concentration_score'
    ],
    'port_usage': [
        'ephemeral_port_ratio', 'well_known_port_ratio', 'registered_port_ratio',
        'dst_port_concentration', 'src_port_concentration'
    ],
    'entropy_diff': [
        'entropy_delta_src_dst_ip', 'entropy_delta_src_dst_port',
        'entropy_balance_score'
    ],
    'spectral': [
        'spectral_entropy', 'traffic_energy', 'traffic_burstiness'
    ],
    'pure_diversity': [
        'as_diversity_count', 'as_diversity_ratio',
        'geo_diversity_count', 'geo_diversity_ratio',
        'cross_as_flow_ratio', 'cross_geo_flow_ratio'
    ]
}

# Protocol numbers
PROTOCOL_NUMBERS = {
    'ICMP': 1,
    'TCP': 6,
    'UDP': 17
}

# Common TCP flags
TCP_FLAGS = {
    'FIN': 0x01,
    'SYN': 0x02,
    'RST': 0x04,
    'PSH': 0x08,
    'ACK': 0x10,
    'URG': 0x20
}

# Default feature configurations by attack type
FEATURES_BY_ATTACK_TYPE = {
    'syn_flood': {
        'include_groups': ['basic', 'traffic_rates', 'entropy', 'protocol', 'tcp_flags', 'syn_flood_specific', 'connection_patterns'],
        'exclude_features': ['cross_border_ratio', 'src_geo_entropy', 'avg_src_ports_per_ip', 'avg_dst_ports_per_ip']
    },
    'syn_ack_flood': {
        'include_groups': ['basic', 'traffic_rates', 'entropy', 'protocol', 'tcp_flags', 'syn_flood_specific', 'connection_patterns'],
        'exclude_features': ['cross_border_ratio', 'src_geo_entropy', 'avg_src_ports_per_ip', 'avg_dst_ports_per_ip']
    },
    'udp_flood': {
        'include_groups': ['basic', 'traffic_rates', 'entropy', 'protocol', 'ip_diversity', 'flow_patterns', 'ip_behavior', 'amplification'],
        'exclude_features': ['tcp_flags', 'syn_flood_specific', 'http_flood_specific', 'connection_patterns']
    },
    'http_flood': {
        'include_groups': ['basic', 'traffic_rates', 'entropy', 'protocol', 'ip_diversity', 'tcp_flags', 'http_flood_specific'],
        'exclude_features': ['avg_inter_arrival_time', 'std_inter_arrival_time', 'min_inter_arrival_time', 'max_inter_arrival_time', 'inter_arrival_cv', 'burst_ratio', 'periodic_pattern_score']
    },
    'all_features': {
        'include_groups': list(FEATURE_GROUPS.keys()),
        'exclude_features': []
    }
}
