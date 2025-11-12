#
# System constants and configurations
#

from framework.constants import FEATURES_BY_ATTACK_TYPE

# Default dataset to use when none is specified
DEFAULT_DATASET = 'itp-downstream-http-flood'

# Default time span for feature aggregation in seconds
DEFAULT_TIME_SPAN = 300

# Default model threshold calculation strategies
# Available strategies for all models:
#   - 'mean_plus_3std': threshold = μ + 3σ
#   - 'mse_plus_8std': threshold = MSE + 8σ
#   - 'mse_plus_80std': threshold = MSE + 80σ
#   - 'percentile_99': threshold = P₉₉(errors)
#   - 'percentile_99_5': threshold = P₉₉.₅(errors)
#   - 'percentile_99_9': threshold = P₉₉.₉(errors)
#   - 'exponential_threshold': threshold = μ + k·e^(-λt)·σ
#   - 'sigmoid_threshold': threshold = μ + σ/(1 + e^(-k(x-x₀)))
MODEL_THRESHOLD_STRATEGIES = {
    'autoencoder': 'exponential_threshold',
    'isolation_forest': 'percentile_99_5',
    'one_class_svm': 'mean_plus_3std',
    'local_outlier_factor': 'mse_plus_8std'
}

# Default model hyperparameters
MODEL_DEFAULT_PARAMS = {
    'autoencoder': {
        'hidden_layers': [24, 18, 12],  # Hidden layer dimensions for encoder (decoder mirrors these). Example: [24, 18, 12] creates encoder: input -> 24 -> 18 -> 12 -> latent
        'latent_dim': 8  # Dimensionality of the latent space (bottleneck layer). Lower = more compression. Typical range: 2-64. Too low may lose important patterns, too high may not compress enough
    },
    'isolation_forest': {
        'contamination': 0.05, # Expected proportion of outliers: 0.01-0.05 (clean data), 0.1-0.2 (noisy data)
        'n_estimators': 200,   # Number of trees in the forest (default: 100). More trees = more stable but slower
        'random_state': 42     # Seed for reproducibility of results
    },
    'one_class_svm': {
        'nu': 0.1,            # Upper bound on fraction of training errors and lower bound on support vectors (0.0-1.0), common values: 0.01-0.05 (strict), 0.1-0.2 (moderate). Lower = tighter boundary
        'kernel': 'rbf',      # Kernel type: 'linear' (fast, simple), 'rbf' (flexible, default), 'poly' (polynomial), 'sigmoid'
        'gamma': 'scale'      # Kernel coefficient: 'scale' (1/(n_features*X.var())), 'auto' (1/n_features), or float, higher values = more complex decision boundary, risk of overfitting
    },
    'local_outlier_factor': {
        'n_neighbors': 20,     # Number of neighbors to use (default: 20). Higher values = smoother decision boundaries
        'contamination': 0.05, # Expected proportion of outliers: 0.01-0.05 (clean data), 0.1-0.2 (noisy data)
        'novelty': True,       # If True, can be used for novelty detection; if False, for outlier detection only
        'random_state': 42,    # Seed for reproducibility of results
        'algorithm': 'auto',   # Algorithm to compute nearest neighbors: 'auto', 'ball_tree', 'kd_tree', 'brute'
        'leaf_size': 30,       # Leaf size for BallTree/KDTree algorithms. Affects construction/query speed (default: 30)
        'metric': 'minkowski', # Distance metric: 'minkowski', 'euclidean', 'manhattan', 'chebyshev', 'cosine', etc.
        'p': 2                 # Power parameter for Minkowski metric (1=Manhattan, 2=Euclidean, inf=Chebyshev)
    }
}

# Hyperparameter optimization configuration
OPTIMIZATION_CONFIG = {
    'n_iter': 10,
    'scoring': 'anomaly_score',
    'random_state': 42
}

# Default EMA (Exponential Moving Average) configuration
# EMA smoothing is applied to reduce noise in time-series features
# 
# How it works:
#   1. Features listed in 'features' will have EMA versions created (suffix: _ema)
#   2. EMA features are always generated and added to the feature set
#   3. Use 'include_groups': ['ema_smoothed'] in feature_config to select them
#   4. Override 'alpha' per time window using 'ema_alpha' in window config
#
# Alpha parameter (0 < alpha < 1):
#   - Lower values (0.05-0.1): More smoothing, slower response to changes
#   - Higher values (0.2-0.5): Less smoothing, faster response to changes
DEFAULT_EMA_CONFIG = {
    'alpha': 0.1,  # Smoothing factor (0 < alpha < 1). Lower = more smoothing
    'features': [  # Features that will have EMA versions created
        'packet_rate', 
        'bit_rate', 
        'flow_rate', 
        'duration_mean', 
        'bytes_std', 
        'packets_std', 
        'flows_per_second'
    ]
}

DATASETS = {
    'itp-downstream-http-flood': {
        'path': './datasets/itp-downstream-http-flood/raw/',
        'description': 'ITP downstream HTTP flood attack dataset',
        'patterns': {
            'train': 'nfcapd.20250714*.csv',
            'validation': 'nfcapd.20250715*.csv',
            'test': 'nfcapd.2025071[67]*.csv'
        },
        # 'feature_config': FEATURES_BY_ATTACK_TYPE['all_features'],
        'feature_config': [
            'total_flows', 'total_packets', 'total_bytes', 'avg_duration',
            'packet_rate', 'bit_rate', 'flow_rate', 'avg_packet_size',
            'packets_per_flow', 'bytes_per_flow', 'src_port_entropy',
            'dst_port_entropy', 'src_ip_entropy', 'unique_src_ips',
            'tcp_ratio', 'udp_ratio', 'icmp_ratio', 'size_uniformity',
            'syn_flag_ratio', 'ack_flag_ratio', 'fin_flag_ratio',
            'rst_flag_ratio', 'psh_flag_ratio', 'large_packet_flow_ratio',
            'avg_dst_port_diversity'
        ],
        'windows': {
            '1': {
                'attack_periods': [
                    ('2025-07-16 20:27:14', '2025-07-16 20:32:14'),
                    ('2025-07-16 22:23:07', '2025-07-16 22:23:07'),
                    ('2025-07-16 22:24:57', '2025-07-16 22:25:03'),
                    ('2025-07-16 22:33:05', '2025-07-16 22:33:07'),
                    ('2025-07-17 00:09:16', '2025-07-17 00:09:19'),
                    ('2025-07-17 00:30:15', '2025-07-17 00:30:17'),
                ],
                'threshold_strategies': {
                    'autoencoder': 'mse_plus_40std',
                    'isolation_forest': 'mse_plus_4_5std',
                    'one_class_svm': 'mse_plus_15std'
                },
                'feature_config': {
                    'one_class_svm': [
                        'total_flows', 'total_packets', 'total_bytes', 'avg_duration',
                        'packet_rate', 'bit_rate', 'flow_rate', 'avg_packet_size', 
                        'packets_per_flow', 'bytes_per_flow', 'fan_in_std',
                        'src_port_entropy', 'dst_port_entropy', 'src_ip_entropy',
                        'tcp_ratio', 'udp_ratio', 'icmp_ratio', 'size_uniformity',
                        'syn_flag_ratio', 'ack_flag_ratio', 'fin_flag_ratio', 
                        'rst_flag_ratio', 'psh_flag_ratio', 'avg_dst_port_diversity',
                        'syn_flood_ratio', 'syn_ack_ratio', 'small_packet_ratio',
                        'avg_tcp_duration', 'zero_duration_ratio', 'avg_ports_per_src',
                        'connection_establishment_ratio', 'connection_teardown_ratio',
                        'incomplete_connection_ratio', 'zero_duration_connections',
                        'very_short_connections', 'max_ports_per_src', 'large_packet_flow_ratio',
                        'well_known_port_ratio', 'dst_port_top_1_ratio', 'bytes_kurtosis',
                        'duration_skewness', 'geo_diversity_count', 'dst_ip_unique_count',
                        'packets_kurtosis', 'src_ip_unique_count', 'unique_src_geo',
                    ],
                    'isolation_forest': [
                        'total_flows', 'total_packets', 'total_bytes', 'avg_duration',
                        'packet_rate', 'bit_rate', 'flow_rate', 'avg_packet_size', 
                        'packets_per_flow', 'bytes_per_flow', 'fan_in_std',
                        'src_port_entropy', 'dst_port_entropy', 'src_ip_entropy',
                        'tcp_ratio', 'udp_ratio', 'icmp_ratio', 'size_uniformity',
                        'syn_flag_ratio', 'ack_flag_ratio', 'fin_flag_ratio', 
                        'rst_flag_ratio', 'psh_flag_ratio', 'avg_dst_port_diversity',
                        'syn_flood_ratio', 'syn_ack_ratio', 'small_packet_ratio',
                        'avg_tcp_duration', 'zero_duration_ratio', 'avg_ports_per_src',
                        'connection_establishment_ratio', 'connection_teardown_ratio',
                        'incomplete_connection_ratio', 'zero_duration_connections',
                        'very_short_connections', 'max_ports_per_src', 'large_packet_flow_ratio',
                        'well_known_port_ratio', 'dst_port_top_1_ratio', 'bytes_kurtosis',
                        'duration_skewness', 'geo_diversity_count', 'dst_ip_unique_count',
                        'packets_kurtosis', 'src_ip_unique_count', 'unique_src_geo',
                    ]
                },
                'params': {
                    'isolation_forest': {
                        'contamination': 0.03,
                        'n_estimators': 168,
                        'max_samples': 64,
                        'max_features': 1.0,
                        'bootstrap': True,
                        'random_state': 82,
                    },
                    'one_class_svm': {
                        'nu': 0.1,
                        'kernel': 'sigmoid',
                        'gamma': 'auto'
                    },
                }
                # 'ema_alpha': 0.15,
            },
            '10': {
                'attack_periods': [
                    ('2025-07-16 20:27:10', '2025-07-16 20:37:00'),
                    ('2025-07-16 22:24:50', '2025-07-16 22:25:10'),
                    ('2025-07-16 22:33:00', '2025-07-16 22:33:00'),
                    ('2025-07-17 00:09:10', '2025-07-17 00:09:10'),
                    ('2025-07-17 00:30:10', '2025-07-17 00:30:10'),
                    ('2025-07-17 00:38:00', '2025-07-17 00:51:40'),
                ],
                'threshold_strategies': {
                    'autoencoder': 'exponential_threshold',
                    'isolation_forest': 'percentile_99_5',
                    'one_class_svm': 'percentile_99'
                },
                'feature_config': {
                    'one_class_svm': ['total_flows', 'total_packets', 'tcp_ratio', 'bit_rate', 'syn_flag_ratio', 'bytes_per_flow'],
                },
            },
            '60': {
                'attack_periods': [
                    ('2025-07-16 20:27:00', '2025-07-16 20:37:00'),
                    ('2025-07-16 22:25:00', '2025-07-16 22:25:00'),
                    ('2025-07-16 22:33:00', '2025-07-16 22:33:00'),
                    ('2025-07-17 00:09:00', '2025-07-17 00:09:00'),
                    ('2025-07-17 00:30:00', '2025-07-17 00:30:00'),
                    ('2025-07-17 00:38:00', '2025-07-17 00:50:00'),
                ],
                'threshold_strategies': {
                    'autoencoder': 'exponential_threshold',
                    'isolation_forest': 'percentile_99_5',
                    'one_class_svm': 'mean_plus_3std',
                    'local_outlier_factor': 'sigmoid_threshold'
                }
            },
            '300': {
                'attack_periods': [
                    ('2025-07-16 20:25:00', '2025-07-16 20:35:00'),
                    ('2025-07-16 22:30:00', '2025-07-16 22:30:00'),
                    ('2025-07-17 00:30:00', '2025-07-17 00:30:00'),
                    ('2025-07-17 00:40:00', '2025-07-17 00:45:00'),
                ],
                'threshold_strategies': {
                    'autoencoder': 'exponential_threshold',
                    'isolation_forest': 'percentile_99_5',
                    'one_class_svm': 'percentile_99'
                }
            }
        }
    },
    'itp-synack-customer-outage': {
        'path': './datasets/itp-synack-customer-outage/raw/',
        'description': 'ITP SYN+ACK flood attack causing customer outage',
        'patterns': {
            'train': ['nfcapd.20251004*.csv', 'nfcapd.20251005*.csv', 'nfcapd.20251006*.csv', 'nfcapd.20251007*.csv', 'nfcapd.20251008*.csv'],
            'validation': ['nfcapd.20251009*.csv', 'nfcapd.20251010*.csv'],
            'test': ['nfcapd.20251011*.csv', 'nfcapd.20251012*.csv', 'nfcapd.20251013*.csv']
        },
        # 'feature_config': FEATURES_BY_ATTACK_TYPE['all_features'],
        'feature_config': [
            'total_flows', 'total_packets', 'total_bytes', 'avg_duration',
            'packet_rate', 'bit_rate', 'flow_rate', 'avg_packet_size', 
            'packets_per_flow', 'bytes_per_flow',
            'src_port_entropy', 'dst_port_entropy', 'src_ip_entropy',
            'tcp_ratio', 'udp_ratio', 'icmp_ratio',
            'syn_flag_ratio', 'ack_flag_ratio', 'fin_flag_ratio', 
            'rst_flag_ratio', 'psh_flag_ratio',
            'syn_flood_ratio', 'syn_ack_ratio', 'small_packet_ratio',
            'avg_tcp_duration', 'zero_duration_ratio', 'avg_ports_per_src',
            'max_ports_per_src',
            'connection_establishment_ratio', 'connection_teardown_ratio',
            'incomplete_connection_ratio', 'zero_duration_connections',
            'very_short_connections'
        ],
        'windows': {
            '1': {
                'attack_periods': [
                    ('2025-10-13 15:26:30', '2025-10-13 17:16:20'),
                ],
                'threshold_strategies': {
                    'autoencoder': 'sigmoid_threshold',
                    'isolation_forest': 'percentile_99_9',
                    'one_class_svm': 'mse_plus_4_5std',
                    'local_outlier_factor': 'mse_plus_15std'
                },
                'params': {
                    'isolation_forest': {
                        'contamination': 0.14,
                        'n_estimators': 320,
                        'random_state': 76
                    },
                    'one_class_svm': {
                        'nu': 0.1,
                        'tol': 1e-4,
                        'kernel': 'sgd_rbf',
                        'gamma': 'scale'
                    },
                    'local_outlier_factor': {
                        'n_neighbors': 100,
                        'contamination': 0.269,
                        'novelty': True,
                        'random_state': 42,
                        'algorithm': 'kd_tree',
                        'leaf_size': 30,
                        'metric': 'minkowski',
                        'p': 1
                    }
                },
                'feature_config': {
                    'one_class_svm': [
                        'total_flows', 'avg_duration',
                        'packet_rate', 'bit_rate', 'flow_rate',
                        'packets_per_flow', 'dst_port_entropy', 'src_ip_entropy',
                        'tcp_ratio', 'syn_flag_ratio', 'ack_flag_ratio', 'psh_flag_ratio',
                        'max_ports_per_src', 'small_packet_ratio', 'zero_duration_connections',
                    ],
                }
            },
            '10': {
                'attack_periods': [
                    ('2025-10-13 15:26:30', '2025-10-13 17:16:20'),
                ],
                'threshold_strategies': {
                    'autoencoder': 'exponential_threshold',
                    'isolation_forest': 'percentile_99_9',
                    'one_class_svm': 'percentile_99_9',
                    'local_outlier_factor': 'percentile_99'
                },
                'params': {
                    'one_class_svm': {
                        'nu': 0.05,
                        'kernel': 'sigmoid',
                        'gamma': 'scale'
                    },
                    'local_outlier_factor': {
                        'n_neighbors': 48,
                        'contamination': 0.1,
                        'novelty': True,
                        'random_state': 86,
                        'algorithm': 'ball_tree',
                        'leaf_size': 42,
                        'metric': 'chebyshev',
                        'p': 2
                    }
                }
            },
            '60': {
                'attack_periods': [
                    ('2025-10-13 15:28:00', '2025-10-13 17:15:00'),
                ],
                'threshold_strategies': {
                    'autoencoder': 'exponential_threshold',
                    'isolation_forest': 'percentile_99_9',
                    'one_class_svm': 'mse_plus_8std',
                    'local_outlier_factor': 'exponential_threshold'
                },
                'params': {
                    'one_class_svm': {
                        'nu': 0.15,
                        'kernel': 'sigmoid',
                        'gamma': 'scale'
                    }
                }
            },
            '300': {
                'attack_periods': [
                    ('2025-10-13 15:35:00', '2025-10-13 17:25:00'),
                ],
                'threshold_strategies': {
                    'autoencoder': 'exponential_threshold',
                    'isolation_forest': 'percentile_99_9',
                    'one_class_svm': 'percentile_99_9'
                },
            }
        }
    },
    'itp-multivector-udp-100gbps-peak': {
        'path': './datasets/itp-multivector-udp-100gbps-peak/raw/',
        'description': 'ITP multi-vector UDP flood attack with 100Gbps peak',
        'patterns': {
            'train': ['nfcapd.20251008*.csv', 'nfcapd.20251009*.csv', 'nfcapd.20251010*.csv'],
            'validation': 'nfcapd.20251011*.csv',
            'test': 'nfcapd.2025101[2-6]*.csv',
            'horizon': ['nfcapd.2025101[7-9]*.csv', 'nfcapd.2025102[0-6]*.csv']
        },
        # 'feature_config': FEATURES_BY_ATTACK_TYPE['all_features'],
        'feature_config': [
            'as_diversity_count', 'bytes_kurtosis',
            'dst_ip_unique_count', 'dst_port_top_1_ratio',
            'duration_kurtosis', 'fan_in_std', 'flow_rate',
            'flows_per_second', 'geo_diversity_count', 'packets_kurtosis',
            'src_ip_unique_count', 'total_flows', 'unique_src_as',
            'unique_src_geo', 'unique_src_ips', 'well_known_port_ratio'
        ],
        'windows': {
            '1': {
                'attack_periods': [
                    ('2025-10-15 20:28:40', '2025-10-15 20:42:10'),
                    ('2025-10-15 20:53:40', '2025-10-15 21:06:20'),
                    ('2025-10-15 22:44:20', '2025-10-15 22:54:40'),
                    ('2025-10-16 00:10:40', '2025-10-16 00:19:40'),
                ],
                'threshold_strategies': {
                    'autoencoder': 'exponential_threshold',
                    'isolation_forest': 'mse_plus_5_5std',
                    'one_class_svm': 'mse_plus_4_5std',
                    'local_outlier_factor': 'mse_plus_40std',
                },
                'params': {
                    'isolation_forest': {
                        'contamination': 0.04,
                        'n_estimators': 82,
                        'random_state': 64,
                        'max_samples': 115,
                        'max_features': 0.8,
                        'bootstrap': True
                    },
                    'one_class_svm': {
                        'nu': 0.1,
                        'kernel': 'sigmoid',
                        'gamma': 'auto'
                    }
                },
                'feature_config': {
                    'autoencoder': [
                        'total_flows', 'total_packets', 'total_bytes', 'avg_duration',
                        'packet_rate', 'bit_rate', 'flow_rate', 'avg_packet_size', 
                        'packets_per_flow', 'bytes_per_flow', 'well_known_port_ratio'
                        'src_port_entropy', 'dst_port_entropy', 'src_ip_entropy',
                        'tcp_ratio', 'udp_ratio', 'icmp_ratio', 'dst_port_top_1_ratio',
                        'avg_tcp_duration', 'zero_duration_ratio', 'avg_ports_per_src',
                        'max_ports_per_src', 'bytes_kurtosis', 'dst_ip_unique_count',
                        'duration_skewness', 'fan_in_std', 'geo_diversity_count',
                        'packets_kurtosis', 'src_ip_unique_count', 'unique_src_geo'
                    ],
                    'isolation_forest': [
                        'total_flows', 'total_packets', 'total_bytes', 'avg_duration',
                        'packet_rate', 'bit_rate', 'flow_rate', 'avg_packet_size', 
                        'packets_per_flow', 'bytes_per_flow', 'well_known_port_ratio'
                        'src_port_entropy', 'dst_port_entropy', 'src_ip_entropy',
                        'tcp_ratio', 'udp_ratio', 'icmp_ratio', 'dst_port_top_1_ratio',
                        'avg_tcp_duration', 'zero_duration_ratio', 'avg_ports_per_src',
                        'max_ports_per_src', 'bytes_kurtosis', 'dst_ip_unique_count',
                        'duration_skewness', 'fan_in_std', 'geo_diversity_count',
                        'packets_kurtosis', 'src_ip_unique_count', 'unique_src_geo'
                    ],
                    'one_class_svm': [
                        'bytes_kurtosis', 'dst_ip_unique_count', 'dst_port_top_1_ratio',
                        'duration_skewness', 'fan_in_std', 'geo_diversity_count',
                        'packets_kurtosis', 'src_ip_unique_count', 'total_flows',
                        'flow_rate', 'unique_src_geo', 'well_known_port_ratio'
                    ],
                },
            },
            '10': {
                'attack_periods': [
                    ('2025-10-15 20:28:40', '2025-10-15 20:42:10'),
                    ('2025-10-15 20:53:40', '2025-10-15 21:06:20'),
                    ('2025-10-15 22:44:20', '2025-10-15 22:54:40'),
                    ('2025-10-16 00:10:40', '2025-10-16 00:19:40'),
                ],
                'threshold_strategies': {
                    'autoencoder': 'exponential_threshold',
                    'isolation_forest': 'percentile_99_5',
                    'one_class_svm': 'percentile_99_5',
                    'local_outlier_factor': 'percentile_99_9'
                },
                'params': {
                    'isolation_forest': {
                        'contamination': 0.05,
                        'n_estimators': 320,
                        'random_state': 64
                    },
                    'one_class_svm': {
                        'nu': 0.06,
                        'kernel': 'sigmoid',
                        'gamma': 'scale'
                    }
                }
            },
            '60': {
                'attack_periods': [
                    ('2025-10-15 20:29:00', '2025-10-15 20:42:00'),
                    ('2025-10-15 20:54:00', '2025-10-15 21:03:00'),
                    ('2025-10-15 22:46:00', '2025-10-15 22:54:00'),
                    ('2025-10-16 00:11:00', '2025-10-16 00:18:00'),
                ],
                'threshold_strategies': {
                    'autoencoder': 'exponential_threshold',
                    'isolation_forest': 'percentile_99_5',
                    'one_class_svm': 'mse_plus_8std',
                    'local_outlier_factor': 'percentile_99_9'
                },
                'params': {
                    'one_class_svm': {
                        'nu': 0.15,
                        'kernel': 'sigmoid',
                        'gamma': 'scale'
                    }
                }
            },
            '300': {
                'attack_periods': [
                    ('2025-10-15 20:25:00', '2025-10-15 20:40:00'),
                    ('2025-10-15 20:50:00', '2025-10-15 21:15:00'),
                    ('2025-10-15 22:45:00', '2025-10-15 22:50:00'),
                    ('2025-10-16 00:10:00', '2025-10-16 00:20:00'),
                ],
                'threshold_strategies': {
                    'autoencoder': 'mse_plus_3std',
                    'isolation_forest': 'percentile_99_9',
                    'one_class_svm': 'mse_plus_8std',
                    'local_outlier_factor': 'percentile_99_9'
                },
                'params': {
                    'isolation_forest': {
                        'contamination': 0.05,
                        'n_estimators': 300,
                        'random_state': 42
                    }
                }
            }
        }
    },
    'isp-synflood-multiple-days': {
        'path': './datasets/isp-synflood-multiple-days/raw/',
        'description': 'ISP SYN-Flood attack during multiple days',
        'patterns': {
            'train': 'nfcapd.2025081[6789]*.csv',
            'validation': 'nfcapd.20250820*.csv',
            'test': 'nfcapd.2025082[1-8]*.csv',
            'horizon': ['nfcapd.20250829*.csv', 'nfcapd.20250830*.csv', 'nfcapd.20250831*.csv', 'nfcapd.2025090[1-9]*.csv']
        },
        # 'feature_config': FEATURES_BY_ATTACK_TYPE['all_features'],
        'feature_config': [
            'total_flows', 'total_packets', 'total_bytes', 'avg_duration',
            'packet_rate', 'bit_rate', 'flow_rate', 'avg_packet_size', 
            'packets_per_flow', 'bytes_per_flow',
            'src_port_entropy', 'dst_port_entropy', 'src_ip_entropy',
            'tcp_ratio', 'udp_ratio', 'icmp_ratio',
            'syn_flag_ratio', 'ack_flag_ratio', 'fin_flag_ratio', 
            'rst_flag_ratio', 'psh_flag_ratio',
            'syn_flood_ratio', 'syn_ack_ratio', 'small_packet_ratio',
            'avg_tcp_duration', 'zero_duration_ratio', 'avg_ports_per_src',
            'max_ports_per_src',
            'connection_establishment_ratio', 'connection_teardown_ratio',
            'incomplete_connection_ratio', 'zero_duration_connections',
            'very_short_connections'
        ],
        'windows': {
            '1': {
                'attack_periods': [
                    ('2025-08-21 09:35:00', '2025-08-21 09:50:00'),
                    ('2025-08-21 19:10:00', '2025-08-21 19:30:00'),
                    ('2025-08-22 11:55:00', '2025-08-22 12:10:00'),
                    ('2025-08-24 06:15:00', '2025-08-24 06:30:00'),
                    ('2025-08-24 16:40:00', '2025-08-24 16:55:00'),
                    ('2025-08-25 03:10:00', '2025-08-25 03:25:00'),
                    ('2025-08-25 13:35:00', '2025-08-25 13:50:00'),
                    ('2025-08-26 00:05:00', '2025-08-26 00:20:00'),
                    ('2025-08-26 10:30:00', '2025-08-26 10:50:00'),
                    ('2025-08-26 21:00:00', '2025-08-26 21:15:00'),
                    ('2025-08-27 07:05:00', '2025-08-27 07:20:00'),
                ],
                'threshold_strategies': {
                    'autoencoder': 'exponential_threshold',
                    'isolation_forest': 'percentile_99_5',
                    'one_class_svm': 'mse_plus_4_5std'
                },
                'params': {
                    'one_class_svm': {
                        'nu': 0.01,
                        'tol': 1e-7,
                        'kernel': 'sgd_rbf',
                    },
                }
            },
            '10': {
                'attack_periods': [
                    ('2025-08-21 09:35:00', '2025-08-21 09:50:00'),
                    ('2025-08-21 19:10:00', '2025-08-21 19:30:00'),
                    ('2025-08-22 11:55:00', '2025-08-22 12:10:00'),
                    ('2025-08-24 06:15:00', '2025-08-24 06:30:00'),
                    ('2025-08-24 16:40:00', '2025-08-24 16:55:00'),
                    ('2025-08-25 03:10:00', '2025-08-25 03:25:00'),
                    ('2025-08-25 13:35:00', '2025-08-25 13:50:00'),
                    ('2025-08-26 00:05:00', '2025-08-26 00:20:00'),
                    ('2025-08-26 10:30:00', '2025-08-26 10:50:00'),
                    ('2025-08-26 21:00:00', '2025-08-26 21:15:00'),
                    ('2025-08-27 07:05:00', '2025-08-27 07:20:00'),
                ],
                'threshold_strategies': {
                    'autoencoder': 'exponential_threshold',
                    'isolation_forest': 'percentile_99_5',
                    'one_class_svm': 'mean_plus_3std'
                }
            },
            '60': {
                'attack_periods': [
                    ('2025-08-21 09:35:00', '2025-08-21 09:50:00'),
                    ('2025-08-21 19:10:00', '2025-08-21 19:30:00'),
                    ('2025-08-22 11:55:00', '2025-08-22 12:10:00'),
                    ('2025-08-24 06:15:00', '2025-08-24 06:30:00'),
                    ('2025-08-24 16:40:00', '2025-08-24 16:55:00'),
                    ('2025-08-25 03:10:00', '2025-08-25 03:25:00'),
                    ('2025-08-25 13:35:00', '2025-08-25 13:50:00'),
                    ('2025-08-26 00:05:00', '2025-08-26 00:20:00'),
                    ('2025-08-26 10:30:00', '2025-08-26 10:50:00'),
                    ('2025-08-26 21:00:00', '2025-08-26 21:15:00'),
                    ('2025-08-27 07:05:00', '2025-08-27 07:20:00'),
                ],
                'threshold_strategies': {
                    'autoencoder': 'exponential_threshold',
                    'isolation_forest': 'percentile_99_5',
                    'one_class_svm': 'mean_plus_3std'
                }
            },
            '300': {
                'attack_periods': [
                    ('2025-08-21 09:35:00', '2025-08-21 09:50:00'),
                    ('2025-08-21 19:10:00', '2025-08-21 19:30:00'),
                    ('2025-08-22 11:55:00', '2025-08-22 12:10:00'),
                    ('2025-08-24 06:15:00', '2025-08-24 06:30:00'),
                    ('2025-08-24 16:40:00', '2025-08-24 16:55:00'),
                    ('2025-08-25 03:10:00', '2025-08-25 03:25:00'),
                    ('2025-08-25 13:35:00', '2025-08-25 13:50:00'),
                    ('2025-08-26 00:05:00', '2025-08-26 00:20:00'),
                    ('2025-08-26 10:30:00', '2025-08-26 10:50:00'),
                    ('2025-08-26 21:00:00', '2025-08-26 21:15:00'),
                    ('2025-08-27 07:05:00', '2025-08-27 07:20:00'),
                ],
                'threshold_strategies': {
                    'autoencoder': 'exponential_threshold',
                    'isolation_forest': 'percentile_99_5',
                    'one_class_svm': 'percentile_99_9'
                }
            }
        }
    }
}
