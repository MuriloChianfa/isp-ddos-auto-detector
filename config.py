#
# System constants and configurations
#

from framework.constants import FEATURES_BY_ATTACK_TYPE

# Default dataset to use when none is specified
DEFAULT_DATASET = 'itp-downstream-http-flood'

# Default time span for feature aggregation in seconds
DEFAULT_TIME_SPAN = 300

# Default model threshold calculation strategies
MODEL_THRESHOLD_STRATEGIES = {
    'autoencoder': 'exponential_threshold',
    'isolation_forest': 'percentile_99_5',
    'one_class_svm': 'mean_plus_3std',
    'local_outlier_factor': 'mse_plus_8std'
}

# Default model hyperparameters
MODEL_DEFAULT_PARAMS = {
    'autoencoder': {
        'latent_dim': 42
    },
    'isolation_forest': {
        'contamination': 0.05,
        'n_estimators': 200,
        'random_state': 42
    },
    'one_class_svm': {
        'nu': 0.1,
        'kernel': 'rbf',
        'gamma': 'scale'
    },
    'local_outlier_factor': {
        'n_neighbors': 20,
        'contamination': 0.05,
        'novelty': True,
        'random_state': 42,
        'algorithm': 'auto',  # Options: 'auto', 'ball_tree', 'kd_tree', 'brute'
        'leaf_size': 30,      # Affects speed of BallTree/KDTree algorithms
        'metric': 'minkowski', # Distance metric: 'minkowski', 'euclidean', 'manhattan', 'chebyshev', etc.
        'p': 2                 # Minkowski metric power parameter (1=Manhattan, 2=Euclidean)
    }
}

DATASETS = {
    'itp-downstream-http-flood': {
        'path': './datasets/itp-downstream-http-flood/',
        'description': 'ITP downstream HTTP flood attack dataset',
        'patterns': {
            'train': 'nfcapd.20250714*.csv',
            'validation': 'nfcapd.20250715*.csv',
            'test': 'nfcapd.2025071[67]*.csv'
        },
        'feature_config': FEATURES_BY_ATTACK_TYPE['http_flood'],
        'windows': {
            '1': {
                'attack_periods': [
                    ('2025-07-16 20:27:00', '2025-07-16 20:36:50'),
                    ('2025-07-16 22:22:50', '2025-07-16 22:22:50'),
                ],
                'threshold_strategies': {
                    'autoencoder': 'exponential_threshold',
                    'isolation_forest': 'percentile_99_5',
                    'one_class_svm': 'mean_plus_3std'
                }
            },
            '10': {
                'attack_periods': [
                    ('2025-07-16 20:27:00', '2025-07-16 20:36:50'),
                    ('2025-07-16 22:22:50', '2025-07-16 22:22:50'),
                ],
                'threshold_strategies': {
                    'autoencoder': 'exponential_threshold',
                    'isolation_forest': 'percentile_99_5',
                    'one_class_svm': 'mean_plus_3std'
                },
            },
            '60': {
                'attack_periods': [
                    ('2025-07-16 20:27:00', '2025-07-16 20:36:50'),
                    ('2025-07-16 22:22:50', '2025-07-16 22:22:50'),
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
                    ('2025-07-16 20:27:00', '2025-07-16 20:36:50'),
                    ('2025-07-16 22:22:50', '2025-07-16 22:22:50'),
                ],
                'threshold_strategies': {
                    'autoencoder': 'exponential_threshold',
                    'isolation_forest': 'percentile_99_5',
                    'one_class_svm': 'percentile_99_9'
                }
            }
        }
    },
    'itp-synack-customer-outage': {
        'path': './datasets/itp-synack-customer-outage/',
        'description': 'ITP SYN+ACK flood attack causing customer outage',
        'patterns': {
            'train': ['nfcapd.20251004*.csv', 'nfcapd.20251005*.csv', 'nfcapd.20251006*.csv', 'nfcapd.20251007*.csv', 'nfcapd.20251008*.csv'],
            'validation': ['nfcapd.20251009*.csv', 'nfcapd.20251010*.csv'],
            'test': ['nfcapd.20251011*.csv', 'nfcapd.20251012*.csv', 'nfcapd.20251013*.csv']
        },
        'feature_config': FEATURES_BY_ATTACK_TYPE['syn_ack_flood'],
        'windows': {
            '1': {
                'attack_periods': [
                    ('2025-10-13 15:26:30', '2025-10-13 17:16:20'),
                ],
                'threshold_strategies': {
                    'autoencoder': 'exponential_threshold',
                    'isolation_forest': 'percentile_99_9',
                    'one_class_svm': 'mean_plus_3std'
                },
                'params': {
                    'isolation_forest': {
                        'contamination': 0.14,
                        'n_estimators': 320,
                        'random_state': 76
                    }
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
                }
            }
        }
    },
    'itp-multivector-udp-100gbps-peak': {
        'path': './datasets/itp-multivector-udp-100gbps-peak/',
        'description': 'ITP multi-vector UDP flood attack with 100Gbps peak',
        'patterns': {
            'train': ['nfcapd.20251008*.csv', 'nfcapd.20251009*.csv', 'nfcapd.20251010*.csv'],
            'validation': 'nfcapd.20251011*.csv',
            'test': 'nfcapd.2025101[2-6]*.csv',
            'horizon': ['nfcapd.2025101[7-9]*.csv', 'nfcapd.2025102[0-6]*.csv']
        },
        'feature_config': FEATURES_BY_ATTACK_TYPE['udp_flood'],
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
                    'isolation_forest': 'percentile_99_9',
                    'one_class_svm': 'mean_plus_3std',
                },
                'params': {
                    'isolation_forest': {
                        'contamination': 0.04,
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
                        'contamination': 0.05, # 'auto'
                        'n_estimators': 320, # usar menos
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
        'path': './datasets/isp-synflood-multiple-days/',
        'description': 'ISP SYN-Flood attack during multiple days',
        'patterns': {
            'train': 'nfcapd.2025081[6789]*.csv',
            'validation': 'nfcapd.20250820*.csv',
            'test': 'nfcapd.2025082[1-8]*.csv',
            'horizon': ['nfcapd.20250829*.csv', 'nfcapd.20250830*.csv', 'nfcapd.20250831*.csv', 'nfcapd.2025090[1-9]*.csv']
        },
        'feature_config': FEATURES_BY_ATTACK_TYPE['syn_flood'],
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
                    'one_class_svm': 'mean_plus_3std'
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
                    'one_class_svm': 'mean_plus_3std'
                }
            }
        }
    }
}
