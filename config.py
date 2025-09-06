# Dataset configurations
# Each dataset should specify the path and date patterns for train/validation/test splits

DATASETS = {
    'itp-downstream-http-flood': {
        'path': './datasets/itp-downstream-http-flood/',
        'description': 'ITP downstream HTTP flood attack dataset',
        'patterns': {
            'train': 'nfcapd.20250714*.csv',
            'validation': 'nfcapd.20250715*.csv', 
            'test': 'nfcapd.2025071[67]*.csv'
        },
        'attack_periods': [
            ('2025-07-16 20:25:00', '2025-07-16 20:35:00'),
            ('2025-07-16 22:25:00', '2025-07-16 22:30:00'),
            ('2025-07-17 00:05:00', '2025-07-17 00:05:00'),
            ('2025-07-17 00:30:00', '2025-07-17 00:30:00'),
            ('2025-07-17 00:40:00', '2025-07-17 00:45:00'),
        ]
    }
}

# Default dataset to use when none is specified
DEFAULT_DATASET = 'itp-downstream-http-flood'
