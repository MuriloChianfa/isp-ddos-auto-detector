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
    },
    'isp-synflood-multiple-days': {
        'path': './datasets/isp-synflood-multiple-days/',
        'description': 'ISP SYN-Flood attack during multiple days',
        'patterns': {
            'train': 'nfcapd.20250819*.csv',
            'validation': 'nfcapd.20250820*.csv',
            'test': 'nfcapd.2025082[1-8]*.csv'
        },
        'attack_periods': [
            ('2025-08-21 10:00:00', '2025-08-21 10:10:00'),
            ('2025-08-21 20:00:00', '2025-08-21 20:10:00'),
            ('2025-08-24 08:00:00', '2025-08-24 08:10:00'),
            ('2025-08-24 18:00:00', '2025-08-24 18:10:00'),
            ('2025-08-25 04:00:00', '2025-08-25 04:10:00'),
            ('2025-08-25 14:00:00', '2025-08-25 14:10:00'),
            ('2025-08-25 14:00:00', '2025-08-25 14:10:00'),
            ('2025-08-26 02:00:00', '2025-08-26 02:10:00'),
            ('2025-08-26 12:00:00', '2025-08-26 12:10:00'),
            ('2025-08-26 22:00:00', '2025-08-26 22:10:00'),
            ('2025-08-27 08:00:00', '2025-08-27 08:10:00'),
        ]
    }
}

# Default dataset to use when none is specified
DEFAULT_DATASET = 'itp-downstream-http-flood'
