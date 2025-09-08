# Dataset configurations
# Each dataset should specify the path and date patterns for train/validation/test splits

from framework.constants import FEATURES_BY_ATTACK_TYPE

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
            ('2025-07-16 22:20:00', '2025-07-16 22:30:00'),
            ('2025-07-17 00:05:00', '2025-07-17 00:05:00'),
            ('2025-07-17 00:20:00', '2025-07-17 00:30:00'),
            ('2025-07-17 00:40:00', '2025-07-17 00:45:00'),
        ],
        'feature_config': FEATURES_BY_ATTACK_TYPE['http_flood']
    },
    'isp-synflood-multiple-days': {
        'path': './datasets/isp-synflood-multiple-days/',
        'description': 'ISP SYN-Flood attack during multiple days',
        'patterns': {
            'train': 'nfcapd.2025081[6789]*.csv',
            'validation': 'nfcapd.20250820*.csv',
            'test': 'nfcapd.2025082[1-8]*.csv'
        },
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
        'feature_config': FEATURES_BY_ATTACK_TYPE['syn_flood']
    }
}

# Default dataset to use when none is specified
DEFAULT_DATASET = 'itp-downstream-http-flood'
