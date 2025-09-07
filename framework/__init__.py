"""
Framework module for ISP DDoS Auto Detector
"""

from .constants import FEATURE_GROUPS, FEATURES_BY_ATTACK_TYPE, PROTOCOL_NUMBERS, TCP_FLAGS

__all__ = [
    'FEATURE_GROUPS',
    'FEATURES_BY_ATTACK_TYPE', 
    'PROTOCOL_NUMBERS',
    'TCP_FLAGS'
]
