#!/usr/bin/env python3
"""
Test suite for DDoS Detection System
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection_system import DDoSDetectionSystem
from feature_engineering import AdvancedFeatureEngineer
from models import StandardAutoencoder, LSTMAutoencoder
from config import AGG_PERIOD, EPOCHS, BATCH_SIZE


class TestDetectionSystem(unittest.TestCase):
    """Test cases for DDoS Detection System"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = DDoSDetectionSystem()
        
    def test_detection_system_initialization(self):
        """Test that detection system initializes correctly"""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.feature_engineer)
        self.assertEqual(len(self.detector.models), 0)  # No models loaded initially
        
    def test_feature_engineering(self):
        """Test feature engineering functionality"""
        # Create sample data
        sample_data = pd.DataFrame({
            'ts_bin': pd.date_range('2025-07-14', periods=10, freq='1Min'),
            'bytes': np.random.randint(1000, 10000, 10),
            'packets': np.random.randint(100, 1000, 10),
            'flows': np.random.randint(10, 100, 10),
            'srcAddr': np.random.randint(1, 50, 10),
            'dstAddr': np.random.randint(1, 50, 10),
            'srcPort': np.random.randint(1024, 65535, 10),
            'dstPort': np.random.randint(1024, 65535, 10),
            'proto': np.random.randint(1, 10, 10),
            'duration': np.random.uniform(0.1, 10.0, 10)
        })
        
        # Test feature extraction
        features = self.detector.feature_engineer.extract_features(sample_data)
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features), 0)
        
    def test_model_creation(self):
        """Test model creation and architecture"""
        # Test Standard Autoencoder
        input_dim = 20
        standard_ae = StandardAutoencoder(input_dim)
        self.assertIsNotNone(standard_ae)
        self.assertEqual(standard_ae.input_dim, input_dim)
        
        # Test LSTM Autoencoder
        lstm_ae = LSTMAutoencoder(n_features=input_dim)
        self.assertIsNotNone(lstm_ae)
        self.assertEqual(lstm_ae.n_features, input_dim)
        
    def test_data_preprocessing(self):
        """Test data preprocessing functionality"""
        # Create sample data
        sample_data = pd.DataFrame({
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10),
            'feature3': np.random.randn(10),
            'ts_bin': pd.date_range('2025-07-14', periods=10, freq='1Min')
        })
        
        # Test feature preparation
        X_scaled, feature_cols = self.detector.prepare_features(sample_data, fit_scaler=True)
        self.assertIsInstance(X_scaled, np.ndarray)
        self.assertIsInstance(feature_cols, list)
        self.assertEqual(X_scaled.shape[1], len(feature_cols))
        
    def test_anomaly_detection_interface(self):
        """Test anomaly detection interface"""
        # Create sample data
        sample_data = pd.DataFrame({
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10),
            'feature3': np.random.randn(10),
            'ts_bin': pd.date_range('2025-07-14', periods=10, freq='1Min')
        })
        
        # Test that the interface exists and is callable
        self.assertTrue(hasattr(self.detector, 'detect_anomalies_enhanced'))
        self.assertTrue(callable(self.detector.detect_anomalies_enhanced))
        
    def test_cache_functionality(self):
        """Test cache functionality"""
        # Test cache directory creation
        self.assertTrue(os.path.exists(self.detector.cache_dir))
        
        # Test cache info method - this might return None if no cache files
        cache_info = self.detector.get_cache_info()
        # Cache info can be None if no cache files exist, which is valid
        if cache_info is not None:
            self.assertIsInstance(cache_info, dict)
        
    def test_configuration_consistency(self):
        """Test that configuration is consistent"""
        from config import AGG_PERIOD, EPOCHS, BATCH_SIZE
        
        self.assertIsInstance(AGG_PERIOD, str)
        self.assertIsInstance(EPOCHS, int)
        self.assertIsInstance(BATCH_SIZE, int)
        self.assertGreater(EPOCHS, 0)
        self.assertGreater(BATCH_SIZE, 0)


class TestFeatureEngineering(unittest.TestCase):
    """Test cases for feature engineering"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.feature_engineer = AdvancedFeatureEngineer()
        
    def test_entropy_calculation(self):
        """Test entropy calculation"""
        # Create sample data with known entropy
        sample_data = pd.DataFrame({
            'ts_bin': ['2025-07-14 00:00:00'] * 10,
            'category': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'D', 'E']
        })
        
        entropy = self.feature_engineer._calculate_entropy(sample_data, 'category')
        self.assertIsInstance(entropy, pd.Series)
        self.assertEqual(len(entropy), 1)  # One time bin
        
    def test_feature_extraction_robustness(self):
        """Test feature extraction with edge cases"""
        # Test with empty data
        empty_data = pd.DataFrame()
        with self.assertRaises(Exception):
            self.feature_engineer.extract_features(empty_data)
            
        # Test with missing columns
        incomplete_data = pd.DataFrame({
            'ts_bin': pd.date_range('2025-07-14', periods=5, freq='1Min'),
            'bytes': np.random.randint(1000, 10000, 5)
        })
        with self.assertRaises(Exception):
            self.feature_engineer.extract_features(incomplete_data)


class TestModels(unittest.TestCase):
    """Test cases for ML models"""
    
    def test_standard_autoencoder_architecture(self):
        """Test standard autoencoder architecture"""
        input_dim = 15
        ae = StandardAutoencoder(input_dim)
        
        # Test model building
        model = ae.build_model()
        self.assertIsNotNone(model)
        
        # Test input shape
        self.assertEqual(model.input_shape[1], input_dim)
        self.assertEqual(model.output_shape[1], input_dim)
        
    def test_lstm_autoencoder_architecture(self):
        """Test LSTM autoencoder architecture"""
        input_dim = 15
        ae = LSTMAutoencoder(input_dim)
        
        # Test model building
        model = ae.build_model()
        self.assertIsNotNone(model)
        
        # Test sequence preparation
        sample_data = np.random.randn(20, input_dim)
        sequences = ae.prepare_sequences(sample_data, sequence_length=10)
        self.assertIsInstance(sequences, np.ndarray)
        self.assertEqual(sequences.shape[0], 11)  # 20 - 10 + 1
        self.assertEqual(sequences.shape[1], 10)  # sequence_length
        self.assertEqual(sequences.shape[2], input_dim)  # n_features


if __name__ == '__main__':
    unittest.main()
