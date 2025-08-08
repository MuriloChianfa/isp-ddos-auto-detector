#!/usr/bin/env python3
"""
Utility functions for DDoS Detection System
"""

# Configure TensorFlow logging at the very beginning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'
os.environ['TF_ENABLE_XLA'] = '0'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow').disabled = True
logging.getLogger('absl').setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf


def configure_tensorflow_early():
    """Configure TensorFlow settings before any models are imported"""
    try:
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            try:
                for gpu in gpu_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU memory growth configured")
            except RuntimeError:
                pass
        
        print("TensorFlow early configuration completed")
        return True
    except Exception as e:
        print(f"Early TensorFlow configuration failed: {e}")
        return False


def configure_gpu_for_training():
    """Configure GPU for training - only call when training new models"""
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print("Configuring GPU for training...")
        try:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU configured for training")
            return True
        except RuntimeError:
            print("GPU already configured")
            return True
    else:
        print("No GPUs detected, using CPU for training")
        return False


def configure_cpu_for_inference():
    """Configure CPU for inference and other operations"""
    print("Configuring CPU for inference...")
    try:
        tf.config.set_visible_devices([], 'GPU')
        print("CPU configured for inference")
        return True
    except RuntimeError as e:
        if "Visible devices cannot be modified after being initialized" in str(e):
            print("TensorFlow already initialized - cannot change visible devices")
            print("CPU configuration skipped (TensorFlow already running)")
            print("This is normal and doesn't affect functionality")
            return False
        else:
            print(f"Failed to configure CPU: {e}")
            return False


def check_gpu_availability():
    """Check if GPU is available for training"""
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print("GPU acceleration available for training")
        return True
    else:
        print("No GPUs detected, will use CPU for all operations")
        return False

