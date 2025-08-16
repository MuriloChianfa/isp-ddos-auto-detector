import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import os
from datetime import datetime
from scipy.stats import entropy

# 1. Load Network Traffic Data with Train/Validation/Test Split
def load_network_data_by_day(data_path="./datasets/ramfs/"):
    """Load network traffic data organized by days for train/validation/test split"""
    
    # Define day patterns (include day 17 in test dataset)
    day_patterns = {
        'train': 'nfcapd.20250714*.csv',      # Day 14 for training
        'validation': 'nfcapd.20250715*.csv', # Day 15 for validation
        'test': 'nfcapd.2025071[67]*.csv'     # Day 16 and 17 for testing
    }
    
    datasets = {}
    
    for split_name, pattern in day_patterns.items():
        csv_files = sorted(glob.glob(os.path.join(data_path, pattern)))
        
        if not csv_files:
            print(f"Warning: No files found for {split_name} with pattern {pattern}")
            continue
            
        data_frames = []
        print(f"Loading {len(csv_files)} CSV files for {split_name} set...")
        
        for i, file in enumerate(csv_files):
            if i % 50 == 0:
                print(f"  Processing {split_name} file {i+1}/{len(csv_files)}: {os.path.basename(file)}")
            
            df = pd.read_csv(file)
            # Add timestamp information for timeline analysis
            filename = os.path.basename(file)
            timestamp_str = filename.replace('nfcapd.', '').replace('.csv', '')
            df['file_timestamp'] = pd.to_datetime(timestamp_str, format='%Y%m%d%H%M')
            data_frames.append(df)
        
        # Combine data for this split
        if data_frames:
            combined_data = pd.concat(data_frames, ignore_index=True)
            datasets[split_name] = combined_data
            print(f"  {split_name.capitalize()} set: {len(combined_data)} records from {len(combined_data['file_timestamp'].unique())} time windows")
    
    return datasets

# Load the datasets split by days
datasets = load_network_data_by_day()

# Extract individual datasets
train_data = datasets.get('train')
validation_data = datasets.get('validation') 
test_data = datasets.get('test')

# 2. Advanced Feature Engineering for Network Anomaly Detection
def calculate_port_entropy(ports):
    """Calculate entropy of port distribution"""
    if len(ports) == 0:
        return 0
    _, counts = np.unique(ports, return_counts=True)
    return entropy(counts, base=2)

def prepare_advanced_features(df):
    """Extract comprehensive features for network anomaly detection"""
    time_grouped = df.groupby('file_timestamp')
    
    features_list = []
    timestamps = []
    
    for timestamp, group in time_grouped:
        if len(group) == 0:
            continue
            
        feature_row = {}
        timestamps.append(timestamp)
        
        # Basic flow statistics
        feature_row['total_flows'] = len(group)
        feature_row['total_packets'] = group['packets'].sum()
        feature_row['total_bytes'] = group['bytes'].sum()
        feature_row['avg_duration'] = group['duration'].mean()
        
        # Traffic rate features (key for DDoS detection)
        time_span = 300  # 5 minutes in seconds (typical nfcapd collection interval)
        feature_row['packet_rate'] = feature_row['total_packets'] / time_span  # packets per second
        feature_row['bit_rate'] = (feature_row['total_bytes'] * 8) / time_span  # bits per second
        feature_row['flow_rate'] = feature_row['total_flows'] / time_span  # flows per second
        
        # Packet size statistics
        feature_row['avg_packet_size'] = group['bytes'].sum() / (group['packets'].sum() + 1)
        feature_row['packets_per_flow'] = group['packets'].mean()
        feature_row['bytes_per_flow'] = group['bytes'].mean()
        
        # Port entropy (diversity indicators)
        feature_row['src_port_entropy'] = calculate_port_entropy(group['srcPort'].values)
        feature_row['dst_port_entropy'] = calculate_port_entropy(group['dstPort'].values)
        
        # Protocol distribution
        proto_counts = group['proto'].value_counts()
        feature_row['tcp_ratio'] = proto_counts.get(6, 0) / len(group)  # TCP
        feature_row['udp_ratio'] = proto_counts.get(17, 0) / len(group)  # UDP
        feature_row['icmp_ratio'] = proto_counts.get(1, 0) / len(group)  # ICMP
        
        # IP diversity (potential for DDoS detection)
        feature_row['unique_src_ips'] = group['srcAddr'].nunique()
        feature_row['unique_dst_ips'] = group['dstAddr'].nunique()
        feature_row['src_ip_entropy'] = calculate_port_entropy(group['srcAddr'].values)
        feature_row['dst_ip_entropy'] = calculate_port_entropy(group['dstAddr'].values)
        
        # Connection patterns
        feature_row['avg_src_ports_per_ip'] = group.groupby('srcAddr')['srcPort'].nunique().mean()
        feature_row['avg_dst_ports_per_ip'] = group.groupby('dstAddr')['dstPort'].nunique().mean()
        
        # Traffic volume distribution
        feature_row['max_bytes_per_flow'] = group['bytes'].max()
        feature_row['std_bytes_per_flow'] = group['bytes'].std()
        feature_row['max_packets_per_flow'] = group['packets'].max()
        feature_row['std_packets_per_flow'] = group['packets'].std()
        
        features_list.append(feature_row)
    
    features_df = pd.DataFrame(features_list)
    features_df['timestamp'] = timestamps
    
    return features_df

# Prepare advanced features for each dataset
features_dict = {}

for split_name, data in datasets.items():
    if data is not None:
        features_dict[split_name] = prepare_advanced_features(data)

# Extract feature matrices
train_features = features_dict.get('train')
validation_features = features_dict.get('validation')
test_features = features_dict.get('test')

# Prepare training data (remove timestamp for model training)
if train_features is not None:
    feature_cols = [col for col in train_features.columns if col != 'timestamp']
    training_features = train_features[feature_cols].copy()
    
    # Remove any infinite or NaN values
    training_features = training_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Use training data for fitting the scaler (normal traffic only)
    normal_data = training_features.copy()
else:
    print("Error: No training data available!")
    exit(1)

# 3. Preprocessing: Scale the data using training data
scaler = MinMaxScaler()
scaled_train_data = scaler.fit_transform(normal_data)

# Prepare validation data if available
scaled_validation_data = None
if validation_features is not None:
    validation_training_features = validation_features[feature_cols].copy()
    validation_training_features = validation_training_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    scaled_validation_data = scaler.transform(validation_training_features)

# Prepare test data if available  
scaled_test_data = None
test_training_features = None
if test_features is not None:
    test_training_features = test_features[feature_cols].copy()
    test_training_features = test_training_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    scaled_test_data = scaler.transform(test_training_features)

# Define the Autoencoder Model
input_dim = scaled_train_data.shape[1]
latent_dim = 8

autoencoder = keras.Sequential([
    keras.layers.Input(shape=(input_dim,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(latent_dim, activation='relu'),  # Bottleneck (encoder)
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(input_dim, activation='linear')  # Output layer (decoder)
])

autoencoder.compile(optimizer='adam', loss='mse')

# Train the Autoencoder
if scaled_validation_data is not None:
    history = autoencoder.fit(scaled_train_data, scaled_train_data,
                              epochs=30,
                              batch_size=32,
                              validation_data=(scaled_validation_data, scaled_validation_data),
                              verbose=1)
else:
    history = autoencoder.fit(scaled_train_data, scaled_train_data,
                              epochs=30,
                              batch_size=32,
                              validation_split=0.1,
                              verbose=1)

# Anomaly Detection on all datasets
def detect_anomalies(scaled_data, features_df, dataset_name):
    """Detect anomalies for a given dataset"""
    if scaled_data is None or features_df is None:
        return None, None, None
        
    reconstructions = autoencoder.predict(scaled_data)
    mse = np.mean(np.power(scaled_data - reconstructions, 2), axis=1)
    
    return reconstructions, mse, features_df.copy()

# Detect anomalies for each dataset
results = {}
all_mse_values = []

train_reconstructions, train_mse, train_results_df = detect_anomalies(scaled_train_data, train_features, "training")
if train_mse is not None:
    results['train'] = (train_reconstructions, train_mse, train_results_df)
    all_mse_values.extend(train_mse)

val_reconstructions, val_mse, val_results_df = detect_anomalies(scaled_validation_data, validation_features, "validation")
if val_mse is not None:
    results['validation'] = (val_reconstructions, val_mse, val_results_df)
    all_mse_values.extend(val_mse)

test_reconstructions, test_mse, test_results_df = detect_anomalies(scaled_test_data, test_features, "test")
if test_mse is not None:
    results['test'] = (test_reconstructions, test_mse, test_results_df)
    all_mse_values.extend(test_mse)

# 7. Set a Threshold and Identify Anomalies (based on ALL normal traffic)
# Combine MSE from both training and validation (both contain only normal traffic)
normal_mse_values = []
if train_mse is not None:
    normal_mse_values.extend(train_mse)
if val_mse is not None:
    normal_mse_values.extend(val_mse)

if len(normal_mse_values) == 0:
    print("Error: No normal traffic data available for threshold calculation!")
    exit(1)

normal_mse_values = np.array(normal_mse_values)

# Calculate threshold using MSE + StdDev of normal traffic reconstruction errors
normal_mse_mean = np.mean(normal_mse_values)
normal_mse_std = np.std(normal_mse_values)
threshold = normal_mse_mean + normal_mse_std

# Apply threshold to all datasets
for split_name, (reconstructions, mse, features_df) in results.items():
    anomalies = mse > threshold
    features_df['reconstruction_error'] = mse
    features_df['is_anomaly'] = anomalies
    results[split_name] = (reconstructions, mse, features_df)

print()

# Combine all results for visualization
all_features = []
all_timestamps = []
all_mse = []
all_anomalies = []
all_datasets = []

for split_name, (_, mse, features_df) in results.items():
    all_features.append(features_df)
    all_timestamps.extend(features_df['timestamp'])
    all_mse.extend(mse)
    all_anomalies.extend(features_df['is_anomaly'])
    all_datasets.extend([split_name] * len(features_df))

# Create combined dataframe for visualization
combined_features = pd.concat(all_features, ignore_index=True)
combined_features['dataset'] = all_datasets

# Analyze temporal distribution of anomalies in test set
if 'test' in results:
    test_features_df = results['test'][2]
    test_anomalies = test_features_df[test_features_df['is_anomaly']]

# Anomaly Detection Visualization
timestamps = combined_features['timestamp']
anomalies = combined_features['is_anomaly']
anomaly_times = timestamps[anomalies]

# Enhanced Anomaly Detection Chart - Test Dataset Only
plt.figure(figsize=(20, 10))

# Filter data to show only test dataset
test_mask = combined_features['dataset'] == 'test'
test_data = combined_features[test_mask]

if len(test_data) > 0:
    test_timestamps = test_data['timestamp']
    test_anomaly_scores = test_data['reconstruction_error']
    test_anomalies_mask = test_data['is_anomaly']

    if np.sum(test_anomalies_mask) > 0:
        anomaly_timestamps = test_timestamps[test_anomalies_mask]
        for i, anomaly_time in enumerate(anomaly_timestamps):
            # Assuming each data point represents a 5-minute window
            window_start = anomaly_time - pd.Timedelta(minutes=2.5)
            window_end = anomaly_time + pd.Timedelta(minutes=2.5)
            plt.axvspan(window_start, window_end, alpha=0.25, color='red', 
                       label='Detected Anomaly Period' if i == 0 else "", zorder=1)

    plt.plot(test_timestamps, test_anomaly_scores, 
            color='green', alpha=0.8, linewidth=1.5, 
            label='Anomaly Scores', zorder=5)

    # Add main anomaly detection threshold line
    plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.8, linewidth=2, 
               label=f'Anomaly Threshold: {threshold:.6f}')

    # Calculate statistical reference lines for normal traffic (training + validation only)
    normal_mask = (combined_features['dataset'] == 'train') | (combined_features['dataset'] == 'validation')
    normal_anomaly_scores = combined_features[normal_mask]['reconstruction_error']

    mae_anomaly_score = np.mean(np.abs(normal_anomaly_scores))
    mse_anomaly_score = np.mean(normal_anomaly_scores)
    mse_plus_std_anomaly_score = np.mean(normal_anomaly_scores) + np.std(normal_anomaly_scores)

    # Add statistical reference lines to the chart
    plt.axhline(y=mae_anomaly_score, color='purple', linestyle='-.', alpha=0.7, linewidth=1.5, 
               label=f'MAE (Normal): {mae_anomaly_score:.6f}')
    plt.axhline(y=mse_anomaly_score, color='orange', linestyle='-.', alpha=0.7, linewidth=1.5, 
               label=f'MSE (Normal): {mse_anomaly_score:.6f}')
    plt.axhline(y=mse_plus_std_anomaly_score, color='brown', linestyle='-.', alpha=0.7, linewidth=1.5, 
               label=f'MSE + STD (Normal): {mse_plus_std_anomaly_score:.6f}')

    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Anomaly Score (Reconstruction Error)', fontsize=12)
    plt.title('Network Traffic Anomaly Detection', fontsize=14, fontweight='bold')

    # Configure x-axis to show hourly ticks
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Every hour
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=1))

    plt.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("No test data available for visualization")
