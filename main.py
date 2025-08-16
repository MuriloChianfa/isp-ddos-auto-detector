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
        combined_data = pd.concat(data_frames, ignore_index=True)
        datasets[split_name] = combined_data
        print(f"  {split_name.capitalize()} set: {len(combined_data)} records from {len(combined_data['file_timestamp'].unique())} time windows")
    
    return datasets

# Load the datasets split by days
datasets = load_network_data_by_day()

# Extract individual datasets
train_data = datasets['train']
validation_data = datasets['validation'] 
test_data = datasets['test']

# 2. Advanced Feature Engineering for Network Anomaly Detection
def calculate_port_entropy(ports):
    """Calculate entropy of port distribution"""
    _, counts = np.unique(ports, return_counts=True)
    return entropy(counts, base=2)

def prepare_advanced_features(df):
    """Extract comprehensive features for network anomaly detection"""
    time_grouped = df.groupby('file_timestamp')
    
    features_list = []
    timestamps = []
    
    for timestamp, group in time_grouped:
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
        feature_row['avg_packet_size'] = group['bytes'].sum() / group['packets'].sum()
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
    features_dict[split_name] = prepare_advanced_features(data)

# Extract feature matrices
train_features = features_dict['train']
validation_features = features_dict['validation']
test_features = features_dict['test']

# Prepare training data (remove timestamp for model training)
feature_cols = [col for col in train_features.columns if col != 'timestamp']
training_features = train_features[feature_cols].copy()

# Remove any infinite or NaN values
training_features = training_features.replace([np.inf, -np.inf], np.nan).fillna(0)

# Use training data for fitting the scaler (normal traffic only)
normal_data = training_features.copy()

# 3. Preprocessing: Scale the data using training data
scaler = MinMaxScaler()
scaled_train_data = scaler.fit_transform(normal_data)

# Prepare validation data
validation_training_features = validation_features[feature_cols].copy()
validation_training_features = validation_training_features.replace([np.inf, -np.inf], np.nan).fillna(0)
scaled_validation_data = scaler.transform(validation_training_features)

# Prepare test data
test_training_features = test_features[feature_cols].copy()
test_training_features = test_training_features.replace([np.inf, -np.inf], np.nan).fillna(0)
scaled_test_data = scaler.transform(test_training_features)

# Define an Enhanced Autoencoder Model
input_dim = scaled_train_data.shape[1]
latent_dim = 8

print(f"Building enhanced autoencoder for {input_dim} features...")

# Enhanced autoencoder with dropout and better architecture
autoencoder = keras.Sequential([
    keras.layers.Input(shape=(input_dim,)),  # Input layer
    
    # Encoder
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    
    keras.layers.Dense(latent_dim, activation='relu'),  # Bottleneck (encoder)
    
    # Decoder
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    
    keras.layers.Dense(input_dim, activation='linear')  # Output layer
])

# Use a more sophisticated optimizer with learning rate scheduling
# Choose one of the two approaches below:

# APPROACH 1: ExponentialDecay (automatic, predictable decay)
initial_learning_rate = 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100,
    decay_rate=0.96,
    staircase=True)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

# APPROACH 2: ReduceLROnPlateau (adaptive, based on validation loss)
# optimizer = keras.optimizers.Adam(learning_rate=0.001)

autoencoder.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Define callbacks for better training
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
]

print("Training enhanced autoencoder...")
# Train the Enhanced Autoencoder with more epochs
history = autoencoder.fit(scaled_train_data, scaled_train_data,
                          epochs=100,  # Increased epochs
                          batch_size=64,  # Larger batch size for better gradient estimates
                          validation_data=(scaled_validation_data, scaled_validation_data),
                          callbacks=callbacks,
                          verbose=1)

# Visualize training history
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE', color='blue')
plt.plot(history.history['val_mae'], label='Validation MAE', color='red')
plt.title('Model MAE During Training')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Training completed after {len(history.history['loss'])} epochs")
print(f"Final training loss: {history.history['loss'][-1]:.6f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
print(f"Final training MAE: {history.history['mae'][-1]:.6f}")
print(f"Final validation MAE: {history.history['val_mae'][-1]:.6f}")

# Anomaly Detection on all datasets
def detect_anomalies(scaled_data, features_df, dataset_name):
    """Detect anomalies for a given dataset"""
    reconstructions = autoencoder.predict(scaled_data)
    mse = np.mean(np.power(scaled_data - reconstructions, 2), axis=1)
    
    return reconstructions, mse, features_df.copy()

# Detect anomalies for each dataset
results = {}
all_mse_values = []

train_reconstructions, train_mse, train_results_df = detect_anomalies(scaled_train_data, train_features, "training")
results['train'] = (train_reconstructions, train_mse, train_results_df)
all_mse_values.extend(train_mse)

val_reconstructions, val_mse, val_results_df = detect_anomalies(scaled_validation_data, validation_features, "validation")
results['validation'] = (val_reconstructions, val_mse, val_results_df)
all_mse_values.extend(val_mse)

test_reconstructions, test_mse, test_results_df = detect_anomalies(scaled_test_data, test_features, "test")
results['test'] = (test_reconstructions, test_mse, test_results_df)
all_mse_values.extend(test_mse)

# 7. Set a Threshold and Identify Anomalies (based on ALL normal traffic)
# Combine MSE from both training and validation (both contain only normal traffic)
normal_mse_values = list(train_mse) + list(val_mse)
normal_mse_values = np.array(normal_mse_values)

# Calculate multiple threshold strategies
normal_mse_mean = np.mean(normal_mse_values)
normal_mse_std = np.std(normal_mse_values)

print("\n" + "="*60)
print("THRESHOLD ANALYSIS")
print("="*60)
print(f"Normal Traffic Statistics:")
print(f"  Mean MSE: {normal_mse_mean:.6f}")
print(f"  Std MSE:  {normal_mse_std:.6f}")
print(f"  Min MSE:  {np.min(normal_mse_values):.6f}")
print(f"  Max MSE:  {np.max(normal_mse_values):.6f}")
print(f"  Median:   {np.median(normal_mse_values):.6f}")

# Different threshold strategies
percentile_95 = np.percentile(normal_mse_values, 95)
percentile_99 = np.percentile(normal_mse_values, 99)
percentile_999 = np.percentile(normal_mse_values, 99.9)
mean_plus_std = normal_mse_mean + normal_mse_std
mean_plus_2std = normal_mse_mean + 2 * normal_mse_std
mean_plus_3std = normal_mse_mean + 3 * normal_mse_std

thresholds = {
    '95th Percentile': percentile_95,
    '99th Percentile': percentile_99,
    '99.9th Percentile': percentile_999,
    'Mean + 1σ': mean_plus_std,
    'Mean + 2σ': mean_plus_2std,
    'Mean + 3σ': mean_plus_3std
}

print(f"\nThreshold Options:")
for name, thresh in thresholds.items():
    print(f"  {name:20s}: {thresh:.6f}")

threshold = mean_plus_2std
print(f"\nSelected Threshold: {threshold:.6f}")
print("="*60)

# Apply threshold to all datasets
for split_name, (reconstructions, mse, features_df) in results.items():
    anomalies = mse > threshold
    features_df['reconstruction_error'] = mse
    features_df['is_anomaly'] = anomalies
    results[split_name] = (reconstructions, mse, features_df)

# Compare different threshold strategies on test data
print("\n" + "="*60)
print("THRESHOLD COMPARISON ON TEST DATA")
print("="*60)
test_mse = results['test'][1]
for name, thresh in thresholds.items():
    test_anomalies = np.sum(test_mse > thresh)
    test_rate = (test_anomalies / len(test_mse)) * 100
    print(f"{name:20s}: {test_anomalies:3d}/466 ({test_rate:5.2f}%) - Threshold: {thresh:.6f}")
print("="*60)

for split_name, (reconstructions, mse, features_df) in results.items():
    # Calculate metrics for this dataset
    mae = np.mean(np.abs(scaled_train_data if split_name == 'train' else 
                        (scaled_validation_data if split_name == 'validation' else scaled_test_data) - reconstructions))
    mse_value = np.mean(mse)
    rmse = np.sqrt(mse_value)
    
    num_anomalies = np.sum(features_df['is_anomaly'])
    total_samples = len(features_df)
    anomaly_percentage = (num_anomalies / total_samples) * 100
    
    print(f"Dataset scores: {split_name}")
    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  Mean Squared Error (MSE):  {mse_value:.6f}")
    print(f"  Root Mean Squared Error:   {rmse:.6f}")
    print(f"  Anomalies detected:        {num_anomalies}/{total_samples} ({anomaly_percentage:.2f}%)")
    print(f"  Threshold used:            {threshold:.6f}")
    print()

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

test_timestamps = test_data['timestamp']
test_anomaly_scores = test_data['reconstruction_error']
test_anomalies_mask = test_data['is_anomaly']

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
           label=f'MAE: {mae_anomaly_score:.6f}')
plt.axhline(y=mse_anomaly_score, color='orange', linestyle='-.', alpha=0.7, linewidth=1.5, 
           label=f'MSE: {mse_anomaly_score:.6f}')
plt.axhline(y=mse_plus_std_anomaly_score, color='brown', linestyle='-.', alpha=0.7, linewidth=1.5, 
           label=f'MSE + STD: {mse_plus_std_anomaly_score:.6f}')

# Print summary statistics
print(f"Normal Traffic (Train + Validation) Statistics:")
print(f"  Mean Reconstruction Error: {mse_anomaly_score:.6f}")
print(f"  MAE (Normal Traffic):      {mae_anomaly_score:.6f}")
print(f"  Standard Deviation:        {np.std(normal_anomaly_scores):.6f}")
print(f"  Anomaly Threshold:         {threshold:.6f}")
print(f"\nTest Set Results:")
print(f"  Total test windows:        {len(test_data)}")
print(f"  Detected anomalies:        {np.sum(test_data['is_anomaly'])}")
print(f"  Anomaly rate:              {(np.sum(test_data['is_anomaly'])/len(test_data)*100):.2f}%")

plt.xlabel('Time', fontsize=12)
plt.ylabel('Anomaly Score (Reconstruction Error)', fontsize=12)
plt.title('Network Traffic Anomaly Detection', fontsize=14, fontweight='bold')

# Configure x-axis to show hourly ticks
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=1))

plt.legend(fontsize=9, loc='upper right')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
