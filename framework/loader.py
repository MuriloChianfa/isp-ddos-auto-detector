import pandas as pd
import glob
import os
from datetime import datetime


class NetworkDataLoader:
    def __init__(self, data_path="./datasets/ramfs/"):
        self.data_path = data_path
        
    def load_network_data_by_day(self):
        """Load network traffic data organized by days for train/validation/test split"""
        
        day_patterns = {
            'train': 'nfcapd.20250714*.csv',
            'validation': 'nfcapd.20250715*.csv',
            'test': 'nfcapd.2025071[67]*.csv'
        }
        
        datasets = {}
        
        for split_name, pattern in day_patterns.items():
            csv_files = sorted(glob.glob(os.path.join(self.data_path, pattern)))
            data_frames = []
            print(f"Loading {len(csv_files)} CSV files for {split_name} set...")
            
            for i, file in enumerate(csv_files):
                if i % 50 == 0:
                    print(f"  Processing {split_name} file {i+1}/{len(csv_files)}: {os.path.basename(file)}")
                
                df = pd.read_csv(file)
                filename = os.path.basename(file)
                timestamp_str = filename.replace('nfcapd.', '').replace('.csv', '')
                df['file_timestamp'] = pd.to_datetime(timestamp_str, format='%Y%m%d%H%M')
                data_frames.append(df)
            
            combined_data = pd.concat(data_frames, ignore_index=True)
            datasets[split_name] = combined_data
            print(f"  {split_name.capitalize()} set: {len(combined_data)} records from {len(combined_data['file_timestamp'].unique())} time windows")
        
        return datasets
