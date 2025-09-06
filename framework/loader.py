import pandas as pd
import glob
import os
from datetime import datetime
from .cache import DataCache


class NetworkDataLoader:
    def __init__(self, dataset_config=None, use_cache=True):
        """
        Initialize NetworkDataLoader with dataset configuration.
        
        Args:
            dataset_config (dict): Dataset configuration containing 'path' and 'patterns'
            use_cache (bool): Whether to use caching
        """
        self.data_path = dataset_config['path']
        self.patterns = dataset_config['patterns']

        self.use_cache = use_cache
        self.cache = DataCache() if use_cache else None
        
    def load_network_data_by_day(self):
        """Load network traffic data organized by days for train/validation/test split"""
        
        # Try to load from cache first
        if self.use_cache and self.cache:
            cached_datasets = self.cache.get_datasets_cache(self.data_path)
            if cached_datasets is not None:
                print("Using cached datasets!")
                return cached_datasets

        print("Loading network traffic data from CSV files...")
        print(f"Dataset path: {self.data_path}")
        
        datasets = {}
        
        for split_name, pattern in self.patterns.items():
            csv_files = sorted(glob.glob(os.path.join(self.data_path, pattern)))
            data_frames = []
            print(f"Loading {len(csv_files)} CSV files for {split_name} set (pattern: {pattern})...")
            
            if len(csv_files) == 0:
                print(f"Warning: No files found for {split_name} set with pattern {pattern}")
                continue
            
            for i, file in enumerate(csv_files):
                if i % 50 == 0:
                    print(f"  Processing {split_name} file {i+1}/{len(csv_files)}: {os.path.basename(file)}")
                
                try:
                    # Check if file is empty
                    if os.path.getsize(file) == 0:
                        print(f"    Warning: Skipping empty file {os.path.basename(file)}")
                        continue
                    
                    df = pd.read_csv(file)
                    
                    # Check if dataframe is empty
                    if df.empty:
                        print(f"    Warning: Skipping file with no data {os.path.basename(file)}")
                        continue
                    
                    filename = os.path.basename(file)
                    timestamp_str = filename.replace('nfcapd.', '').replace('.csv', '')
                    df['file_timestamp'] = pd.to_datetime(timestamp_str, format='%Y%m%d%H%M')
                    data_frames.append(df)
                    
                except pd.errors.EmptyDataError:
                    print(f"    Warning: Skipping empty or malformed file {os.path.basename(file)}")
                    continue
                except Exception as e:
                    print(f"    Error processing file {os.path.basename(file)}: {str(e)}")
                    continue
            
            if data_frames:
                combined_data = pd.concat(data_frames, ignore_index=True)
                datasets[split_name] = combined_data
                print(f"  {split_name.capitalize()} set: {len(combined_data)} records from {len(combined_data['file_timestamp'].unique())} time windows")
            else:
                print(f"  Warning: No valid data found for {split_name} set")
        
        # Save to cache for next time
        if self.use_cache and self.cache:
            self.cache.save_datasets_cache(datasets, self.data_path)
        
        return datasets
