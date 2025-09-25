import pandas as pd
import glob
import os
from datetime import datetime
from .cache import DataCache
from typing import Iterator, Dict, Optional, Tuple
from multiprocessing import Pool, cpu_count
import functools


class NetworkDataLoader:
    def __init__(self, dataset_config=None, use_cache=True, batch_size=1000, max_processes=None):
        """
        Initialize NetworkDataLoader with dataset configuration.
        
        Args:
            dataset_config (dict): Dataset configuration containing 'path' and 'patterns'
            use_cache (bool): Whether to use caching
            batch_size (int): Number of records to process at once
            max_processes (int): Maximum number of processes to use for parallel processing. 
        """
        self.data_path = dataset_config['path']
        self.patterns = dataset_config['patterns']
        self.use_cache = use_cache
        self.cache = DataCache() if use_cache else None
        self.batch_size = batch_size
        self.max_processes = max_processes
        self._file_metadata = {}
        
    def _get_file_metadata(self, split_name: str) -> list:
        """Get metadata about files for a split without loading the data"""
        if split_name in self._file_metadata:
            return self._file_metadata[split_name]
            
        pattern = self.patterns[split_name]
        
        # Handle both single patterns and lists of patterns
        if isinstance(pattern, list):
            csv_files = []
            for p in pattern:
                csv_files.extend(glob.glob(os.path.join(self.data_path, p)))
            csv_files = sorted(csv_files)
        else:
            csv_files = sorted(glob.glob(os.path.join(self.data_path, pattern)))
        
        file_info = []
        for file in csv_files:
            if os.path.getsize(file) == 0:
                continue
                
            filename = os.path.basename(file)
            timestamp_str = filename.replace('nfcapd.', '').replace('.csv', '')
            
            try:
                timestamp = pd.to_datetime(timestamp_str, format='%Y%m%d%H%M')
                timestamp_utc = timestamp.tz_localize('UTC')
                file_info.append({
                    'path': file,
                    'timestamp': timestamp_utc,
                    'timestamp_str': timestamp_str
                })
            except Exception:
                continue
                
        self._file_metadata[split_name] = file_info
        return file_info
    
    def get_data_generator(self, split_name: str, parallel=True) -> Iterator[pd.DataFrame]:
        """
        Generator that yields data chunks for a specific split with parallel processing
        
        Args:
            split_name (str): Name of the data split ('train', 'validation', 'test')
            parallel (bool): Whether to use parallel processing for file loading
            
        Yields:
            pd.DataFrame: Chunks of network data
        """
        file_info = self._get_file_metadata(split_name)
        
        print(f"Starting generator for {split_name} set with {len(file_info)} files")
        
        if parallel and len(file_info) > 1:
            # Use parallel processing for multiple files
            if self.max_processes is not None:
                num_processes = min(cpu_count(), len(file_info), self.max_processes)
            else:
                num_processes = min(cpu_count(), len(file_info), 48)

            print(f"  Using {num_processes} parallel processes")
            
            # Process files in chunks to avoid memory issues
            chunk_size = max(1, len(file_info) // num_processes)
            
            with Pool(num_processes) as pool:
                # Create chunks of file_info
                file_chunks = [file_info[i:i + chunk_size] for i in range(0, len(file_info), chunk_size)]
                
                for chunk_idx, chunk in enumerate(file_chunks):
                    print(f"  Processing chunk {chunk_idx + 1}/{len(file_chunks)} with {len(chunk)} files")
                    
                    # Process chunk in parallel
                    results = pool.map(self._process_file, chunk)
                    
                    # Yield results that are not None
                    for df in results:
                        if df is not None:
                            yield df
        else:
            # Sequential processing (fallback or single file)
            for i, info in enumerate(file_info):
                if i % 10 == 0:
                    print(f"  Loading file {i+1}/{len(file_info)}: {os.path.basename(info['path'])}")
                
                df = self._process_file(info)
                if df is not None:
                    yield df
    
    def _process_file(self, info):
        """Process a single file - used for both sequential and parallel processing"""
        try:
            df = pd.read_csv(info['path'])
            
            if df.empty:
                return None
                
            df['file_timestamp'] = info['timestamp']
            return df
                
        except Exception as e:
            print(f"    Error processing file {os.path.basename(info['path'])}: {str(e)}")
            return None
    
    def get_data_info(self) -> Dict[str, Dict]:
        """Get information about datasets without loading the data"""
        info = {}
        
        for split_name in self.patterns.keys():
            file_info = self._get_file_metadata(split_name)
            
            # Estimate total records by sampling a few files
            total_records_estimate = 0
            sample_files = file_info[:min(3, len(file_info))]
            
            if sample_files:
                sample_record_counts = []
                for info_item in sample_files:
                    try:
                        df_sample = pd.read_csv(info_item['path'])
                        sample_record_counts.append(len(df_sample))
                    except Exception:
                        continue
                
                if sample_record_counts:
                    avg_records = sum(sample_record_counts) / len(sample_record_counts)
                    total_records_estimate = int(avg_records * len(file_info))
            
            info[split_name] = {
                'file_count': len(file_info),
                'estimated_records': total_records_estimate,
                'time_range': (
                    min(item['timestamp'] for item in file_info) if file_info else None,
                    max(item['timestamp'] for item in file_info) if file_info else None
                )
            }
        
        return info
        
    def load_network_data_by_day(self):
        """Load network traffic data organized by days for train/validation/test split"""
        print("Warning: This method loads all data into memory.")
        print("Consider using get_data_generator() for memory-efficient processing.")
        
        # Try to load from cache first
        if self.use_cache and self.cache:
            cached_datasets = self.cache.get_datasets_cache(self.data_path)
            if cached_datasets is not None:
                print("Using cached datasets!")
                return cached_datasets

        print("Loading network traffic data from CSV files...")
        print(f"Dataset path: {self.data_path}")
        
        datasets = {}
        
        for split_name in self.patterns.keys():
            data_frames = []
            
            for batch in self.get_data_generator(split_name):
                data_frames.append(batch)
            
            if data_frames:
                combined_data = pd.concat(data_frames, ignore_index=True)
                datasets[split_name] = combined_data
                print(f"  {split_name.capitalize()} set: {len(combined_data)} records")
        
        # Save to cache for next time
        if self.use_cache and self.cache:
            self.cache.save_datasets_cache(datasets, self.data_path)
        
        return datasets
