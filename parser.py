#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incremental CSV to Parquet converter for live data folders.
- Never deletes original files
- Tracks processed files by checking existing parquet files
- Handles new files appearing over time
"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm

class IncrementalConverter:
    def __init__(self, source_dir='spikehistory', target_dir='processed/parquet'):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
    
    def get_processed_files(self):
        """Get set of already processed files by checking existing parquet files."""
        existing_parquets = set(p.stem + '.csv' for p in self.target_dir.glob('*.parquet'))
        return existing_parquets
    
    def convert_csv_to_parquet(self, csv_path):
        """Convert single CSV to Parquet."""
        df = pd.read_csv(csv_path)
        
        # Add metadata from filename
        parts = csv_path.stem.split('_')
        if len(parts) >= 2:
            df['symbol'] = parts[0]
            df['type'] = parts[1]
        
        # Convert timestamp if exists
        if 'timestamp_ms' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
        
        # Save as parquet
        output_path = self.target_dir / f"{csv_path.stem}.parquet"
        df.to_parquet(output_path, compression='snappy')
        return output_path
    
    def process_new_files(self, batch_size=None):
        """Process only new CSV files that haven't been converted yet."""
        processed = self.get_processed_files()
        all_csvs = list(self.source_dir.glob('*.csv'))
        new_files = [f for f in all_csvs if f.name not in processed]
        
        print(f"Status: {len(processed)} already done, {len(new_files)} new files")
        
        if not new_files:
            print("All files already converted")
            return
        
        # Process in batches if specified
        if batch_size:
            new_files = new_files[:batch_size]
            print(f"Processing batch of {len(new_files)} files...")
        else:
            print(f"Converting {len(new_files)} new files...")
        
        successful = 0
        failed = 0
        
        for csv_file in tqdm(new_files, desc="Converting"):
            try:
                # Check if parquet already exists (double-check)
                output_path = self.target_dir / f"{csv_file.stem}.parquet"
                if not output_path.exists():
                    self.convert_csv_to_parquet(csv_file)
                    successful += 1
                else:
                    print(f"\nSkipping {csv_file.name}: parquet already exists")
                    
            except Exception as e:
                failed += 1
                if failed <= 5:  # Only show first 5 errors
                    print(f"\nWarning: Skipped {csv_file.name}: {e}")
        
        print(f"\nDone! Successfully converted: {successful}, Failed: {failed}")
        print(f"Total processed so far: {len(self.get_processed_files())}")

def main():
    converter = IncrementalConverter()
    converter.process_new_files()

if __name__ == "__main__":
    main()