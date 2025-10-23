"""
Monitor training progress in real-time
Run this in a separate terminal while train_comprehensive.py is running
"""

import pandas as pd
import time
import os

results_file = 'training_results.csv'

print("Monitoring training progress...")
print("Press Ctrl+C to stop\n")

last_count = 0

while True:
    try:
        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
            df['val_smape'] = df['val_smape'].astype(float)
            
            current_count = len(df)
            if current_count > last_count:
                last_count = current_count
                
                print(f"\n{'='*80}")
                print(f"Progress: {current_count} models tested")
                print(f"{'='*80}")
                
                # Show best models
                top_5 = df.nsmallest(5, 'val_smape')
                print("\nTop 5 Models So Far:")
                print(top_5[['full_name', 'val_smape', 'train_time']].to_string(index=False))
                
                # Show statistics
                print(f"\nStatistics:")
                print(f"  Best SMAPE: {df['val_smape'].min():.2f}%")
                print(f"  Worst SMAPE: {df['val_smape'].max():.2f}%")
                print(f"  Average SMAPE: {df['val_smape'].mean():.2f}%")
                print(f"  Models < 40%: {(df['val_smape'] < 40).sum()}")
                print(f"  Models < 30%: {(df['val_smape'] < 30).sum()}")
                print(f"  Models < 25%: {(df['val_smape'] < 25).sum()}")
        
        time.sleep(10)  # Check every 10 seconds
        
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(10)

