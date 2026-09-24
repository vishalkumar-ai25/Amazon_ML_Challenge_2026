import os
import pandas as pd

def inspect_dataset(data_dir="dataset"):
    print("=" * 60)
    print("  AMAZON ML CHALLENGE 2026: DATASET INSPECTION")
    print("=" * 60)
    files = [f for f in os.listdir(data_dir) if f.endswith(('.csv', '.parquet', '.json'))]
    print(f"Files found in {data_dir}: {files}\n")
    for f in sorted(files):
        path = os.path.join(data_dir, f)
        if f.endswith('.csv'):
            df = pd.read_csv(path, nrows=5000)
            total_rows = sum(1 for _ in open(path)) - 1
        elif f.endswith('.parquet'):
            df = pd.read_parquet(path)
            total_rows = len(df)
        print(f"File: {f} | Rows: {total_rows:,} | Cols: {len(df.columns)}")
        print(f"Columns: {list(df.columns)}")
        print(f"Sample Data:\n{df.head(2)}")
        print("-" * 60)

if __name__ == "__main__":
    inspect_dataset()
