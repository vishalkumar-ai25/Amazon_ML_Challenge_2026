import sys
import pandas as pd
import numpy as np

def validate_submission(submission_path, test_path, id_col, target_col):
    print(f"Auditing submission: {submission_path} against {test_path}...")
    sub = pd.read_csv(submission_path)
    test = pd.read_csv(test_path)
    assert len(sub) == len(test), f"Row count mismatch! Expected {len(test):,}, got {len(sub):,}"
    assert id_col in sub.columns, f"Missing ID column: {id_col}"
    assert target_col in sub.columns, f"Missing target column: {target_col}"
    assert (sub[id_col].values == test[id_col].values).all(), "ID values/order mismatch!"
    assert not sub[target_col].isna().any(), "Found NaN values in predictions!"
    assert not np.isinf(sub[target_col]).any(), "Found Infinite values in predictions!"
    print("SUCCESS: Submission passed all validation checks!")
    print(f"Summary: {len(sub):,} rows | Target dtype: {sub[target_col].dtype}")
    print(f"Preview:\n{sub.head(3)}")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python -m src.validate <sub_path> <test_path> <id_col> <target_col>")
    else:
        validate_submission(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
