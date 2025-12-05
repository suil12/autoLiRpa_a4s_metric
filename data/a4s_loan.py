#!/usr/bin/env python3
"""
Preprocess A4S lcld_v2.csv loan data for certified robustness.
"""

import pandas as pd
from pathlib import Path

input_path = "tests/data/lcld_v2.csv"
output_path = "data/a4s_loan_clean.csv"

def preprocess():
    df = pd.read_csv(input_path)
    print(f"Original shape: {df.shape}")
 
    target_col = "charged_off"
  
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    if target_col not in numeric_cols:
        numeric_cols.append(target_col)
    

    df_clean = df[numeric_cols].dropna()
    
    print(f"After filtering numeric + no NA: {df_clean.shape}")
    print(f"Target distribution:\n{df_clean[target_col].value_counts()}")
  
    df_clean.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset to {output_path}")

if __name__ == "__main__":
    preprocess()
