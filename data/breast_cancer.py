#!/usr/bin/env python3
"""
Preprocess UCI Breast Cancer Wisconsin dataset for certified robustness.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def preprocess_breast_cancer():
    """Load, clean, and save breast cancer dataset."""
    
    # Load raw data (adjust path if needed)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    columns = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(30)]
    
    df = pd.read_csv(url, header=None, names=columns)
    
    # Convert diagnosis to binary: M=1 (malignant), B=0 (benign)
    df['target'] = (df['diagnosis'] == 'M').astype(int)
    df = df.drop(['id', 'diagnosis'], axis=1)
    
    # Check data quality
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Save clean version
    output_path = Path("data/breast_cancer_clean.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Saved to {output_path}")
    
    return df

if __name__ == "__main__":
    preprocess_breast_cancer()
