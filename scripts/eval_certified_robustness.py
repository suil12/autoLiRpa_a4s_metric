#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

import argparse
import json
import uuid

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from a4s_eval.data_model.evaluation import Dataset, DataShape, Feature, FeatureType
from a4s_eval.metrics.model_metrics.certified_robust_accuracy_metric import (
    certified_robust_accuracy,
)


def create_datashape_from_df(df: pd.DataFrame, target_col: str = "target") -> DataShape:
    feature_cols = [c for c in df.columns if c != target_col]
    features = []
    for col in feature_cols:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        features.append(
            Feature(
                pid=uuid.uuid4(),
                name=col,
                feature_type=FeatureType.FLOAT,
                min_value=min_val,
                max_value=max_val,
            )
        )
    target = Feature(
        pid=uuid.uuid4(),
        name=target_col,
        feature_type=FeatureType.INTEGER,
        min_value=float(df[target_col].min()),
        max_value=float(df[target_col].max()),
    )
    return DataShape(features=features, target=target)


def main():
    parser = argparse.ArgumentParser(description="Certified Robustness Evaluation")
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to CSV dataset (must contain target column)",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="target",
        help="Name of the target column in the dataset",
    )
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for certification"
    )
    parser.add_argument(
        "--lirpa-method",
        type=str,
        default="backward",
        help="auto_LiRPA method (e.g. backward, forward)",
    )
    args = parser.parse_args()

    print(
        f"evaluation with: eps={args.eps}, method={args.lirpa_method}, device={args.device}"
    )


    if args.dataset_path is not None:
        print(f"Loading dataset from: {args.dataset_path}")
        df = pd.read_csv(args.dataset_path)
        if args.target_col not in df.columns:
            raise ValueError(
                f"Target column '{args.target_col}' not found in dataset columns"
            )
        target_col = args.target_col
    else:
        print("No dataset_path provided, generating synthetic linear dataset.")
        n, input_dim, num_classes = 500, 10, 3
        X = np.random.randn(n, input_dim).astype(np.float32)
        W = np.random.randn(input_dim, num_classes) * 0.1
        logits = X @ W
        y = logits.argmax(axis=1)

        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(input_dim)])
        df["target"] = y.astype(int)
        target_col = "target"

    print(f"Dataset: {df.shape}")

    datashape = create_datashape_from_df(df, target_col=target_col)
    dataset = Dataset(pid=uuid.uuid4(), shape=datashape, data=df)

    
    feature_cols = [c for c in df.columns if c != target_col]
    input_dim = len(feature_cols)
    num_classes = int(df[target_col].nunique())

    model_nn = nn.Sequential(nn.Linear(input_dim, num_classes))
    model_nn.train()

    X_train = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y_train = torch.tensor(df[target_col].values, dtype=torch.long)

    optimizer = torch.optim.SGD(model_nn.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    for _ in tqdm(range(100), desc="Train"):
        optimizer.zero_grad()
        logits = model_nn(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    model_nn.eval()
    with torch.no_grad():
        acc = (model_nn(X_train).argmax(1) == y_train).float().mean()
    print(f"Nominal acc: {acc:.3f}")

    model_wrapper = type("ModelWrapper", (), {"model": model_nn})()

    print("Certified robustness...")
    try:
        measures = certified_robust_accuracy(
            datashape=datashape,
            model=model_wrapper,
            dataset=dataset,
            eps=args.eps,
            norm=float("inf"),
            batch_size=args.batch_size,
            device=args.device,
            lirpa_method=args.lirpa_method,
        )

        for m in measures:
            print(f"   {m.name}: {m.score:.4f}")

        results = {
            "measures": [{"name": m.name, "score": float(m.score)} for m in measures]
        }

        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)

        df_out = pd.DataFrame(results["measures"])

        base = (
            os.path.splitext(os.path.basename(args.dataset_path))[0]
            if args.dataset_path
            else "dataset"
        )
        eps_str = str(args.eps).replace(".", "_")
        csv_name = f"measure_{base}_{eps_str}.csv"

        notebook_dir = os.path.join(os.path.dirname(__file__), "..", "notebook")
        os.makedirs(notebook_dir, exist_ok=True)

        csv_path = os.path.join(notebook_dir, csv_name)
        df_out.to_csv(csv_path, index=False)

        print("Saved results to:")
        print("  results.json")
        print(f"  {csv_path}")

    except Exception as e:
        print("Failed to compute metric")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
