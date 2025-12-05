"""
A4S Model Metric: Certified Robust Accuracy using auto_LiRPA (PyTorch-only)

This file implements a Model Metric compatible data model definitions
found in a4s_eval.data_model. The metric computes *certified robust accuracy*
using the auto_LiRPA library and returns A4S-compatible Measure objects.

Assumptions & behavior:
  - The metric ONLY supports PyTorch models. If the provided
    `model.model` is not a torch.nn.Module the metric raises ValueError.
  - Dataset is expected to be an a4s_eval.data_model.evaluation.Dataset
    instance whose `.data` attribute is a pandas.DataFrame containing feature
    columns with names matching datashape.features[*].name and the target
    column matching datashape.target.name.
  - Returns two Measure objects (A4S Measure):
      * certified_robust_accuracy 
      * nominal_accuracy 
      * robustness_ratio
      * avg_margin
      * avg_certified_margin

Requirements:
  pip install torch auto-LiRPA pandas numpy
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import List

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except Exception as e:
    raise ImportError("This metric requires PyTorch. Install it with `pip install torch`.") from e

try:
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
except Exception:
    raise ImportError("This metric requires auto_LiRPA. Install it with `pip install auto-LiRPA`.")

import pandas as pd

from a4s_eval.data_model.evaluation import Dataset, DataShape, Model
from a4s_eval.data_model.measure import Measure

# Metric registry decorator: import the real decorator 
try:
    from a4s_eval.metric_registries.model_metric_registry import model_metric
except Exception:
    def model_metric(name: str):
        def _decorator(fn):
            fn._a4s_name = name
            return fn
        return _decorator


@model_metric(name="certified_robust_accuracy")
def certified_robust_accuracy(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    eps: float = 0.03,
    norm: int | float = float("inf"),
    batch_size: int = 32,
    device: str | torch.device = "auto",
    lirpa_method: str = "CROWN-IBP",
) -> List[Measure]:
    """Compute certified robust accuracy for a classification PyTorch model.

    Args:
        datashape: DataShape describing feature and target names.
        model: A4S Model wrapper. Must contain a PyTorch nn.Module in `.model`.
        dataset: A4S Dataset wrapper. `.data` must be a pandas.DataFrame.
        eps: perturbation radius.
        norm: p for Lp perturbation.
        batch_size: batch size for evaluation.
        device: 'auto'|'cpu'|'cuda' or torch.device.
        lirpa_method: auto_LiRPA bound method.

    Returns:
        List[Measure] compatible with a4s_eval.data_model.measure.Measure.
    """

    # device handlin
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Validate model is a torch.nn.Module only pytorch supported
    if model is None or getattr(model, "model", None) is None:
        raise ValueError("Model object missing: expected .model to contain a PyTorch nn.Module.")

    pt_model = getattr(model, "model")
    if not isinstance(pt_model, nn.Module):
        raise ValueError("certified_robust_accuracy metric only supports PyTorch nn.Module models.")

    pt_model = pt_model.to(device)
    pt_model.eval()

    # Validion of dataset
    if dataset is None or getattr(dataset, "data", None) is None:
        raise ValueError("Dataset missing: expected Dataset.data to be a pandas.DataFrame.")
    df: pd.DataFrame = dataset.data

    # Extract feature columns and target column from datashape
    feature_names = [f.name for f in datashape.features]
    if datashape.target is None:
        raise ValueError("DataShape must include a target feature for classification tasks.")
    target_name = datashape.target.name

    for col in feature_names + [target_name]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset.data")

    # tensors building
    X = df[feature_names].to_numpy(dtype=float)
    y = df[target_name].to_numpy(dtype=int)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset_torch = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset_torch, batch_size=batch_size)

   
    sample_x = X_tensor[:1].to(device)

    lirpa_model = BoundedModule(pt_model, sample_x, device=device)

    total = 0
    certified = 0
    correct = 0
    margins = []              
    certified_margins = []

    for x_batch, y_batch in loader:
        x = x_batch.to(device)
        y = y_batch.to(device)
        bs = x.shape[0]

        with torch.no_grad():
            logits = pt_model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()

        perturbation = PerturbationLpNorm(norm=norm, eps=eps)
        bounded_x = BoundedTensor(x, perturbation)

        try:
            lb, ub = lirpa_model.compute_bounds(x=(bounded_x,), method=lirpa_method)
        except TypeError:
            lb, ub = lirpa_model.compute_bounds((bounded_x,), method=lirpa_method)

        lb_cpu = lb.detach().cpu()
        ub_cpu = ub.detach().cpu()
        y_cpu = y.detach().cpu()

        for i in range(bs):
            total += 1
            true = int(y_cpu[i].item())
            lower_true = float(lb_cpu[i, true].item())
            if ub_cpu.shape[1] > 1:
                upper_others = float(
                    torch.cat([ub_cpu[i, :true], ub_cpu[i, true + 1:]]).max().item()
                )
            else:
                upper_others = float("-inf")

            margin = lower_true - upper_others
            margins.append(margin)

            if lower_true > upper_others:
                certified += 1
                certified_margins.append(margin)


    certified_accuracy = certified / total if total > 0 else 0.0
    nominal_accuracy = correct / total if total > 0 else 0.0
    robustness_ratio = (
        certified_accuracy / nominal_accuracy if nominal_accuracy > 0 else 0.0
    )
    avg_margin = float(np.mean(margins)) if margins else 0.0
    avg_certified_margin = (
        float(np.mean(certified_margins)) if certified_margins else 0.0
    )

    now = datetime.utcnow()

    measures: List[Measure] = [
        Measure(name="certified_robust_accuracy", score=certified_accuracy, time=now),
        Measure(name="nominal_accuracy", score=nominal_accuracy, time=now),
        Measure(name="robustness_ratio", score=robustness_ratio, time=now),
        Measure(name="avg_margin", score=avg_margin, time=now),
        Measure(name="avg_certified_margin", score=avg_certified_margin, time=now),
        Measure(name="eps", score=float(eps), time=now),
    ]

    return measures

