import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import uuid

from a4s_eval.data_model.evaluation import Dataset, DataShape, Feature, FeatureType
from a4s_eval.data_model.measure import Measure
from a4s_eval.metrics.model_metrics.certified_robust_accuracy_metric import certified_robust_accuracy


def make_linear_dataset(n=100, input_dim=4, num_classes=3):
    X = np.random.randn(n, input_dim).astype(float)
    W = np.random.randn(input_dim, num_classes) * 0.1
    logits = X.dot(W)
    y = logits.argmax(axis=1).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(input_dim)])
    df["target"] = y
    return df


@pytest.fixture
def datashape_and_dataset():
    df = make_linear_dataset(n=100, input_dim=4, num_classes=3)

    feature_cols = [f"f{i}" for i in range(4)]
    feature_mins = df[feature_cols].min().values
    feature_maxs = df[feature_cols].max().values

    features = [
        Feature(
            pid=uuid.uuid4(),
            name=f"f{i}",
            feature_type=FeatureType.FLOAT,
            min_value=float(feature_mins[i]),
            max_value=float(feature_maxs[i]),
        )
        for i in range(4)
    ]
    target = Feature(
        pid=uuid.uuid4(),
        name="target",
        feature_type=FeatureType.INTEGER,
        min_value=float(df["target"].min()),
        max_value=float(df["target"].max()),
    )

    datashape = DataShape(features=features, target=target)
    dataset = Dataset(pid=uuid.uuid4(), shape=datashape, data=df)

    return datashape, dataset


@pytest.fixture
def pytorch_model_wrapper(datashape_and_dataset):
    _, dataset = datashape_and_dataset
    df = dataset.data

    model_nn = nn.Sequential(nn.Linear(4, 3))
    model_nn.train()

    optimizer = optim.SGD(model_nn.parameters(), lr=0.1)
    train_X = torch.tensor(df[[f"f{i}" for i in range(4)]].values, dtype=torch.float32)
    train_y = torch.tensor(df["target"].values, dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    for _ in range(100):
        optimizer.zero_grad()
        logits = model_nn(train_X)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()

    model_nn.eval()

    class ModelWrapper:
        def __init__(self, model):
            self.model = model

    return ModelWrapper(model_nn)


@pytest.mark.parametrize("device_str", ["cpu"])
def test_certified_metric_smoke(datashape_and_dataset, pytorch_model_wrapper, device_str):
    """Smoke test: metric works and returns the expected measures."""
    datashape, dataset = datashape_and_dataset
    model_wrapper = pytorch_model_wrapper

    measures = certified_robust_accuracy(
        datashape,
        model_wrapper,
        dataset,
        eps=0.01,
        batch_size=16,
        device=device_str,
    )


    assert isinstance(measures, list)
    assert len(measures) >= 2
    assert all(isinstance(m, Measure) for m in measures)

    names = {m.name for m in measures}
    assert "certified_robust_accuracy" in names
    assert "nominal_accuracy" in names
    assert "robustness_ratio" in names
    assert "avg_margin" in names
    assert "avg_certified_margin" in names
    assert "eps" in names

    cert_acc = next(m.score for m in measures if m.name == "certified_robust_accuracy")
    nom_acc = next(m.score for m in measures if m.name == "nominal_accuracy")

    assert 0.0 <= cert_acc <= 1.0
    assert 0.0 <= nom_acc <= 1.0
    assert nom_acc >= 0.7


@pytest.mark.parametrize("eps", [0.001, 0.01])
def test_certified_accuracy_increases_with_smaller_eps(datashape_and_dataset, pytorch_model_wrapper, eps):
    datashape, dataset = datashape_and_dataset
    model_wrapper = pytorch_model_wrapper

    measures_small_eps = certified_robust_accuracy(datashape, model_wrapper, dataset, eps=eps)
    measures_large_eps = certified_robust_accuracy(datashape, model_wrapper, dataset, eps=eps * 10)

    cert_small = next(m.score for m in measures_small_eps if m.name == "certified_robust_accuracy")
    cert_large = next(m.score for m in measures_large_eps if m.name == "certified_robust_accuracy")

    assert cert_small >= cert_large - 0.05


def test_model_validation_error(datashape_and_dataset):
    datashape, dataset = datashape_and_dataset

    class NoModelWrapper:
        model = None

    with pytest.raises(ValueError, match="Model object missing"):
        certified_robust_accuracy(datashape, NoModelWrapper(), dataset)


def test_non_torch_model_error(datashape_and_dataset):
    datashape, dataset = datashape_and_dataset

    class InvalidModelWrapper:
        model = "not a torch module"

    with pytest.raises(ValueError, match="only supports PyTorch nn.Module"):
        certified_robust_accuracy(datashape, InvalidModelWrapper(), dataset)


def test_dataset_column_validation(datashape_and_dataset, pytorch_model_wrapper):
    datashape, dataset = datashape_and_dataset
    model_wrapper = pytorch_model_wrapper

    df_corrupt = dataset.data.copy()
    df_corrupt.rename(columns={"f0": "missing_feature"}, inplace=True)
    corrupt_dataset = Dataset(pid=uuid.uuid4(), shape=datashape, data=df_corrupt)

    with pytest.raises(ValueError, match="Column 'f0' not found"):
        certified_robust_accuracy(datashape, model_wrapper, corrupt_dataset)
