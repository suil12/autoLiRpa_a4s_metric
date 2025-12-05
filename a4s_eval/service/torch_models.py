import numpy as np
import torch

from a4s_eval.data_model.evaluation import ModelConfig
from a4s_eval.service.functional_model import TabularClassificationModel
from a4s_eval.typing import Array


def load_torch_classification(model_config: ModelConfig) -> TabularClassificationModel:
    model = torch.jit.load(model_config.path)

    def predict_proba(x: Array) -> Array:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            y_pred = model(x)
        return y_pred.detach().cpu().numpy()

    def predict_class(x: Array) -> Array:
        y_pred = predict_proba(x)
        return np.argmax(y_pred, axis=-1)

    def predict_proba_grad(x: Array) -> Array:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.requires_grad_()
        y_pred = model(x)
        return y_pred

    return TabularClassificationModel(
        predict_class=predict_class,
        predict_proba=predict_proba,
        predict_proba_grad=predict_proba_grad,
    )
