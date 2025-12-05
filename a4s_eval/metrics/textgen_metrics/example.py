from datetime import datetime
from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.textgen_metric_registry import textgen_metric
from a4s_eval.service.functional_model import TextGenerationModel


@textgen_metric(name="textgen_example")
def my_test_metric(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    functional_model: TextGenerationModel,
) -> list[Measure]:
    if dataset.data is None:
        raise ValueError

    features = dataset.data[[f.name for f in datashape.features]]

    prediction = functional_model.generate_text(str(features.iloc[1].values))
    print(prediction)

    my_measure = 0.99

    current_time = datetime.now()
    return [Measure(name="my_measure", score=my_measure, time=current_time)]
