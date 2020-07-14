import itertools
from typing import List, Tuple

import tensorflow.keras as keras
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import nsml


class Metrics(Callback):
    """
    Keras callback to compute metrics one every epoch, and store them on NSML.

    Arguments:
        name: Name of the model that's being trained. Will show up in the NSML visualization.
        classes: List of the class names, used to build the classification report.
        val_data: Keras data generator.
        n_val_samples: How many datapoints the data generator contains.
        batch_size: Number of datapoints per batch.
    """

    def __init__(self, name: str, classes: List[str], val_data, n_val_samples: int, batch_size: int = 128):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = batch_size
        self.name = name
        self.classes = classes
        self.n_val_samples = n_val_samples

    def on_epoch_end(self, epoch, logs={}):
        val_true, val_pred = evaluate(data_gen=self.validation_data, model=self.model)
        val_true_class = np.argmax(val_true, axis=1)
        val_pred_class = np.argmax(val_pred, axis=1)

        cls_report = classification_report(
            y_true=val_true_class,
            y_pred=val_pred_class,
            output_dict=True,
            target_names=self.classes,
            labels=np.arange(len(self.classes))
        )

        for label, res in cls_report.items():
            if isinstance(res, dict):
                to_report = {f'val__{self.name}/{label}/{k}': v for k, v in res.items()}
            else:
                to_report = {f'val__{self.name}/{label}': res}

            nsml.report(step=epoch, **to_report)

            # NOTE the val__ is needed for the nsml plotting.
            for k, v in to_report.items():
                logs[k.replace('val__', 'val/')] = v


def evaluate(data_gen, model) -> Tuple[np.ndarray]:
    """
    Uses a model to predict on all examples in a data generator.

    Args:
        data_gen: Keras data generator
        model: Keras model
        n_batches: int

    Returns:
        y_true: Ground truth labels. Shape (n_examples, )
        y_pred: Predicted labels. Shape (n_examples, )

    """
    y_preds = []
    y_trues = []
    for _ in range(len(data_gen)):
        X, y_true = next(data_gen)
        y_trues.append(y_true)
        y_pred = model.predict(X)
        y_preds.append(y_pred)

    y_true = np.concatenate(y_trues, axis=0)
    y_pred = np.concatenate(y_preds, axis=0)
    return y_true, y_pred


class NSMLReportCallback(keras.callbacks.Callback):
    def __init__(self, to_log=None, prefix=None):
        if to_log is None:
            to_log = ['loss', 'accuracy']
        self.to_log = to_log + [f'val_{key}' for key in to_log]
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        nsml_keys = []
        for k in self.to_log:
            if 'val' in k:
                k_ = k.replace("val_", "val__")
            else:
                k_ = f'train__{k}'
            if self.prefix is not None:
                nsml_keys.append(f'{self.prefix}__{k_}')
            else:
                nsml_keys.append(k_)
        nsml.report(
            step=epoch,
            **{nsml_key: self._to_json_serializable(logs.get(k)) for nsml_key, k
               in zip(nsml_keys, self.to_log)}
        )

    def _to_json_serializable(self, v):
        return v if not isinstance(v, np.float32) else v.item()
