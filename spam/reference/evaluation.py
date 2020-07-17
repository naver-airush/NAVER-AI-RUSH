# evaluation.py
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

classes = ['normal', 'monotone', 'screenshot', 'unknown']


def read_prediction(prediction_file):
    df = pd.read_csv(prediction_file)
    return df


def read_ground_truth(ground_truth_file):
    df = pd.read_csv(ground_truth_file)
    return df


def evaluate(prediction, ground_truth):
    if set(prediction['filename']) != set(ground_truth['filename']):
        raise ValueError('Prediction is missing predictions for some files.')
    if len(prediction) != len(ground_truth):
        raise ValueError('Prediction and ground truth have different about of elements.')

    df = prediction.merge(ground_truth, on='filename')
    cls_report = classification_report(
        y_true=df['annotation'],
        y_pred=df['y_pred'],
        output_dict=True,
        target_names=classes,
        labels=np.arange(len(classes)),
        zero_division=0
    )

    # Geometrical mean of f1-scores of non-normal classes
    score = (cls_report['monotone']['f1-score'] * cls_report['screenshot']['f1-score'] * cls_report['unknown'][
        'f1-score']) ** (1 / 3)

    return score


def evaluation_metrics(prediction_file: str, ground_truth_file: str):
    prediction = read_prediction(prediction_file)
    ground_truth = read_ground_truth(ground_truth_file)
    return evaluate(prediction, ground_truth)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # --prediction is set by file's name that contains the result of inference. (nsml internally sets)
    args.add_argument('--prediction', type=str, default='pred.txt')  # output_file from data_loader.py
    args.add_argument('--test_label_path', type=str)  # Ground truth  test/test_label <- csv file
    config = args.parse_args()
    try:
        print(evaluation_metrics(config.prediction, config.test_label_path))
    except:
        print(0.0)
