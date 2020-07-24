import os
import copy
import nsml
import torch
import torch.nn as nn
from nsml import DATASET_PATH
from data import HateSpeech
from model import BaseLine
from main import bind_model
from argparse import ArgumentParser

model_lst = [
    ('5', 'yonsweng/hatespeech-1/5'),
    ('6', 'yonsweng/hatespeech-1/25')
]


class MyEnsemble(nn.Module):
    def __init__(self, models):  # models: list
        super(MyEnsemble, self).__init__()
        self.models = models
        
    def forward(self, x):
        x = [model(x) for model in self.models]
        x = sum(x) / len(x)
        return x


parser = ArgumentParser()
parser.add_argument('--mode', default='train')
parser.add_argument('--pause', default=0)
args = parser.parse_args()
if args.pause:
    task = HateSpeech()
    models = []
    for i, (checkpoint, session) in enumerate(model_lst):
        model = BaseLine(256, 3, 0.2, task.max_vocab_indexes['syllable_contents'], 384)
        model.to("cuda")
        models.append(model)
    ensemble_model = MyEnsemble(models)
    bind_model(ensemble_model)
    nsml.paused(scope=locals())
if args.mode == 'train':
    TRAIN_DATA_PATH = '{}/train/train_data'.format(DATASET_PATH)
    task = HateSpeech(TRAIN_DATA_PATH, (9, 1))  # train 9 : test 1
    models = []
    for i, (checkpoint, session) in enumerate(model_lst):
        model = BaseLine(256, 3, 0.2, task.max_vocab_indexes['syllable_contents'], 384)
        model.to("cuda")
        bind_model(model)
        nsml.load(checkpoint=checkpoint, session=session)
        nsml.save(i)
        models.append(model)
    ensemble_model = MyEnsemble(models)
    bind_model(ensemble_model)
    nsml.save(len(models))
    torch.save({'model': ensemble_model, 'task': type(task).__name__}, 'model')
