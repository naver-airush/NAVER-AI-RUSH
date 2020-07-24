import sys
import json
import math
import typing
from typing import Dict, List
import os
from argparse import ArgumentParser
import random
from torch import nn, optim
import torch
from torchtext.data import Iterator
from tqdm import tqdm
import numpy as np
import nsml
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML
from torchtext.data import Example

from model import BaseLine, Word2Vec
from data import HateSpeech

from sklearn.metrics import recall_score, precision_score, f1_score

WORD2VEC_LOAD = True
WORD2VEC_CHECKPOINT = 'word2vec1199999'
WORD2VEC_SESSION = 'yonsweng/hate_2/73'
print(WORD2VEC_CHECKPOINT, WORD2VEC_SESSION)


def bind_model(model):
    def save(dirname, *args):
        checkpoint = {
            'model': model.state_dict()
        }
        torch.save(checkpoint, os.path.join(dirname, 'model.pt'))

    def load(dirname, *args):
        checkpoint = torch.load(os.path.join(dirname, 'model.pt'))
        model.load_state_dict(checkpoint['model'])

    def infer(raw_data, **kwargs):
        model.eval()
        examples = HateSpeech(raw_data).examples
        tensors = [torch.tensor(ex.syllable_contents, device='cuda').reshape([-1, 1]) for ex in examples]
        results = [model(ex).tolist() for ex in tensors]
        return results

    nsml.bind(save=save, load=load, infer=infer)


class Trainer(object):
    TRAIN_DATA_PATH = '{}/train/train_data'.format(DATASET_PATH[0])
    UNLABELED_DATA_PATH = '{}/train/raw.json'.format(DATASET_PATH[1])

    def __init__(self, model, hdfs_host: str = None, device: str = 'cpu'):
        self.device = device
        self.task = HateSpeech(self.TRAIN_DATA_PATH, (9, 1))  # train 9 : test 1
        # self.embedding = self.train_embedding()
        self.model = model
        self.model.to(self.device)
        self.loss_fn = nn.BCELoss()
        self.batch_size = 32
        self.__test_iter = None
        bind_model(self.model)
        print(f'batch_size: {self.batch_size}')

    def train_embedding(self):
        word2vec = Word2Vec(self.task.max_vocab_indexes['syllable_contents'], EMBEDDING_SIZE)
        word2vec.to('cuda')
        bind_model(word2vec)
        if WORD2VEC_LOAD:
            nsml.load(WORD2VEC_CHECKPOINT, session=WORD2VEC_SESSION)
        else:
            preprocessed = []
            with open(self.TRAIN_DATA_PATH) as fp:
                for line in fp:
                    if line:
                        tokens = json.loads(line)['syllable_contents']
                        preprocessed.append(tokens)
            with open(self.UNLABELED_DATA_PATH) as fp:
                for line in fp:
                    if line:
                        tokens = json.loads(line)['syllable_contents']
                        preprocessed.append(tokens)
            print('# of sentences:', len(preprocessed))

            # count occurrences of tokens
            token_cnt = [0] * self.task.max_vocab_indexes['syllable_contents']
            for sentence in preprocessed:
                for token in sentence:
                    token_cnt[token] += 1
            token_sum = sum(token_cnt)
            token_prob = [cnt / token_sum for cnt in token_cnt]
            # print('token probabilities')
            # print('2:', token_prob[2])
            # print('100:', token_prob[100])
            token_prob = [prob ** 0.75 for prob in token_prob]
            token_prob_sum = sum(token_prob)
            token_prob = [prob / token_prob_sum for prob in token_prob]
            # print('token adjusted probabilities')
            # print('2:', token_prob[2])
            # print('100:', token_prob[100])
            tokens = [i for i in range(self.task.max_vocab_indexes['syllable_contents'])]

            lr = 0.03
            print('word2vec lr:', lr)
            optimizer = optim.Adam(word2vec.parameters(), lr=lr)
            loss_fn = nn.BCELoss()
            losses = []
            batch_cnt = 0
            negative_sample_size = 5
            window_size = 5
            word2vec_batch_size = 1024
            for epoch in range(1):
                print('word2vec epoch:', epoch)
                epoch_loss = 0
                epoch_token_cnt = 0
                sentence_loss = 0
                sentence_token_cnt = 0
                random.shuffle(preprocessed)
                for i_sentence, sentence in enumerate(preprocessed):
                    if (i_sentence+1) % 10000 == 0:
                        print(i_sentence+1, 'loss:', sentence_loss/sentence_token_cnt)
                        nsml.report(step=i_sentence+1, loss=sentence_loss/sentence_token_cnt)
                        sentence_loss = 0
                        sentence_token_cnt = 0
                        
                    # save checkpoint
                    if (i_sentence+1) % 100000 == 0:
                        nsml.save('word2vec' + str(i_sentence+1))

                    # sample only 5
                    for center in random.sample(range(len(sentence)), min(10, len(sentence))):
                    # for center in range(len(sentence)):
                        left = center - window_size
                        if left < 0:
                            left = 0
                        right = center + window_size

                        # negative sample
                        negative_sample = np.random.choice(tokens, negative_sample_size, p=token_prob)
                        positive_sample = np.array([sentence[center]])
                        sample = np.concatenate([positive_sample, negative_sample])
                        target = np.where(sample == sentence[center], 1., 0.)
                        sample = torch.tensor(sample, device='cuda')
                        target = torch.tensor(target, device='cuda')

                        x = torch.tensor(sentence[left:center] + sentence[center+1:right+1], device='cuda')
                        output = word2vec(x, sample)  # (vocab_size,)
                        losses.append(loss_fn(output.double(), target.double()))

                        batch_cnt += 1
                        if batch_cnt == word2vec_batch_size:
                            optimizer.zero_grad()
                            loss = sum(losses)
                            epoch_loss += loss.tolist()
                            sentence_loss += loss.tolist()
                            sentence_token_cnt += word2vec_batch_size
                            epoch_token_cnt += word2vec_batch_size
                            loss.backward()
                            losses = []
                            optimizer.step()
                            batch_cnt = 0
                print('loss:', epoch_loss / epoch_token_cnt)
        # print(word2vec.embedding.weight[2])
        return word2vec.embedding.weight

    @property
    def test_iter(self) -> Iterator:
        if self.__test_iter:
            self.__test_iter.init_epoch()
            return self.__test_iter
        else:
            self.__test_iter = Iterator(self.task.datasets[1], batch_size=self.batch_size, repeat=False,
                                        sort_key=lambda x: len(x.syllable_contents), train=False,
                                        device=self.device)
            return self.__test_iter

    def train(self):
        max_epoch = 32
        lr = 0.001
        print('lr:', lr)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        total_len = len(self.task.datasets[0])
        print('shuffle: True')
        ds_iter = Iterator(self.task.datasets[0], batch_size=self.batch_size, repeat=False,
                           shuffle=True, train=True, device=self.device)
        min_iters = 10
        for epoch in range(max_epoch):
            loss_sum, acc_sum, len_batch_sum = 0., 0., 0.
            ds_iter.init_epoch()
            tr_total = math.ceil(total_len / self.batch_size)
            tq_iter = tqdm(enumerate(ds_iter), total=tr_total, miniters=min_iters, unit_scale=self.batch_size,
                           bar_format='{n_fmt}/{total_fmt} [{elapsed}<{remaining} {rate_fmt}] {desc}')

            true_lst = list()
            pred_lst = list()

            self.model.train()
            for i, batch in tq_iter:
                self.model.zero_grad()
                pred = self.model(batch.syllable_contents)
                acc = torch.sum((torch.reshape(pred, [-1]) > 0.5) == (batch.eval_reply > 0.5), dtype=torch.float32)
                loss = self.loss_fn(pred, batch.eval_reply)
                loss.backward()
                optimizer.step()

                true_lst += batch.eval_reply.tolist()
                pred_lst += pred.tolist()

                len_batch = len(batch)
                len_batch_sum += len_batch
                acc_sum += acc.tolist()
                loss_sum += loss.tolist() * len_batch
                if i % min_iters == 0:
                    tq_iter.set_description('{:2} loss: {:.5}, acc: {:.5}'.format(epoch, loss_sum / len_batch_sum, acc_sum / len_batch_sum), True)
                if i == 3000:
                    break

            # calc f1-score
            y_true = np.array(true_lst) > 0.5
            y_pred = np.array(pred_lst) > 0.5
            train_recall_score = recall_score(y_true, y_pred)
            train_precision_score = precision_score(y_true, y_pred)
            train_f1_score = f1_score(y_true, y_pred)

            tq_iter.set_description('{:2} loss: {:.5}, acc: {:.5}'.format(epoch, loss_sum / total_len, acc_sum / total_len), True)
            print(json.dumps(
                {'type': 'train', 'dataset': 'hate_speech',
                 'epoch': epoch, 'loss': loss_sum / total_len, 'acc': acc_sum / total_len,
                 'recall': train_recall_score, 'precision': train_precision_score, 'f1': train_f1_score}))

            pred_lst, loss_avg, acc_lst, te_total = self.eval(self.test_iter, len(self.task.datasets[1]))

            # calc f1-score
            y_pred = np.array(pred_lst) > 0.5
            y_true = np.where(np.array(acc_lst) > 0.5, y_pred, 1 - y_pred)
            test_recall_score = recall_score(y_true, y_pred)
            test_precision_score = precision_score(y_true, y_pred)
            test_f1_score = f1_score(y_true, y_pred)
            print(json.dumps(
                {'type': 'test', 'dataset': 'hate_speech',
                 'epoch': epoch, 'loss': loss_avg,  'acc': sum(acc_lst) / te_total,
                 'recall': test_recall_score, 'precision': test_precision_score, 'f1': test_f1_score}))
            nsml.save(epoch)
            self.save_model(self.model, 'e{}'.format(epoch))

            # plot graphs
            train_loss = loss_sum / total_len
            test_loss = loss_avg
            nsml.report(step=epoch, train_loss=train_loss,
                        train_recall=train_recall_score, train_precision=train_precision_score, train_f1=train_f1_score)
            nsml.report(step=epoch, test_loss=test_loss,
                        test_recall=test_recall_score, test_precision=test_precision_score, test_f1=test_f1_score)

    def eval(self, iter:Iterator, total:int) -> (List[float], float, List[float], int):
        tq_iter = tqdm(enumerate(iter), total=math.ceil(total / self.batch_size),
                       unit_scale=self.batch_size, bar_format='{r_bar}')
        pred_lst = list()
        loss_sum= 0.
        acc_lst = list()

        self.model.eval()
        for i, batch in tq_iter:
            preds = self.model(batch.syllable_contents)
            accs = torch.eq(preds > 0.5, batch.eval_reply > 0.5).to(torch.float)
            losses = self.loss_fn(preds, batch.eval_reply)
            pred_lst += preds.tolist()
            acc_lst += accs.tolist()
            loss_sum += losses.tolist() * len(batch)
        return pred_lst, loss_sum / total, acc_lst, total

    def save_model(self, model, appendix=None):
        file_name = 'model'
        if appendix:
            file_name += '_{}'.format(appendix)
        torch.save({'model': model, 'task': type(self.task).__name__}, file_name)


if __name__ == '__main__':
    # Constants
    HIDDEN_DIM = 256
    FILTER_SIZE = 3
    DROPOUT_RATE = 0.2
    EMBEDDING_SIZE = 384

    parser = ArgumentParser()
    parser.add_argument('--mode', default='train')
    parser.add_argument('--pause', default=0)
    args = parser.parse_args()
    task = HateSpeech()
    model = BaseLine(HIDDEN_DIM, FILTER_SIZE, DROPOUT_RATE, task.max_vocab_indexes['syllable_contents'], EMBEDDING_SIZE)
    if args.pause:
        model.to("cuda")
        bind_model(model)
        nsml.paused(scope=locals())
    if args.mode == 'train':
        trainer = Trainer(model, device='cuda')
        trainer.train()
