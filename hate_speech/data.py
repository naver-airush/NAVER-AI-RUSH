import json
import typing
from typing import List, Dict, Tuple
from pydoc import locate
from collections import Counter

from torchtext.data import Dataset, Example, Field, LabelField
from torchtext.vocab import Vocab
import torch
from nsml.constants import DATASET_PATH


class HateSpeech(object):
    MAX_LEN = 512
    UNK_TOKEN = 0  # '<unk>'
    PAD_TOKEN = 1  # '<pad>'
    SPACE_TOKEN = 2  # '<sp>'
    INIT_TOKEN = 3  # '<s>'
    EOS_TOKEN = 4  # '<e>'
    TOKENS = [PAD_TOKEN, UNK_TOKEN, SPACE_TOKEN, INIT_TOKEN, EOS_TOKEN]
    FIELDS_TOKEN_ATTRS = ['init_token', 'eos_token', 'unk_token', 'pad_token']
    FIELDS_ATTRS = FIELDS_TOKEN_ATTRS + ['sequential', 'use_vocab', 'fix_length']

    VOCAB_PATH = 'fields.json'

    def __init__(self, corpus_path=None, split: Tuple[int, int] = None):
        self.fields, self.max_vocab_indexes = self.load_fields(self.VOCAB_PATH)

        if corpus_path:
            self.examples = self.load_corpus(corpus_path)
            if split:
                total = len(self.examples)
                pivot = int(total / sum(split) * split[0])
                self.datasets = [Dataset(self.examples[:pivot], fields=self.fields),
                                 Dataset(self.examples[pivot:], fields=self.fields)]
            else:
                self.datasets = [Dataset(self.examples, fields=self.fields)]


    def load_corpus(self, path) -> List[Example]:
        preprocessed = []
        with open(path) as fp:
            for line in fp:
                if line:
                    ex = Example()
                    for k, v in json.loads(line).items():
                        setattr(ex, k, v)
                    preprocessed.append(ex)
        return preprocessed

    def dict_to_field(self, dicted: Dict) -> Field:
        field = locate(dicted['type'])(dtype=locate(dicted['dtype']))
        for k in self.FIELDS_ATTRS:
            setattr(field, k, dicted[k])

        if 'vocab' in dicted:
            v_dict = dicted['vocab']
            vocab = Vocab()
            vocab.itos = v_dict['itos']
            vocab.stoi.update(v_dict['stoi'])
            vocab.unk_index = v_dict['unk_index']
            if 'freqs' in v_dict:
                vocab.freqs = Counter(v_dict['freqs'])
        else:
            vocab = Vocab(Counter())
            field.use_vocab = False
        field.vocab = vocab

        return field

    def load_fields(self, path) -> Dict[str, Field]:
        loaded_dict = json.loads(open(path).read())
        max_vocab_indexes = {k: v['max_vocab_index'] for k, v in loaded_dict.items()}
        return {k: self.dict_to_field(v) for k, v in loaded_dict.items()}, max_vocab_indexes
