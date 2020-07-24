from nsml import DATASET_PATH
from data import HateSpeech
from torchtext.data import Iterator

TRAIN_DATA_PATH = '{}/train/train_data'.format(DATASET_PATH[0])
UNLABELED_DATA_PATH = '{}/train/raw.json'.format(DATASET_PATH[1])
N_GRAM_SIZE = 2

task = HateSpeech(TRAIN_DATA_PATH)

vocab_size = task.max_vocab_indexes['syllable_contents']

ds_iter = Iterator(task.datasets[0], batch_size=2, repeat=False, shuffle=False, train=False,
                   sort_key=lambda x: -len(x.syllable_contents))
ds_iter.init_epoch()

for i, batch in enumerate(ds_iter):
    print(batch.syllable_contents, batch.eval_reply)

    if i >= 10:
        break
