from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class BaseLine(nn.Module):
    def __init__(self, hidden_dim, filter_size, dropout_rate, vocab_size, embedding_dim, pre_trained_embedding=None):
        super().__init__()

        print('hidden_dim:', hidden_dim)

        self.hidden_dim = hidden_dim
        self.filter_size = filter_size
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim

        if pre_trained_embedding is None:
            self.vocab_size = vocab_size
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=1)
        else:
            freeze = False
            print('freeze:', freeze)
            self.embedding = nn.Embedding.from_pretrained(pre_trained_embedding, freeze=freeze, padding_idx=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.conv1d = nn.Conv1d(self.embedding_dim, self.hidden_dim, self.filter_size)
        # How about changing to GRU
        self.bi_rnn = nn.LSTM(self.hidden_dim, int(self.hidden_dim / 2), num_layers=1, batch_first=True, bidirectional=True)
        self.uni_rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)
        # self.max_pool = nn.AdaptiveAvgPool2d((1, self.hidden_dim))
        self.linear = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch_size, sentence_len)
        x = self.embedding(x).transpose(0, 1).transpose(1, 2)
        # x: (sentence_len, embedding_dim, batch_size)
        x = self.conv1d(x).transpose(1, 2).transpose(0, 1)
        # x: (batch_size, sentence_len, hidden_dim)
        x = self.relu(x)
        x = self.dropout(x)
        x_res = x
        x, _ = self.bi_rnn(x)
        x, _ = self.uni_rnn(x + x_res)
        # x: (batch_size, sentence_len, hidden_dim)
        x = self.dropout(x)
        x, _ = torch.max(x, 0)
        # x: (batch_size, hidden_dim)
        x = self.linear(x)
        x = self.sigmoid(x).squeeze()
        return x


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.W = Variable(torch.randn(embedding_dim, vocab_size, device='cuda'), requires_grad=True)

    def forward(self, x, sample):
        x = self.embedding(x)  # x: (window_size,) -> (window_size, embedding_dim)
        x = torch.sum(x, 0)
        x = torch.matmul(x, self.W[:, sample])  # (sample_size,)
        return torch.sigmoid(x)
