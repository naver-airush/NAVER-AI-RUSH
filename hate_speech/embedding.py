import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.linear(x)
        return F.log_softmax(x)


word2vec = Word2Vec(11, 384)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(word2vec.parameters())

sentences = [
    [3, 10, 5, 4], [3, 10, 2, 10, 6, 7, 4]
]

embeds = nn.Embedding(11, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([0, 1], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)


# # # word2vec embedding

# from gensim.models import Word2Vec



# str_sentences = []
# for sentence in sentences:
#     str_sentences.append([str(token) for token in sentence])

# print(sentences)
# embedding_model = Word2Vec(size=384, window=2, min_count=0, workers=4, sg=1)
# embedding_model.build_vocab(sentences)
# embedding_model.train(sentences, total_examples=embedding_model.corpus_count, epochs=30)
# print(embedding_model.wv[10])

# # from gensim.test.utils import datapath
# # from gensim import utils

# # class MyCorpus(object):
# #     """An interator that yields sentences (lists of str)."""

# #     def __iter__(self):
# #         # corpus_path = datapath('lee_background.cor')
# #         # for line in open(corpus_path):
# #         for line in str_sentences:
# #             # assume there's one document per line, tokens separated by whitespace
# #             yield line

# # import gensim.models

# # sentences = MyCorpus()
# # model = gensim.models.Word2Vec(sentences=sentences)
# # model.build_vocab()

# # vec_king = model.wv['king']

# # for i, word in enumerate(model.wv.vocab):
# #     if i == 10:
# #         break
# #     print(word)
