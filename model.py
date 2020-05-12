import os
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence

SAVE_DIR = "save"


def model(full_data, test_data):
    # model的建立
    model_ = Word2Vec(size=100, min_count=5, window=5, workers=4)
    model_.build_vocab(full_data)
    model_.train(full_data, epochs=model_.epochs, total_examples=model_.corpus_count)
    model_.save(os.path.join(SAVE_DIR, "model.pkl"))
    # 构造词典
    dic = Dictionary()
    dic.doc2bow(model_.wv.vocab.keys(), allow_update=True)
    # 构造词到数字的索引
    word2idx = dict()
    for idx, word in dic.items():
        word2idx[word] = idx + 1
    # 构建索引到向量的字典
    nums = len(word2idx) + 1
    idx2vec = np.zeros((nums, 100))
    for word, index in word2idx.items():
        idx2vec[index, :] = model_.wv[word]
    # 将句子中的词替换为索引
    sen2idx = []
    for sen in full_data:
        tmp = []
        for word in sen:
            if word in word2idx.keys():
                tmp.append(word2idx[word])
            else:
                tmp.append(0)
        sen2idx.append(tmp)
    # padding 设置最大句子长度为100 太长也不利于lstm存储有效信息
    sen2idx = sequence.pad_sequences(sen2idx, maxlen=100)

    # test sen2idx
    test_sen2idx = []
    for sen in test_data:
        tmp = []
        for word in sen:
            if word in word2idx.keys():
                tmp.append(word2idx[word])
            else:
                tmp.append(0)
        test_sen2idx.append(tmp)
    test_sen2idx = sequence.pad_sequences(test_sen2idx, maxlen=100)
    return nums, sen2idx, idx2vec, test_sen2idx
