import csv
import os
from load_data import load_data
from model import model
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation

RESULT_DIR = "Save"


def train(nums, embedding_weights, train_data, train_label, test_data):
    train_model = Sequential()
    # Embedding层
    Embedding_layer = Embedding(output_dim=300, input_dim=nums, mask_zero=True,
                                weights=[embedding_weights], input_length=100)
    train_model.add(Embedding_layer)
    # LSTM层
    LSTM_layer = LSTM(activation="sigmoid", units=50)
    train_model.add(LSTM_layer)
    # 过完LSTM 过一个Dropout
    train_model.add(Dropout(0.4))
    # 然后过全连接层得到结果
    train_model.add(Dense(1))
    # 使用sigmoid作为激活函数
    train_model.add(Activation('sigmoid'))

    # 使用交叉熵损失函数 Adam优化器 使用acc作为评价函数
    # 其实应该使用macro-f1训练会更好
    train_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    train_model.fit(train_data, train_label, batch_size=32, epochs=5, verbose=1)

    ret = train_model.predict_classes(test_data)
    return ret


def main():
    full_data, labels, test_data = load_data()
    nums, sen2idx, idx2vec, test_sen2idx = model(full_data, test_data)

    ret = train(nums, idx2vec, sen2idx, labels, test_sen2idx)
    with open(os.path.join(RESULT_DIR, 'result.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(ret)):
            writer.writerow([i, ret[i][0]])


if __name__ == "__main__":
    main()
