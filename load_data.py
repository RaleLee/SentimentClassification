import os
import re
import numpy as np

from pyltp import Segmentor

LTP_DATA_DIR = "D:/Course/IR/ltp_data_v3.4.0"
STOP_WORDS_PATH = "train_data/stopwords.txt"
POS_DATA_PATH = "train_data/sample.positive.txt"
NEG_DATA_PATH = "train_data/sample.negative.txt"
TEST_DATA_PATH = "test_data/test.txt"
CWS_MODEL_PATH = os.path.join(LTP_DATA_DIR, "cws.model")


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().replace("\n", "")
    pattern = r'<review id="\d+">(.+?)</review>'
    text_list = re.findall(pattern, text, flags=re.S)
    return text_list


def get_stop_words():
    """
    从指定的文件中获取stopwords
    :return: 文件不存在则报错，存在则返回stopwords列表
    """
    stopwords = []
    path = STOP_WORDS_PATH
    if not os.path.exists(path):
        print("No stop words file!")
        return
    with open(path, "r", encoding="utf-8") as f:
        stopwords.append(f.readline().strip())
    return stopwords


def seg(stopwords, needsegs, segor: Segmentor):
    """
    分词执行程序，将会进行分词和去停用词
    :param stopwords: 停用词表
    :param needsegs: 需要分词的列表
    :param segor: 分词程序
    :return: 列表，每个元素是分词去停用词之后的词列表
    """
    ret = []
    for data in needsegs:
        words = list(segor.segment(data))
        ret.append(remove_stop_words(stopwords, words))
    return ret


def remove_stop_words(stopwords: list,
                      text_words: list):
    """
    对分词结果进行去停用词处理
    :param stopwords: 停用词列表
    :param text_words: 分词列表
    :return: 去掉停用词后的分词结果
    """
    ret = []
    for text_word in text_words:
        if text_word not in stopwords:
            ret.append(text_word)
    return ret


def load_data():
    stop_words = get_stop_words()
    segmentor = Segmentor()
    segmentor.load(CWS_MODEL_PATH)
    pos_data = seg(stop_words, load_text(POS_DATA_PATH), segmentor)
    neg_data = seg(stop_words, load_text(NEG_DATA_PATH), segmentor)
    test_data = seg(stop_words, load_text(TEST_DATA_PATH), segmentor)
    segmentor.release()
    full_data = np.concatenate((pos_data, neg_data))
    labels = np.concatenate((np.ones(len(pos_data)), np.zeros(len(neg_data))))
    return full_data, labels, test_data


if __name__ == "__main__":
    full, label, test = load_data()


