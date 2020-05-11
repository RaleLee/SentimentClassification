# SentimentClassification
这是选修课情感分析的大作业，写一个简单的情感分类系统，方法自定。
## Requirements
详见requirements.txt，可以使用pip install -r requirements.txt来安装，
如果pyltp安装麻烦可以使用jieba分词，建议在python3.6环境下运行

Keras==2.3.1

numpy==1.17.3

gensim==3.8.3

pyltp==0.2.1

## Approach
首先对数据进行分词和去停用词处理，详见load_data.py，使用了pyltp作为分词工具，
停用词表采用SCIR的停用词表。

使用Word2Vec做Embedding，使用LSTM作为model，对训练数据进行训练，
实现一个简单的情感二分类系统。代码详见model.py train.py。

实验环境只有一块低压锐龙R7，没有显卡，没有对超参数进行调参选择。

结果保存在Save文件夹中