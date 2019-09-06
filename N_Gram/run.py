
import time
import string
import math
from collections import Counter
import numpy as np 
import tensorflow as tf 
from SG_Model import Skip_Gram

class Config(object):
    #超参数
    epochs = 10
    embedding_size = 300
    windows_size = 5
    batch_size = 100
    n_sample = 100
    #文本数据
    text = None
    vocab_size = None
    int2Word = None


def loadData(FileName , low_threshold = 10 , high_freq_threshold = 0.85):

    #洗文本
    text = open(FileName).read().lower()
    for i in string.punctuation:
        text = text.replace(i , ' ')
    text = text.split()
    #去低频
    wordCounter = Counter(text)
    text = [word for word in text if wordCounter[word] >= low_threshold]
    #去高频
    high_freq = 1e-3
    wordCounter = Counter(text)
    totalCount = len(text)
    wordFreq = {word : (1 - math.sqrt(high_freq / (count / totalCount))) for word , count in wordCounter.items()}
    text = [word for word in text if wordFreq[word] < high_freq_threshold]
    #生成字典
    vocab = set(text)
    vocab_List = list(vocab)

    word2Int = {word : index for index , word in enumerate(vocab_List)}
    int2Word = {index : word for index , word in enumerate(vocab_List)}
    encode = {word2Int[word] for word in text}


    return vocab , word2Int ,int2Word , encode



if __name__ == '__main__':

    vocab , word2Int , int2Word , encode = loadData('data/text')

    config = Config()
    config.text = encode
    config.vocab_size = len(vocab)
    config.int2Word = int2Word

    with tf.Graph().as_default() as graph:
        start = time.time()
        model = Skip_Gram(config)
        init_op = tf.global_variables_initializer()
        
        print "took {:.2f} seconds\n".format(time.time() - start)
    graph.finalize()

    with tf.Session(graph=graph) as sess:
        sess.run(init_op)


    print('Done!')
