#import SG_Model
import time
import string
import math
from collections import Counter
import numpy as np 
import tensorflow as tf 


def loadData(FileName , low_threshold = 10 , high_freq_threshold = 0.85):

    #洗文本
    text = open(FileName).read().lower()
    for i in string.punctuation:
        text = text.replace(i , ' ')
    text = text.split()
    #去高频
    wordCounter = Counter(text)
    text = [word for word in text if wordCounter[word] >= low_threshold]

    high_freq = 1e-3
    wordCounter = Counter(text)
    totalCount = len(text)
    wordFreq = {word : (1 - math.sqrt(high_freq / (count / totalCount)) for word,count in wordCounter.items()}
    text = [word for word in text if wordFreq[word] < high_freq_threshold]

    vocab = set(text)
    vocab_List = list(vocab)

    word2Int = {word : index for index , word in enumerate(vocab_List)}
    int2word = {index : word for index , word in enumerate(vocab_List)}
    encode = {word2Int[word] for word in text}
    return vocab , word2Int ,int2Word , encode


vocab , word2Int , int2Word , encode = loadData('data/text')