#import SG_Model
import time
import string
from collections import Counter
import numpy as np 
import tensorflow as tf 


def loadData(FileName , low_threshold = 10 , high_freq_threshold = 0.85):

    text = open(FileName).read().lower()

    for i in string.punctuation:
        text = text.replace(i , ' ')

    text = text.split()
    
    wordCounter = Counter(text)

    

    vocab = set()
    vocab = list(vocab)

    word2Int = 

    
    return vocab , word2Int ,int2Word , encode


vocab , word2Int , int2Word , encode = loadData('data/text')