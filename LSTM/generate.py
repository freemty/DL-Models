import tensorflow as tf 
import numpy as np 

from model import *
from train import*

def pick_top_n (preds,vocab_size,top_n = 5):



    return c

def sample (checkpoint,n_samples,lstm_num_units,vocab_size,prime = "The"):



    return sample


if __main__ = '__main__':
    vocab,vocab2Int,int2Vocab,encode = ltsm.loadData('Harry_Potter1-7.txt')

    checkpoint = tf.train.latest_checkpoint('checkpoint/')

    print(checkpoint)

    samp = sample(checkpoint,150,lstm_num_units,len(vocab),prime = "Hi,")
    print('--------------------------')
    print(samp)
