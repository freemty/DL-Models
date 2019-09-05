import tensorflow as tf 
import numpy as np 

import model


n_seqs = 100
n_sequencd_length = 100
lstm_num_units = 512 
num_layers = 2
keep_prob = 0.5
learning_rate = 0.01
epochs = 200
fileName = 'data/Harry_Potter1-7.txt'


vocab , vocab2Int , int2Vocab , encod = model.loadData(fileName)
char_rnn = model.char_RNN(vacab = vocab , n_seqs= n_seqs , n_sequendcd_length= n_sequencd_length \
    , keep_prob = keep_prob , learning_rate= learning_rate , clip_val= 5)
count = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        for x,y in model.get_batch():
            count += 1

            feed = {
                    char_rnn.input:x,
                    char_rnn.target:y
                    }

            _,loss,_ = sess.run([,char_rnn.optimizer],feed_dict = feed)

            if count % 500 == 0:
                print('----------------------')
                print('')


        



