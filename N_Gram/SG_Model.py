import numpy as np 
import tensorflow as tf 
from model import Model


class Skip_Gram(Model):



    def add_placeholder(self):
        self.input_placeholder = tf.placeholder(shape = (None , None) , dtype = tf.int32 , name = 'input')
        self.label_placeholder = tf.placeholder(shape = (None) , dtype = tf.int32 , name = "label")


    def get_batch(self , ):
        
        return 

    def create_feed_dict(self , input_batch , label_batch):

        feed_dict = {self.input_placeholder : input_batch , self.label_placeholder : label_batch}
        return feed_dict
    
    def add_pred_op(self , vocab_size , embedding_size):
        self.W = tf.Variable(shape = (vocab_size , embedding_size))
        self.bias = tf.Variable(tf.zeros(vocab_size))

    def add_loss_op(self , pred):
        loss = tf.nn.sampled_softmax_loss(weights = self.W , biases = self.bias , labels = self.label_placeholder , )


    def add_embedding(self):
        self.pretrained_embeddings = tf.Variable(dtyp  = float32 , ) 


    def add_training_op(self , loss):
        train_op = tf.train.AdagradOptimizer().minimize(loss)
        return train_op
    
    def train_on_batch(self , sess , input_batch , label_batch):
        feed = self.create_feed_dict(input_batch , label_batch)
        _ , loss = sess.run([self.train_op , self.loss] , feed_dict = feed)

    def run_epopch(self):

    def fit(self):
        losses = []

    
    def model_build(self):
        self.add_placeholder()
        self.pred = self.add_pred_op()
        self.loss = self.add_loss_op()
        self.train_op = self.add_training_op()



    def __init__ (self):
            self.config = config()
            self.build()