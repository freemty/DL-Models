import time
import numpy as np 
import tensorflow as tf 
from model import Model


class Skip_Gram(Model):

    def add_placeholder(self):
        self.input_placeholder = tf.placeholder(shape = (None , None) , dtype = tf.int32 , name = 'input')
        self.label_placeholder = tf.placeholder(shape = (None) , dtype = tf.int32 , name = "label")


    def get_windows_word(self , word_id):
        start = word_id - self.config.windows_size 
            if word_id - self.config.windows_size >=0 else 0
        end = word_id + self.config.windows_size 
            if word_id + self.config.windows_size > len(self.config.text) else (len(self.config.text) - 1)
        window_word = set(self.config.text[start : word_id] + self.config.text[word_id + 1 : end + 1] )
        return list(window_word)

    def get_batch(self):

        x , y = [] , []
        for id in range(len(text)):
            window_word = self.get_windows_word(id)
            x.extend([text[id]] * len(window_word))
            y.extend(window_word)
        '''
        combine = list(zip(x, y))
        random.shuffle(combine)
        x[:], y[:] = zip(*combine)
        '''
        n_batch = len(text) // batch_size
        text = text[ : n_batch * batch_size]

        for i in range(0 , len(text) , batch_size):
            batch_x = x[i : i + batch_size]
            batch_y = y[i : i + batch_size]

            yield batch_x , batch_y

    def create_feed_dict(self , input_batch , label_batch):

        feed_dict = {self.input_placeholder : input_batch , self.label_placeholder : label_batch}
        return feed_dict

    def add_pred_op(self):
        self.W = tf.Variable(tf.truncated_normal(shape=(self.config.vocab_size, self.config.embedding_size), stddev=0.1))
        self.bias = tf.Variable(tf.zeros(self.config.vocab_size))
        self.embeddings = tf.Variable(shape=(self.config.vocab_size, self.config.embedding_size), stddev=0.1))
        pred = tf.nn.embedding_lookup(params=self.embeddings, ids=self.input_placeholder)
        return pred

    def add_loss_op(self , pred):
        
        loss = tf.nn.sampled_softmax_loss(weights=self.W, biases=self.bias, labels=label, inputs=pred,
                                    num_sampled=self.config.n_sample, num_classes=vocab_size)
        loss = tf.reduce_mean(loss)
        return loss

    def add_training_op(self , loss):
        train_op = tf.train.AdagradOptimizer().minimize(loss)
        return train_op
    
    def train_on_batch(self , sess , input_batch , label_batch):
        feed = self.create_feed_dict(input_batch , label_batch)
        _ , loss = sess.run([self.train_op , self.loss] , feed_dict = feed)

    def run_epopch(self):


    def fit(self , n_epoch , sess , inputs , labels):

        losses = []
        count = 0
        
        for epoch in range(self.config.epochs)：
            for x , y in self.get_batch(self.config.text , self.config.window_size , self.config.batch_size)

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
    

    def __init__ (self , config):
        self.config = config
        self.build()








'''
    valid_size = 16
    valid_examples = np.array(random.sample(range(vocab_size), valid_size))
    valid_size = len(valid_examples)
    # 验证单词集
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # 计算每个词向量的模并进行单位化
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embedding = embedding / norm
    # 查找验证单词的词向量
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    # 计算余弦相似度
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))
'''