import time
import random
import numpy as np 
import tensorflow as tf 
from model import Model


class Skip_Gram(Model):

    def add_placeholder(self):
        self.input_placeholder = tf.placeholder(shape = (None , None) , dtype = tf.int32 , name = 'input')
        self.label_placeholder = tf.placeholder(shape = (None) , dtype = tf.int32 , name = "label")

    def get_windows_word(self , word_id):
        start = word_id - self.config.windows_size if word_id - self.config.windows_size >=0 else 0
        end = word_id + self.config.windows_size if word_id + self.config.windows_size > len(self.config.text) else (len(self.config.text) - 1)
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

    def normalized_embedding(self):
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings)),keep_dims= Trus , axis = 1))
        normalized_embedding = self.embeddings / norm
        return normalized_embedding


    def conpute_similarity(self):
        valid_examples = np.array(random.sample(range(self.config.vocab_size) ,self.config.valid_size))

        valid_dataset = tf.constant(valid_examples , dtype = tf.int32)

        normalized_embedding = self.normalized_embedding
        valid_embedding = tf.nn.embedding_lookup(normalized_embedding,valid_dataset)
        similarity = tf.matmul(valid_embedding , tf.transpose(normalized_embedding))

        for i in range(self.config.valid_size):


            print(log)
    

    def run_epoch(self , sess , saver):
        self.epoch += 1
        for x, y in self.get_batch():

            start = time.time()
            self.count += 1
            loss = self.train_on_batch(sess , x , y)
            end = time.time()

            if count % 300 == 0:
                print(epoch {} / {}).format(self.epoch , self.config.epochs)
                print(count {}).format(self.count)

            if count % 500 == 0:
                self.conpute_similarity(sess)

            if count % 500 == 0:
                saver.save(sess, "checkpoints/model.ckpt", global_step=count)
                embed_mat = self.normalized_embedding()
                print('----------------')
                print(type(embed_mat))
                print('- - - - - - - - ')
                print(embed_mat)
                print('- - - - - - - - ')


    def fit(self , sess , saver):
        self.count = 0
        self.epoch = 0
        for i in range(self.config.epochs):
            start_time = time.time()

            loss = self.run_epoch(sess , saver)


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

 count = 0
        for epoch in range(epochs):
            #获取每个batch
            for x, y in get_batch(text, windows_size, batch_size):
                start = time.time()

                count += 1
                feed = {
                    input:x,
                    label:np.array(y)[:, None]
                }
                #训练
                batch_loss, _ = sess.run([loss, optimizer], feed_dict=feed)
                end = time.time()

                #定期打印数据
                if count % 300 == 0:
                    print('epoch:%d/%d'%(epoch, epochs))
                    print('count:', count)
                    print('time span:', end - start)
                    print('loss:', batch_loss)

                #定期打印文本学习情况
                if count % 500 == 0:
                    # 计算similarity
                    sim = similarity.eval()
                    for i in range(valid_size):
                        valid_word = int_to_vocab[valid_examples[i]]
                        top_k = 8  # 取最相似单词的前8个
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log = 'Nearest to [%s]:' % valid_word
                        for k in range(top_k):
                            close_word = int_to_vocab[nearest[k]]
                            log = '%s %s,' % (log, close_word)
                        print(log)

                if count % 500 == 0:
                    saver.save(sess, "checkpoints/model.ckpt", global_step=count)
                    embed_mat = sess.run(normalized_embedding)
                    print('----------------')
                    print(type(embed_mat))
                    print('- - - - - - - - ')
                    print(embed_mat)
                    print('- - - - - - - - ')

'''