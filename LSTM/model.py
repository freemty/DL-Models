import tensorflow as tf 
import numpy as np 


def loadData(fileName):
    text = open(fileName , encoding = 'utf-8').read()
    vocab = set(text)
    vocabList = list(vocab)
    vocabList.sort()
    vocab2Int = {word : index for index , word in enumerate(vocabList)}
    int2Vocab = {index : word for word , index in vocab2Int.items() } 


    return vocab , vocab2Int , int2Vocab , encode

def get_Batch(input_data , n_seqs , n_sequendcd_length):



def model_Input(n_seqs , n_sequendcd_length):


    return input , target

def model_LSTM(lstm_num_units , keep_prob , num_layers , n_seqs):

    for i in range(num_layers):
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = lstm_num_units)

        drop = tf.nn.rnn_cell.DropoutWrapper(cell = cell , output_keep_prob = keep_prob)

        lstms.append(drop)

    cell = tf.nn.rnn_cell.MultiRNNCell(lstms)

    init_state = cell.zero_state(dtype = tf.float32 , batch_size = n_seqs)

    return cell , init_state

def model_Output(lstm_output , in_size , out_size):


    return output , logits


def model_Loss(target , logits , num_class):


    return loss

def model_Optimizer(learning_rate , loss , clip_val):

    train_op = tf.train.AdamOptimizer(learning_rate = learning_rate)

    allTvars = tf.trainable_variables()

    allGradient,_ = tf.clip_by_global_norm( , clip_norm = clip_val)  

    return optimizer



class char_RNN():
    def __init__(self ,vocab,n_seqs = 10 ,n_sequendcd_length = 30 ,lstm_num_units = 128 ,
                 keep_prob = 0.5 , num_layers = 3 ,learning_rate = 0.01 ,clip_val = 5):
        
        self.input,self.target = module_Input()

        input_one_hot = tf.one_hot(self.input , len(vocab))

        cell , self.init_state = model_LSTM(lstm_num_units = lstm_num_units , keep_prob = keep_prob ,\
             num_layers = num_layers , n_seqs = n_seqs)

        self.loss = model_Loss()

        self.optimizer = model_Optimizer(learning_rate = learning_rate , loss = self.loss , clip_val = clip.val )
        
    