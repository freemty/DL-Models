import numpy as np 

import numpy as np 
import sys

from activators import Sigmoid
from abstraction_layer import dl_model 
from MNIST import MNIST_loader



class layer(object):

    def __init__(self,intput_size,output_size,activators = Sigmoid):
        self.intput_size = intput_size
        self.output_size = output_size
        self.activators = activators()#加括号妈的

        self.W = np.random.uniform(-0.1,0.1,[output_size,intput_size])
        self.b = np.zeros([output_size,1])
        self.output = np.zeros([output_size,1])

    def forward(self,inputs):

        self.inputs = inputs
        z = np.dot(self.W,self.inputs) + self.b

        self.output = self.activators.forward(z)
        return self.output

    def backward(self,last_delta,leaening_rate = 0.01):

        self.calc_grad(last_delta)
        self.update_weights(leaening_rate)
        return self.delta

    def calc_grad(self,last_delta):
        self.W_grad = np.zeros(self.W.shape)
        self.b_grad = np.zeros(self.b.shape)

        error = self.activators.backward(self.output) * last_delta #这里算delta得细心
        self.delta = np.dot(self.W.T,error)
        self.W_grad = np.dot(error,self.inputs.T)
        self.b_grad = error
        
    
    def update_weights(self,learning_rate):
        self.W += learning_rate *self.W_grad
        self.b_grad += learning_rate*self.b_grad

    def dump(self):
        print('W:{}\nb:{}\n'.format(self.W,self.b))

class MLP():
    def __init__(self,layer_size):
        self.layers = []
        for i in range(len(layer_size) - 1):
            self.layers.append(layer(layer_size[i] , layer_size[i+1] ,Sigmoid))

    def forward(self,input_data,label):
        '''
        多层网络的前向传播
        返回loss , pred
        '''
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        pred = output
        loss = 0.5*((label - pred)*(label - pred)).sum()
        return loss , pred
           
    def backward(self,label,pred,learning_rate = 0.01):
        '''
        '''
        delta = label - pred
        for layer in self.layers[::-1]:
            delta = layer.backward(delta,learning_rate)
        return delta

    def train(self,inputs,labels,epochs = 10,learning_rate = 0.01,\
        eva = True,test_images = None , test_labels = None):
        '''
    
        '''
        for i in range(10):
            batch_loss = 0
            for batch in zip(inputs,labels):
                input_data = batch[0].reshape([-1,1])
                label = batch[1].reshape([-1,1])
                loss,pred = self.forward(input_data,label)
                delta = self.backward(label,pred,learning_rate)
                batch_loss += loss

            print('epoch = {},loss = {}'.format(i + 1,batch_loss/60000))
    
    def gradicent_check(self,input_test,label_test):
        '''
        MLP内的梯度检查
        '''
        
        _,pred = self.forward(input_test,label_test)
        _ = self.backward(label_test,pred)

        epsilon = 10e-4
        for n,layer in enumerate(self.layers):
            for i in range(layer.W.shape[0]):
                for j in range(layer.W.shape[1]):
                    layer.W[i,j] += epsilon
                    loss1 ,_ = self.forward(input_test,label_test)
                    layer.W[i,j] -= 2*epsilon
                    loss2 ,_ = self.forward(input_test,label_test)
                    layer.W[i,j] += epsilon
                    except_grad = (loss1 - loss2)/(2*epsilon)
                    print("layer{},W[{},{}] W_grad = {},except_grad = {}".\
                        format(n+1,i+1,j+1,[layer.W_grad[i,j]],[except_grad]))




def model_test():
    test = MLP([3,5,5,1])
    for i in range(3):
        print('layers:{},W = {}'.format(i+1,test.layers[i].W))

def gradicent_check():
    net = MLP([5,3,5])
    labels_test = np.array([0.1,0.2,0.3,0.5,0.1]).reshape([-1,1])
    inputs_test = np.array([0.1,0.3,0.3,0.2,0.5]).reshape([-1,1])
    net.gradicent_check(inputs_test,labels_test)

def train_on_mnist():
    iamges , labels , test_images , test_labels= MNIST_loader()

    

    Network = MLP([784,300,300,10])

    Network.train(iamges,labels)




if __name__ == "__main__":
    #model_test()
    #gradicent_check()
    train_on_mnist()

