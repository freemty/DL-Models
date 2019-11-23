from vectorOP import VectorOp
from activators import Sigmoid
#from __future__ import print_function
from functools import reduce

#单层感知机,并进行逻辑拟合

class Perceptron():

    def __init__(self,input_num,activator,learning_rate = 0.01):
        self.activator = activator()
        self.weights = [1.0] * input_num
        self.bias = 1.0

        self.learning_rate = learning_rate

    def predict(self,inputs):
        pred = self.activator.forward(VectorOp.dot(inputs , self.weights) + self.bias)
        return pred

    def train(self,epochs,inputs,labels):

        for i in range(epochs):
            self.run_epoch(inputs,labels)

    def run_epoch(self,inputs,labels):

        batch = zip(inputs,labels)

        for (x , y) in batch:
            pred = self.predict(x)
            self.update_weights(x,pred,y)
            print('inputs = {},pred = {},label ={}'.format(list(x),pred,y))

    def update_weights(self,input_x,pred,label):

        loss = 0.5 * (pred - label) * (pred - label)
        delta = self.activator.backward(loss)
        self.weights = VectorOp.element_add(self.weights , VectorOp.scala_multiply(input_x, self.learning_rate*delta) )
        self.bias += self.learning_rate * delta
        print('loss = {}'.format(loss))

def get_dataset(logtic = 'and'):

    if logtic == 'and':
        inputs = [[1,1],[1,0],[0,1],[0,0]]
        labels = [1,0,0,0]
    elif logtic == 'or':
        inputs = [[1,1],[1,0],[0,1],[0,0]]
        labels = [1,1,1,0]       
    elif logtic == 'xor':
        inputs = [[1,1],[1,0],[0,1],[0,0]]
        labels = [0,1,1,0]
    else:
        raise ValueError('不支持的操作')
    return inputs,labels

def logtic_test():

    inputs,labels =  get_dataset(logtic = 'and')
    p = Perceptron(2,Sigmoid,1)
    p.train(1000,inputs,labels)
    return p

if __name__ == "__main__":
    
    P = logtic_test()
    print(P)
    print('Done!')








