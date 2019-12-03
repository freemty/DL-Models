import numpy as np 

class Relu(object):
    def forward(self,inputs):
        return max(0,inputs)

    def backward(self,outputs):
        return 
class Sigmoid(object):
    def forward(self,inputs):
        return 1.0/(1.0 + np.exp(-inputs))

    def backward(self,outputs):
        return outputs * (1 - outputs)

class Softmax(object):
    def forward(self,inputs):
        inputs_sum = np.sum(np.exp(inputs),keepdims= True)
        return np.exp(inputs)/inputs_sum

    def backward(self):
        raise NotImplementedError
        
class Tanh(object):
    def forward(self,inputs):
        return 0

    def backward(self,outputs):
        return 0


class CE_Softmax(object):
    def forward(self,inputs,label):
        '''
        label必须是onehot
        '''
        inputs_sum = np.sum(np.exp(inputs),keepdims= True)
        pred = np.exp(inputs)/inputs_sum
        return np.sum(- label * np.log(pred))

    def backward(self,pred,label):
        return pred - label