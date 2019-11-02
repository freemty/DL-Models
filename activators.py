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
        return 

    def backward(self,outputs):
        return 0

class Tanh(object):
    def forward(self,inputs):
        return 

    def backward(self,outputs):
        return 


if __name__ == "__main__":

    print(Sigmoid().forward(5))