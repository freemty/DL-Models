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
        output = np.exp(inputs)/inputs_sum
        return output
    def backward(self,outputs):
        raise NotImplementedError

class Tanh(object):
    def forward(self,inputs):
        return 0

    def backward(self,outputs):
        return 0


if __name__ == "__main__":

    array1 = np.array([[1,1,3,5],[2,1,4,5]])

    print(Softmax().forward(array1))