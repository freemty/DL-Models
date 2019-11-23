import numpy as np 
import sys
sys.path.append('../')
 
from Models import activators
from Models.activators import Sigmoid

class RNNCell(object):

    def __init__(self,input_size,state_size,activators = Sigmoid):
        self.activators = activators 
        self.input_size = input_size
        self.state_size = state_size
        
        self.b = np.random.uniform(-1e-4,1e-4,[self.state_size,1])
        self.We = np.random.uniform(-1e-4,1e-4,[self.state_size,self.input_size]) 
        self.Wh = np.random.uniform(-1e-4,1e-4,[self.state_size,self.state_size])
        
        self.times = 0
        self.states = []
        self.states.append(np.zeros([self.state_size,1]))
        self.x_list = []
        self.x_list.append(np.zeros([self.input_size,1]))

    def reset(self):
        '''
        '''
        self.time = 0
        self.statas = []
        self.x_list = []
        self.x_list.append(self.zeros([self.input_size,1]))
        self.states.append(self.zeros([self.state_size,1]))

    def forward(self , input_array):
        '''
        '''
        self.times += 1
        x = input_array
        self.x_list.append(x)
        state =np.dot(self.We,x) + np.dot(self.Wh,self.states[-1]) + self.b
        self.states.append(state)

    def backward(self,last_delta):
        '''
        BPTT的实现
        '''
        next_delta,delta_list = self.calc_delta(last_delta)
        self.calc_grad(delta_list)
        return next_delta

    def update(self,learning_rate):
        '''
        更新梯度
        '''
        self.We -= learning_rate * self.We_grad
        self.Wh -= learning_rate * self.Wh_grad
        self.b -= learning_rate * self.b_grad

        self.b_grad = np.zeros([self.state_size,1])
        self.We_grad = np.zeros([self.state_size,self.input_size])
        self.Wh_grad = np.zeros([self.state_size,self.input_size])

    def calc_delta(self,last_delta):
        '''
        计算t时刻的delta,为0到t-1的delta之和
        t时刻共有t+1个delta
        '''
        delta_list = []
        for t in range(self.times+1):
            delta_list.append(np.zeros([self.state_size,1]))
        delta_list[-1] = last_delta.copy()#先把t时刻的delta存进去
        delta = last_delta
        next_delta = np.dot(delta,self.We)
        for k in range(self.times - 1,0 -1):
            delta = self.calc_k_delta(delta,k)
            delta_list[k-1] = delta
            #这是t-2时刻的delta
        return next_delta,delta_list

    def calc_k_delta(self,last_delta,k):
        '''
        k可以理解为t-1
        '''
        gama = np.dot(self.Wh.T,last_delta)
        error = self.activators.backward(self.statas[k])
        k_delta = np.dot(gama.T,error)
        return k_delta

    def calc_grad(self,delta_list):
        '''
        计算梯度
        '''
        We_grad_list = []
        Wh_grad_list = []
        b_grad_list = []
        for t in range(self.times + 1):
            Wh_grad_list.append(np.zeros(self.Wh.shape))
            We_grad_list.append(np.zeros(self.We.shape))
            b_grad_list.append(np.zeros(self.b.shape))
        for k in range(self.times , 0, -1):
            We_grad_list[k],Wh_grad_list[k],b_grad_list[k] = \
                self.calc_k_grad(delta_list[k],k)
        
        self.Wh_grad = np.sum(Wh_grad_list,axis = 0)
        self.We_grad = np.sum(We_grad_list,axis = 0)
        self.b_grad = np.sum(b_grad_list,axis = 0)

    def calc_k_grad(self,k_delta,k):

        We_k_grad = np.dot(k_delta,self.x_list[k].T)
        Wh_k_grad = np.dot(k_delta,self.states[k-1].T)
        b_k_grad = k_delta

        return We_k_grad,Wh_k_grad,b_k_grad

class RNN(object):
    '''
    DeepRNN
    '''

    def __init__(self,layer_num,size):

        self.maxlength = 10
        self.state_size = size[-1]
        self.U = np.random.uniform(-1e-4,1e-4,\
            [self.state_size,size[-2]])
        self.b = np.random.uniform(-1e-4,1e-4,\
            [self.state_size,1])

        self.layers = []
        for i in range(layer_num):
            self.layers.append(RNNCell(size[i],size[i+1]))
            
    def train(self,epochs,input_batch,label_batch,learning_rate = 0.01):

        for i in range(epochs):
            batch_loss = 0
            for input_array,label_array in zip(input_batch,label_batch):
                self.states = []
                for t in self.maxlength:
                    pred = self.Net_forward(input_array,label_array)
                    loss = self.clac_loss(pred,label_array[t])
                    self.Net_backward(loss)
                    self.update(learning_rate)
                    batch_loss += loss
            print('epoch{},loss = {}'.format(i+1,batch_loss/100))

    def Net_forward(self,input_array,label_array):
        '''
        对序列进行一次forward(走一个字儿)
        '''
        output = input_array

        for layer in self.layers:
            output = layer.forward(output)
            state = output
            self.states.append(state)
        pred = np.dot(self.U,state) + self.b
        return pred

    def Net_backward(self,pred,label):
        '''
        RNN反传
        '''
        for t in range(self.maxlength):
            delta1 = label - pred
            # CE+softmaxd的反传结果delta1是真的简洁
            self.U_grad = np.dot(delta1,self.states[-1])
            self.b_grad = delta1

            error1 = np.dot(delta1 , self.U)
            error2 = self.states[-1] * (1-self.states[-1])
            delta2 = np.dot(error1 , error2)
            delta = delta2
            for layer in layers[::-1]:
                delta = layer.backward(delta)
    def update(self,learning_rate = 0.01):

        self.U -= learning_rate * self.U_grad
        self.b -= learning_rate * self.b_grad
        for layer in self.layers:
            layer.update(learning_rate)
    
    def calc_loss(self,preds,labels):
        '''
        softmax + cross_entropy
        '''
        #assert labels.shape == preds.shape
    
        activators.Softmax(preds)
        loss = -(np.log(preds - labels)).sum
        return loss

def grad_check
        

def cell_test():
    l = RNNCell(3,2)
    x = [np.array([[1], [2], [3]]),
         np.array([[2], [3], [4]])]
    d = np.array([[1], [2]])
    l.forward(x[0])
    print(l.states[-1])
    l.forward(x[1])
    print(l.states[-1])
    l.backward(d)


def RNN_test():


if __name__ == "__main__":
    layer_test()
    RNN_test()
