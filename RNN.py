import numpy as np 

import activators
from activators import Sigmoid



class RNNCell(object):

    def __init__(self,input_size,state_size,activators = Sigmoid):
        self.activators = activators() 
        self.input_size = input_size
        self.state_size = state_size
        
        self.b = np.zeros([self.state_size,1])
        self.We = np.random.uniform(-1,1,[self.state_size,self.input_size]) 
        self.Wh = np.random.uniform(-1,1,[self.state_size,self.state_size])
        
        self.times = 0
        self.states = []
        self.states.append(np.zeros([self.state_size,1]))
        self.x_list = []
        self.x_list.append(np.zeros([self.input_size,1]))

    def reset(self):
        '''
        '''
        self.times = 0
        self.states = []
        self.x_list = []
        self.x_list.append(np.zeros([self.input_size,1]))
        self.states.append(np.zeros([self.state_size,1]))

    def forward(self , input_word):
        '''
        '''
        self.times += 1
        x = input_word
        self.x_list.append(x)
        state = np.dot(self.We,x) + np.dot(self.Wh,self.states[-1]) + self.b
        element_wise_op(state , self.activators.forward)
        self.states.append(state)
        return state

    def backward(self,last_delta):
        '''
        BPTT的实现
        '''
        next_delta = self.calc_delta(last_delta)
        self.calc_grad()
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
        self.delta_list = [np.zeros([self.state_size,1]) for i in range(self.times)]
        delta = self.states[-1].copy()
        element_wise_op(delta,self.activators.backward)
        self.delta_list.append(delta * last_delta)#先把t时刻的delta存进去
        for k in range(self.times - 1,0,-1):
            self.calc_k_delta(k)
        next_delta = np.dot(self.delta_list[-1].T,self.We).T
        return next_delta
    
    def calc_k_delta(self,k):
        '''
        k可以理解为t-1
        delta_list[k+1]是文档中t时刻的delta2\n
        state_list[k+1] = z_(t-1) = h_(t)\n
        np.dot(np.dot(self.delta_list[k+1].T, self.W),np.diag(state[:,0])).T
        '''
        gama = np.dot(self.delta_list[k+1].T,self.Wh)
        a = self.states[k+1].copy()
        element_wise_op(a,self.activators.backward)  
        k_delta = np.dot(gama,np.diag(a[:,0])).T
        return k_delta

    def calc_grad(self):
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
                self.calc_k_grad(self.delta_list[k],k)
        
        self.Wh_grad = np.sum(Wh_grad_list,axis = 0)
        self.We_grad = np.sum(We_grad_list,axis = 0)
        self.b_grad = np.sum(b_grad_list,axis = 0)

    def calc_k_grad(self,k_delta,k):

        We_k_grad = np.dot(k_delta,self.x_list[k].T)
        Wh_k_grad = np.dot(k_delta,self.states[k-1].T)
        b_k_grad = k_delta

        return We_k_grad,Wh_k_grad,b_k_grad

def element_wise_op(array, op):
    for i in np.nditer(array,op_flags=['readwrite']):
        i[...] = op(i)

class RNN(object):
    '''
    DeepRNN
    '''

    def __init__(self,layer_num,size):

        self.maxlength = 5
        self.pred_size = size[-1]
        self.state_size = size[-2]#还有输出的全连阶层要考虑
        self.pred_size
        self.U = np.random.uniform(-1,1,\
            [self.pred_size,self.state_size])
        self.b = np.random.uniform(-1,1,\
            [self.pred_size,1])
        self.layers = []
        for i in range(layer_num):
            self.layers.append(RNNCell(size[i],size[i+1]))
        self.states = []
            
    def train(self,epochs,input_batch,label_batch,learning_rate = 0.01):

        for i in range(epochs):
            batch_loss = 0
            for input_array,label_array in zip(input_batch,label_batch):
                for t in range(self.maxlength):
                    input_word = input_array[t].reshape([-1,1])
                    label_word = label_array[t].reshape([-1,1])
                    pred = self.Net_forward(input_word)
                    loss = self.calc_loss(pred,label_word)
                    self.Net_backward(pred,label_word)
                    self.update(learning_rate)
                    batch_loss += loss
                self.reset()
            print('epoch{},loss = {}'.format(i+1,batch_loss/100))

    def Net_forward(self,input_word):
        '''
        对序列进行一次forward(走一个字儿)
        return pred,state
        '''
        output = input_word

        for layer in self.layers:
            output = layer.forward(output)
        state = output
        pred = np.dot(self.U,state) + self.b
        pred = activators.Softmax().forward(pred)
        self.states.append(state)
        return pred

    def Net_backward(self,pred,label):
        '''
        RNN反传
        '''
        delta1 = label - pred
        # CE+softmaxd的反传结果delta1是真的简洁
 
        self.U_grad = np.dot(delta1,self.states[-1].T)
        self.b_grad = delta1

        error1 = np.dot(delta1.T,self.U).T
        error2 = self.layers[-1].activators.backward(self.states[-1])
        delta2 = error1 * error2
        delta = delta2
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)

    def update(self,learning_rate = 0.01):

        self.U -= learning_rate * self.U_grad
        self.b -= learning_rate * self.b_grad
        for layer in self.layers:
            layer.update(learning_rate)
    
    def calc_loss(self,pred,label):
        '''
        softmax + cross_entropy
        '''
        #assert labels.shape == preds.shape
        a = np.log(abs(label-pred))
        loss = -1 * np.sum(label * a)
        return loss

    def reset(self):
        for l in self.layers:
            l.reset()
        self.states = []


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
def RNN_test():
    net = RNN(3,[3,4,4,4,3])
    inputs_batch = [np.random.uniform(-1.0,1,[5,3]) for i in range(10)]
    labels_batch = [np.ones([5,3]) for i in range(10)]
    net.train(10,inputs_batch,labels_batch,0.01)

def Cell_grad_check():
    '''
    写了一天终于成了，干你妈的BPTT
    '''
    lc = RNNCell(3,2)
    x = [np.random.uniform(-1,1,[3,1]) for i in range(10)]
    for i in range(10):
        lc.forward(x[i])
  
    sensitivity_array = np.ones([2,1])
    lc.backward(sensitivity_array)
    epsilon = 10e-4

    print('for Wh:')
    for i in range(lc.We.shape[0]):
        for j in range(lc.We.shape[1]):
            lc.We[i,j] += epsilon
            lc.reset()
            for k in range(10):
                lc.forward(x[k])
            error1 = np.sum(lc.states[-1])
            lc.reset()
            lc.We[i,j] -= 2 * epsilon
            for k in range(10):
                lc.forward(x[k])
            error2 = np.sum(lc.states[-1])
            except_grad = (error1 - error2)/(2 * epsilon)
            lc.We[i,j] += epsilon
            #reldiff = abs(except_grad - lc.We_grad[i,j]) / max(1, abs(except_grad), abs(lc.We_grad[i,j]))
            print("We[{},{}]: clac_grad = {}  except_grad = {},".format(i+1,j+1,lc.We_grad[i,j],
            except_grad))
    lc.reset()

def net_grad_check():
    nc = RNN(2,[3,4,4,3])
    x = [np.random.uniform(-1,1,[3,1]) for i in range(2)]
    label = np.array([0,1,0]).reshape([-1,1])
    for k in range(2):
        pred = nc.Net_forward(x[k])
    nc.Net_backward(pred,label)
    epsilon = 1e-4
    #error_func = lambda o: o.sum()
    for i in range(nc.U.shape[0]):
        for j in range(nc.U.shape[1]):
            nc.U[i,j] += epsilon
            for k in range(2):
                pred = nc.Net_forward(x[k])
            error1 = nc.calc_loss(pred,label)
            nc.reset()
            nc.U[i,j] -= 2 * epsilon
            for k in range(2):
                pred = nc.Net_forward(x[k])
            error2 = nc.calc_loss(pred,label)
            except_grad = (error1 - error2)/(2 * epsilon)
            nc.U[i,j] += epsilon

            reldiff = abs(except_grad - nc.U_grad[i,j]) / max(1, abs(except_grad), abs(nc.U_grad[i,j]))
            print("reldiff = {}".format(reldiff))


if __name__ == "__main__":
    #RNN_test()
    Cell_grad_check()
    #net_grad_check()



