# MLP

基本的多层感知机.
代码部分参考了[这里](https://www.zybuluo.com/hanbingtao/note/476663)

### Layer

建立一个单层layer的类,包含的基本成分有input,output,W,b,W    _grad,b_grad,last_delta.

#### forward

进行前向计算

``` python
output = activator.forward(W * input + bias)
```

#### backward

向后传播残差,计算梯度,更新权重

##### calc_grad

``` python
'''
这里只是伪代码
'''
grad_W = activator.backward(output) * last_delta * inputs
grad_b = activator.backward(output) * last_delta
delta = activator.backward(output) * last_delta * W

```

##### update_weight

``` python
W = W + grad_W #这里的grad_W是实际grad的相反数
b = b + grad_b
```

### Network

#### bulid

``` python
layers = []
for i in range(layer_num):
    layers.append(layer)
```

#### network_forward

逐层前向传播,没什么好说的

``` python
pred = input
for layer in layers:
    pred = layer.forward(pred)
loss = 0.5*((pred - label)*(pred - label)).sum()
```

#### network_backward

``` python
delta = label - pred
for layer in layers[::-1]:#for还有这种写法!!!
    delta = layer.backward(delta)
```

$\delta$是梯度的相反数,别忘了!

### train

``` python
for i in range(epochs):
    pred = network_forward(inputs)
    network_backward(labels,pred)
```

### Gradicent Check

**会写梯度检查太重要了!!!**
说起来丢人,这是第一次写梯度检查

``` python
layer.W[i,j] += epsilon
loss1 ,_ = self.forward(input_test,label_test)
layer.W[i,j] -= 2*epsilon
loss2 ,_ = self.forward(input_test,label_test)
layer.W[i,j] += epsilon
except_grad = (loss1 - loss2)/(2*epsilon)
```

## Train on MNIST

**loader写的太屎了,有空再改改**
基本跑起来了,第一个真正手撸的网络而不是完形填空或者搭积木(numpy不算XD),还是比较有成就感的