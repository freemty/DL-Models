
# RNN

## Neutral Network

$$e^{(t)} = Ex^{(t)}$$

$$ h = \sigma(W_hh^{(t - 1)} + W_ee^{(t)} + b_1) $$

$$ \hat{y} = softmax(Uh^{(t)} + b_2) $$

$$ J^{(t)} = CE(y^{(t)},\hat{y}^{(t)}) =
\sum_{j = 1}^Vy_j^{(t)}ln\hat{y_j}^{(t)}$$
疑惑度:
$$ P(|)$$

## Calc the Derivative

RNN最重要的两个梯度
$$
\frac{\partial J^{(t)}}{\partial W_h}|_{(t)} =
\sum_{i = 0}^t\frac{\partial J^{(t)}}{\partial W_h}|_{(i)}
$$

$$
\frac{\partial J^{(t)}}{\partial W_e}|_{(t)} =
\sum_{i = 0}^t\frac{\partial J^{(t)}}{\partial W_e}|_{(i)}
$$

记

``` python
'''
V: vocab size
Dh:hidden laver size
d: embedding size
B: batch size
'''
x.shape = [V,B]
E.shape = [d,V]
e.shape = [d,B]
U.shape = [V,Dh]
W_h.shape = [Dh,Dh]
W_e.shape = [Dh,d]
z.shape = [Dh,B]
b1.shape = [Dh,1]
h.shape = [Dh,B]
b2.shape = [V,1]
y_hat.shape = [V,B]

```

### 对于t时刻

先定义
$$ \theta^{(t)} = Uh^{(t)} + b_2$$

$$ z^{(t)} = W_hh^{(t - 1)} + W_ee^{(t)} + b_1$$

$$
\delta_1^{(t)} =
\frac{\partial J^{(t)}}{\partial \theta} =
{(\hat{y}^{(t)} - y^{(t)})}^T$$

$$
\delta_2^{(t)} =
\frac{\partial J}{\partial z} = \delta_1^{(t)}
\frac{\partial \theta}{\partial h^{(t)}}
\frac{\partial h^{(t)}}{\partial z^{(t)}} =
\delta_1^{(t)}U z \otimes (1-z)
$$

```python
delta1.shape = [B,V]
delta2.shape = [B,Dh]
```

算出
$$
\frac{\partial J^{(t)}}{\partial U} =
\frac{\partial J^{(t)}}{\partial \theta}
\frac{\partial \theta}{\partial U} =
{\delta_1^{(t)}}^T{h^{(t)}}^T
$$

$$
\frac{\partial J^{(t)}}{\partial e^{(t)}}=
\delta^{(t)}_2
\frac{\partial z^{(t)}}{\partial e^{(t)}}=
{\delta^{(t)}_2}{W_e}
$$

    Solotion 里把分母布局的的结果又转置了一下,为了和e的shape相同

$$
\frac{\partial J^{(t)}}{\partial W_e}|_{(t)} =
\delta^{(t)}_2\frac{\partial z^{(t)}}{\partial W_e} =
{\delta^{(t)}_2}^T{e^{(t)}}^T
$$

$$
\frac{\partial J^{(t)}}{\partial W_h}|_{(t)} =
\delta^{(t)}_2\frac{\partial z^{(t)}}{\partial W_h} =
\delta^{(t)}_2{h^{(t-1)}}^T
$$

### 对于t-1时刻

先定义
$$
\gamma^{(t-1)} =
\frac{\partial J^{(t)}}{\partial h^{(t-1)}} =
\delta_2^{(t)}\frac{\partial z^{(t)}}{\partial h^{(t-1)}} =
W_h^T\delta_2^{(t)}
$$

$$
\sigma'(h^{(t-1)}) = z^{(t-1)}\circ (1-z^{(t-1)})
$$

``` python
gamma1.shape = [B,V]
```

算出

$$
\frac{\partial J^{(t)}}{\partial e^{(t-1)}} =
\frac{\partial J^{(t)}}{\partial h^{(t-1)}}
\frac{\partial h^{(t-1)}}{\partial z^{(t-1)}}
\frac{\partial z^{(t-1)}}{\partial e^{(t-1)}} =
{\gamma^{(t-1)}}^T \sigma'(h^{(t-1)}){W_e}
$$

$$
\frac{\partial J^{(t)}}{\partial W_e}|_{(t-1)} =
\frac{\partial J^{(t)}}{\partial h^{(t-1)}}
\frac{\partial h^{(t-1)}}{\partial z^{(t-1)}}
\frac{\partial z^{(t-1)}}{\partial W_e} =
\gamma^{(t-1)} \sigma'(h^{(t-1)}) {e^{(t-1)}}
$$

$$
\frac{\partial J^{(t)}}{\partial W_h}|_{(t-1)} =
\frac{\partial J^{(t)}}{\partial h^{(t-1)}}
\frac{\partial h^{(t-1)}}{\partial z^{(t-1)}}
\frac{\partial z^{(t-1)}}{\partial W_h} =
{\gamma^{(t-1)}}^T \sigma'(h^{(t-1)}){h^{(t-2)}}^T
$$

## 梯度爆炸的问题

## 矩阵相关

### 转置问题

分母布局下,把分母看作行向量或分子看作列向量.
分子布局下,把分子看作列向量或分子看作行向量.
*在cs224n以至大多ML课程中,均默认采用分母布局*
计算的时候,一定在摆出链式法则之后,先把计算顺序调整好,再根据形式确定套哪个公式(草)

#### 几个重要结论

    以下结论是分母布局下
    以下分母的X均被看作行向量

$$
\frac{\partial a^TXb}{\partial X} = \frac{\partial ba^TX}{\partial X} = ab^T
$$

$$
\frac{\partial a^TX^Tb}{\partial X} =\frac{\partial ba^TX^T}{\partial X} =  ba^T
$$
