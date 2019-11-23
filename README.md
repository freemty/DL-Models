# Basic Models

这段时间沉迷用tensorflow做小玩具不可自拔,但说实话这么久了,对于很多基本模型的基本解构浅尝辄止,不管什么数据,就直接tf.nn.rnn,优化只会Adamoptimizer,过拟合就nn.dropout.仔细想想,没了tensorflow的牛逼API,自己什么也写不出来,这样下去和调参侠也没什么区别里,未来一段时间有必要开始手撸一下基本模型了(好在也有几个不错的轮子可供参考),预计会写的有

    机器学习
    《统计学习方法》里每一章的算法

    深度学习基本模型
    MLP,RNN,CNN,LSTM

## MLP

基本的多层感知机.
代码部分参考了[这里](https://www.zybuluo.com/hanbingtao/note/476663)
第一次手写了全连接层和多层网络,附带grad_check,实现了run on MNIST,虽然很简单,但还是很有成就感的

## reference

    https://github.com/hanbt/learn_dl
    https://github.com/Dod-o/Statistical-Learning-Method_Code