# 词嵌入 by Tensorflow

## DataProsess

### 洗文本

### 去高频 低频词

### 生成字典

## Skip_Gram Model

###

###

### 

## Others

### Python 补充

### 类继承

这部分的第一次接触是在CS224n Assignment 里tf的抽象层Model的写法
对于类继承比较详细的解释在[这里](https://www.liaoxuefeng.com/wiki/1016959663602400/1017497232674368)

#### set（）

set（）可创建集合
本次用于list去重

```python
list1 = ['hello' , 'hello' , 'bye']
set1 = set(list1)
print(list1 , set1)
```

``` shell
['hello', 'hello', 'bye'] {'bye', 'hello'}
```

#### emuneart()

迭代器,将一个可遍历的数据对象组合为一个索引序列，同时列出数据和索引

``` python
list1 = ['cat' , 'dog' , 'tiger']
for index , word in enumerate(list1):
    print(index , word)
```

``` shell
0 cat
1 dog
2 tiger
```

#### yield

生成器
