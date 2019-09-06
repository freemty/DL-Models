import time
import string
import math
from collections import Counter

class Config(object):
    epochs = 10
    embedding_size = 300
    windows_size = 5
    batch_size = 100
    text = None


fileName = 'Harry_Potter1-7.txt'
low_threshold = 10
high_freq_threshold = 0.85

text = open(fileName).read().lower()


for c in string.punctuation:    # ， . : ; !
    text = text.replace(c, ' ')


new= open('new_text.txt' , 'w')
new.write(text)

text = text.split()

wordCount = Counter(text)
text = [word for word in text if wordCount[word] >= low_threshold]
high_freq = 1e-3
wordCount = Counter(text)
totalCount = len(text)
word_frep = {word: (1 - math.sqrt(high_freq / (count / totalCount))) for word, count in wordCount.items()}
#保留非高频词
text = [word for word in text if word_frep[word] < high_freq_threshold]

#创建字典
vocab = set(text)
vocab_List = list(vocab)
vocab_List.sort()

#创建两种索引的词典
word2Int = {word:index for index, word in enumerate(vocab_List)}
int2Word1 = {index : word for index , word in enumerate(vocab_List)}
int2Word2 = {index : word for word , index in word2Int.items()}
'''
int2Word = {index: word for word, index in word2Int.items()}
'''

for i in range(100):
    print(int2Word1[i] , int2Word2[i] )

#对文本进行编码
encode = [word2Int[word] for word in text]

config = Config()

config.text = encode

print(config.text)
print('Done!')