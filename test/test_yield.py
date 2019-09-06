

def _test():
    list1 = ['hello' , 'bye' , 'task' , 'trust']

    for i in range(len(list1)):
        x = list1[i]
    yield x

print(_test)

