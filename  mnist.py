import numpy as np 


def load_minst(filepath):
    dataArr,labelArr = [],[]

    print('start Loading')

    with open(filepath,'r') as f:
        for line in f.readlines:

            data_line = line..split('')
            if int(data_lime[0]) > 5:
                labelArr.append(1)
            else:
                labelArr.append(-1)
            dataArr.append([int(num) for num in dataline[1:]])

    return dataArr,labelArr


