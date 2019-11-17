import numpy as np 
import struct
import pickle



def MNIST_loader():
    train_image_path = 'data/mnist/train-images.idx3-ubyte'
    train_label_path = 'data/mnist/train-labels.idx1-ubyte'
    test_image_path = 'data/mnist/t10k-images.idx3-ubyte'
    test_label_path = 'data/mnist/t10k-labels.idx1-ubyte'


    with open(train_image_path,'rb') as f:
        content = f.read()
        print('loading train_image_set')
        magic_num,image_num,rows_num,columns_num = struct.unpack_from('>iiii',content,offset = 0)
        print('magic num:{},images num:{},rows:{},columns:{}'.format(magic_num,image_num,rows_num,columns_num))

        image_size = rows_num * rows_num
        fmt_image = '>'+str(image_num * image_size)+'B'
        offset = struct.calcsize('>IIII')
        images_bytes = struct.unpack_from(fmt_image, content ,offset)
        images = np.reshape(images_bytes,[image_num , image_size])
        f.close()

    with open(train_label_path,'rb') as f:
        content = f.read()
        print('loading train_label_set')
        magic_num,labels_num= struct.unpack_from('>ii',content,offset = 0)
        print('magic num:{},labels num:{}'.format(magic_num,labels_num))
        fmt_label = '>' + str(labels_num) + 'B'
        offset = struct.calcsize('>II')
        labels_bytes = struct.unpack_from(fmt_label,content,offset)
        labels = np.reshape(labels_bytes,[-1])

    print('Doen!')

    one_hot = np.zeros([labels_num,10])
    for i,ids in enumerate(labels):
        one_hot[i,ids] = 1
    labels = one_hot

    return images,labels




#print(np.reshape(images[2,:],[28,28]))






'''

[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  67 232  39   0   0   0   0   0]
[  0   0   0   0  62  81   0   0   0   0   0   0   0   0   0   0   0   0   0   0 120 180  39   0   0   0   0   0]
[  0   0   0   0 126 163   0   0   0   0   0   0   0   0   0   0   0   0   0   2 153 210  40   0   0   0   0   0]
[  0   0   0   0 220 163   0   0   0   0   0   0   0   0   0   0   0   0   0  27 254 162   0   0   0   0   0   0]
[  0   0   0   0 222 163   0   0   0   0   0   0   0   0   0   0   0   0   0 183 254 125   0   0   0   0   0   0]
[  0   0   0  46 245 163   0   0   0   0   0   0   0   0   0   0   0   0   0 198 254  56   0   0   0   0   0   0]
[  0   0   0 120 254 163   0   0   0   0   0   0   0   0   0   0   0   0   23 231 254  29  0   0   0   0   0   0]
[  0   0   0 159 254 120   0   0   0   0   0   0   0   0   0   0   0   0 163 254 216  16   0   0   0   0   0   0]
[  0   0   0 159 254  67   0   0   0   0   0   0   0   0   0  14  86 178 248 254  91   0   0   0   0   0   0   0]
[  0   0   0 159 254  85   0   0   0  47  49 116 144 150 241 243 234 179 241 252  40   0   0   0   0   0   0   0]
[  0   0   0 150 253 237 207 207 207 253 254 250 240 198 143  91  28   5 233 250   0   0   0   0   0   0   0   0]
[  0   0   0   0 119 177 177 177 177 177  98  56   0   0   0   0   0 102 254 220   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 169 254 137   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 169 254  57   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 169 254  57   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 169 255  94   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 169 254  96   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 169 254 153   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 169 255 153   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  96 254 153   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]
[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0]]

'''