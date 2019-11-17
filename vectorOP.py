from functools import reduce

class VectorOp(object):

    @staticmethod
    def dot(x , y):
        return reduce(lambda a ,b:a + b,VectorOp.element_multiply(x,y),0.0)

    @staticmethod
    def element_multiply(x , y):

        return list(map(lambda z : z[0] * z[1], zip(x,y)))

    @staticmethod
    def element_add(x , y):
        return list(map(lambda z : z[0] + z[1], zip(x,y)))
    
    @staticmethod
    def scala_multiply(v,s):
        return map(lambda e:e * s, v)
