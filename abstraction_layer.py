def dl_model(object):

    def predict(self,inputs):
        raise NotImplementedError('Each Model must re-implement this method')

    def train(self,inputs,labels,learning_rate):
        raise NotImplementedError('Each Model must re-implement this method')

    def update_weights(self,inputs_x,pred,label):
        raise NotImplementedError('Each Model must re-implement this method')

    def run_epoch(self,inputs,labels):
        raise NotImplementedError('Each Model must re-implement this method')



    
