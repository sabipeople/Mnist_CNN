import numpy as np
import pdb
class sigmoid(object):
    def __init__(self):
        self.x=None
        self.y=None
        self.dy=None
        self.learning_rate=0.1

    def sigmoid_func(self,x):
        return 1/(1 + np.exp(-x))
	
    def forward(self, din):
        self.x=din
        self.y=self.sigmoid_func(din)
        return self.y

    def backward(self, dout):
        self.dy=np.multiply(dout,np.multiply(self.y,1-self.y))
        return self.dy
