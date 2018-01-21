import numpy as np
import pdb
class affine:
    def __init__(self, rows,cols,batch_size,weight_std=0.01):
        self.w=weight_std*np.random.rand(rows,cols)#weight_std * np.random.randn(rows, cols)
        self.B=np.random.rand(1,cols)#np.zeros((batch_size, cols))
        self.x=None
        self.y=None
        self.dw=None
        self.dB=None
        self.dy=None
        self.learning_rate=0.01
	
    def forward(self,din):
        self.x=din
        self.y=np.dot(din,self.w)+self.B
        return self.y

    def backward(self, dout):
        self.dw=np.dot(self.x.transpose(),dout)
        self.dB=dout
        self.dy=np.dot(dout,self.w.transpose())
        return self.dy
    def update(self):
        self.w-=self.learning_rate*self.dw
        self.B-=self.learning_rate*np.sum(self.dB,axis=0)

    def update_learningrate(self, ratio):
        self.learning_rate *=ratio
