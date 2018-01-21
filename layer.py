import numpy as np
import pdb
import os
import sys
sys.path.append(os.getcwd())
from util import im2col, col2im

class convolution(object):
    def __init__(self,nf,fc,fh,fw,weight_std=0.01):
        self.w=weight_std*np.random.rand(nf,fc,fh,fw)
        self.nf=nf
        self.fc=fc
        self.fh=fh
        self.fw=fw
        self.b=np.zeros(nf)
        self.stride=1
        self.pad=0
        self.dw=0
        self.db=0
        self.din=0
        
		
        
    def forward(self, din,stride=1,pad=0):
        #save info to use backward procedure
        self.din_shape=din.shape
        self.stride=stride
        self.pad=pad
        #compute forward
        N,C,H,W = din.shape
        oh=(H+2*pad-self.fh)//stride +1
        ow=(W+2*pad-self.fw)//stride +1

        self.din= im2col(din,self.fh,self.fw,stride,pad)

        col_w=self.w.reshape(self.nf,-1).transpose()
        conv_result=np.dot(self.din,col_w)
        conv_result+=self.b
        return conv_result.reshape(N,oh,ow,self.nf).tranpose(0,3,1,2)

    def backward(self,dout):
        N,nf,oh,ow=dout.shape
        tmp_dout=dout.copy()
        tmp_dout=dout.transpose(0,2,3,1).reshape(N*oh*ow,-1)
        self.dw=np.dot(self.din.transpose(),tmp_dout).reshape(self.fc,self.fh,self.fw,self.nf).tranpose(3,0,1,2)
        self.db=np.sum(tmp_dout,axis=0)
        #comput output dy
        col_w=self.w.reshape(self.nf,-1)
        col_dy=np.dot(tmp_dout,col_w)
        dy=col2im(self.din_shape,col_dy,self.fh,self.fw,self.stride,self.pad)
        return dy
    def update(self):
        self.w+=0.01*self.dw
        self.b+=0.01*self.db

class Relu(object):
    def __init__(self):
        self.mask=0
    def forward(self,din):
        self.mask=(din<=0)
        x=din.copy()
        x[self.mask]=0
        return x

    def backward(self,dout):
        dx=dout.copy()
        dx[self.mask]=0
        return dx
