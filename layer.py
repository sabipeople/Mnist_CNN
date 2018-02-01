import numpy as np
import pdb
import os
import sys
sys.path.append(os.getcwd())
from util import im2col, col2im
class affine:
    def __init__(self, insize,outsize,weight_std=0.01):
        self.w=weight_std*np.random.rand(insize,outsize)#weight_std * np.random.randn(rows, cols)
        self.B=np.random.rand(1,outsize)#np.zeros((batch_size, cols))
        self.x=None
        self.y=None
        self.dw=None
        self.dB=None
        self.dy=None
        self.learning_rate=0.01
	
    def forward(self,din):
        self.original_shape=din.shape
        self.x=din.reshape(din.shape[0],-1)
        self.y=np.dot(self.x,self.w)+self.B
        return self.y

    def backward(self, dout):
        self.dw=np.dot(self.x.transpose(),dout)
        self.dB=np.sum(dout,axis=0)
        self.dy=np.dot(dout,self.w.transpose())
        #reshape to original shape
        self.dy=self.dy.reshape(self.original_shape)
        return self.dy

    def update(self):
        self.w-=self.learning_rate*self.dw
        self.B-=self.learning_rate*np.sum(self.dB,axis=0)

    def update_learningrate(self, ratio):
        self.learning_rate *=ratio

class convolution(object):
    def __init__(self,nf,fc,fh,fw,stride=1,pad=0,weight_std=0.01):
        self.w=weight_std*np.random.rand(nf,fc,fh,fw)
        self.nf=nf
        self.fc=fc
        self.fh=fh
        self.fw=fw
        self.b=np.zeros(nf)
        self.stride=stride
        self.pad=pad
        self.dw=0
        self.db=0
        self.din=0
        
    def forward(self, din):
        #save info to use backward procedure
        self.din_shape=din.shape
        #compute forward
        N,C,H,W = din.shape
        oh=(H+2*self.pad-self.fh)//self.stride +1
        ow=(W+2*self.pad-self.fw)//self.stride +1

        self.din= im2col(din,self.fh,self.fw,self.stride,self.pad)

        col_w=self.w.reshape(self.nf,-1).transpose()
        conv_result=np.dot(self.din,col_w)
        conv_result+=self.b
        return conv_result.reshape(N,oh,ow,self.nf).transpose(0,3,1,2)

    def backward(self,dout):
        N,nf,oh,ow=dout.shape
        tmp_dout=dout.copy()
        tmp_dout=dout.transpose(0,2,3,1).reshape(N*oh*ow,-1)
        self.dw=np.dot(self.din.transpose(),tmp_dout).reshape(self.fc,self.fh,self.fw,self.nf).transpose(3,0,1,2)
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
class Maxpool(object):
    def __init__(self,fh,fw,stride=1,pad=0):
        self.mask=0
        self.fh=fh
        self.fw=fw
        self.stride=stride
        self.pad=pad

    def forward(self,din):
        self.din_shape=din.shape
        if len(self.din_shape )==4:
            N,C,H,W=self.din_shape
        else:
            C,H,W=self.din_shape
        oh=int(1+(H-self.fh)/self.stride)
        ow=int(1+(W-self.fw)/self.stride)

        imcol=im2col(din,self.fh,self.fw,self.stride,self.pad)
        #N*oh*ow*C, fh*fw로 모양 변화
        imcol=imcol.reshape(-1,self.fh*self.fw)

        self.mask=np.argmax(imcol,axis=1)
        dout=din[self.mask]
        return dout.reshape(N,oh,ow,C).transpose(0,3,1,2)
    def backward(self,dout):
        N,C,oh,ow=dout.shape
        dtmp=dout.transpose(0,2,3,1).reshape(-1)
        dy=np.zeros(N*oh*ow*C,self.fw*self.fh)
        dy[self.mask]=dtmp
        dy=dy.reshape(N*oh*ow,C*self.fh*self.fw)
        img=col2im(self.original_shape,dy,self.fh,self.fw,self.stride,self.pad)
        return img


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

class softmax(object):
    def __init__(self):
        self.x=None
        self.y=None
        self.dy=None
        self.error=None

    def softmax(self,x):
        max_x=x.max(1)
        exp_x=np.exp(x-max_x.reshape(max_x.shape[0],1))
        sum_of_row=np.sum(exp_x,1)
        for i in range(sum_of_row.shape[0]):
            exp_x[i,:]=exp_x[i,:]/sum_of_row[i]
        return exp_x

    def crossEntropyError(self,x,label):
        delta=1e-7
        ln_x=np.log(x+delta)
        error=-np.sum(np.multiply(label,ln_x),1)
        return error

    def forward(self, x,label):
        self.x=x.copy()
        self.y=self.softmax(self.x)
        self.error=self.crossEntropyError(self.y,label)
        return self.y

    def backward(self, label):
        self.dy=self.y-label
        return self.dy
