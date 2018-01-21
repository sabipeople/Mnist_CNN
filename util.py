import numpy as np
import pdb

def im2col(din,fh,fw,stride=1, pad=0):
	if din.ndim == 2:
		pdb.set_trace()
	elif din.ndim ==3:
		pdb.set_trace()
	elif din.ndim > 3:
		N,C,H,W=din.shape
		oh=(H+2*pad-fh)//stride + 1
		ow=(W+2*pad-fw)//stride + 1

		pad_img=np.pad(din,[(0,0),(0,0),(pad,pad),(pad,pad)],'constant')
		col=np.zeros(shape=(N,C,fh,fw,oh,ow))
		
		for y in range(fh):
			ymax=y+stride*oh
			for x in range(fw):
				xmax=x+stride*ow
				col[:,:,y,x,:,:]=pad_img[:,:,y:ymax:stride, x:xmax:stride]

		return col.transpose(0,4,5,1,2,3).reshape(N*oh*ow,-1)

def col2im(din_shape,imcol,fh,fw, stride=1, pad=0):
		
	N,C,H,W = din_shape
	oh=(H+2*pad-fh)//stride +1
	ow=(W+2*pad-fw)//stride +1
	img=np.zeros(shape(N,C,H+2*pad,W+2*pad))
	col = imcol.reshape(N,oh,ow,C,H+2*pad,W+2*pad).tranpose(0,3,4,5,1,2)
	for y in range(fh):
		ymax=y+stride*oh
		for x in range(fw):
			xmax=x+stride*ow
			img[:,:,y:ymax:stride,x:xmax:stride]+=col[:,:,y,x,:,:]
	
		return img[:,:,pad:H,pad:W]




