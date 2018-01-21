import numpy as np
import pdb
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
