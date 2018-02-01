#!/home/sabi/anaconda3/envs/DL_py/bin/python
import numpy as np
import sys
import os
import pdb
import layer

sys.path.append("/home/sabi/workspace/reference_code/deep-learning-from-scratch/")
sys.path.append(os.getcwd())
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt
if __name__=="__main__":
    (x_train, t_train),(x_test, t_test)=load_mnist(normalize=True, one_hot_label=True)
    y={}
    dy={}
    bat_size=100
    network=[]
    network.append(layer.convolution(30,1,5,5))
    network.append(layer.Relu())
    network.append(layer.Maxpool(2,2,2,0))
    network.append(layer.affine(50*13*13,100))
    network.append(layer.Relu())
    network.append(layer.affine(100,10))
    output_layer=layer.softmax()
    loss=[]
    acculacy=[]
    per_epoch=int(max(x_train.shape[0]/bat_size,1))
    train_image=x_train.reshape(x_train.shape[0],1,28,28)
    for i in range(10000):
        idx_list=np.random.choice(x_train.shape[0],bat_size)
    
    #predict process
    dout=train_image[idx_list,:]
    for hidden_layer in network:
        pdb.set_trace()
        dout=hidden_layer.forward(dout)
    
    output_layer.forward(dout,t_train[idx_list,:])
        
'''
    #back propagation
    dout=output_layer.backward(t_train[idx_list,:])
    for hidden_layer in network.reverse():
        dout=hidden_layer.backward(dout)
    #update parameter
    affine_L5.update()
    affine_L3.update()
    affine_L1.update()


    dout=x_train
    for hidden_layer in network:
        dout=hidden_layer(dout)
    output_layer.forward(dout,t_train[idx_list,:])

    if i % per_epoch == 0:
#            print("predict_t", softmax_L6.y[0,:])
#            print("t", t_train[idx_list[0],:])
            #accumulate loss
        loss.append(np.sum(softmax_L6.error))
            #if len(loss)>=2 and loss[-1]<=loss[-2]:
            #    affine_L5.update_learningrate(0.1)
            #    affine_L3.update_learningrate(0.1)
            #    affine_L1.update_learningrate(0.1)
            #compute acculacy
        predict_t=dout.argmax(axis=1)
        if t_train.ndim != 1: t=t_train[idx_list,:].argmax(axis=1)
        else: t=t_train[idx_list,:].copy()
#            pdb.set_trace()
        acculacy.append(t[np.where(predict_t==t)].shape[0]/bat_size)
        print("loss: %f, accuracy: %f"%(loss[-1],acculacy[-1])) 
    x=np.arange(0,len(acculacy))
    plt.figure(1)
    plt.plot(x,acculacy)
    plt.title('accuracy')
    plt.figure(2)
    plt.plot(x,loss)
    plt.title('loss')
    plt.show()
    pdb.set_trace()
    dout=affine_L1.forward(x_test)
    dout=sigmoid_L2.forward(dout)
    dout=affine_L3.forward(dout)
    dout=sigmoid_L4.forward(dout)
    dout=affine_L5.forward(dout)
    dout=softmax_L6.forward(dout, t_test)

    predict_t=softmax_L6.y.argmax(axis=1)
    if t_test.ndim !=1: t=t_test.argmax(1)
    else: t=t_test.copy()

    print("accuracy: %f" %(t[np.where(predict_t==t)].shape[0]/t.shape[0]))
    
    '''
