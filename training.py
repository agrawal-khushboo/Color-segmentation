
import pickle
with open("label.txt","rb") as f:
    label = pickle.load(f)
    print(len(label))
    
    import numpy as np
import cv2
import os
from math import exp

def sigmoid(a):
    return 1/(1+np.exp(-a))
  
#Finding the weights 
  folder="trainset"
weight=np.zeros((1,4),dtype=np.float)
# weight=np.random.rand(1,4)
weight=np.matrix(weight)
weight=weight.T
for epoch in range(50):
    image=0
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,file))
        if image<47:
            x=[]
            for i in range(3):
                col=img[:,:,i]
                col=col.flatten()
                col=col*(2/255)-1
                x.append(col)
            x.append(np.ones(800*1200))
            x=np.matrix(x)
            x=x.T
            y=label[image]
            image+=1
            y=np.matrix(y)
            y=y.T
            np.shape(y)
            score=np.matmul(x,weight)
            newscore=np.multiply(y,score)
            newscore=sigmoid(newscore)
            newscore=1-newscore
            newscore=np.multiply(x,newscore)
            newscore=np.multiply(y,newscore)
            newscore=newscore.sum(axis=0)
            newscore=newscore.T
            weight=weight+0.0001*newscore

    
# Predicting the label

img=cv2.imread('trainset/44.png')
x=[]
for i in range(3):
    col=img[:,:,i]
    col=col.flatten()
    col=col/255         
    x.append(col)
x.append(np.ones(800*1200))
x=np.matrix(x)
x=x.T
score=np.matmul(x,weight)
# score=sigmoid(score)
predict=[]
for i in range(score.shape[0]):
    if (np.float(score[i]))>=0:
        predict.append(1)
    else:
        predict.append(0)




