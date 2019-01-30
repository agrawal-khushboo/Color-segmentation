import os, cv2
from skimage.measure import label, regionprops
import numpy as np


class BarrelDetector():
    def __init__(self):
        self.weights=np.array([7.17373143,-11.80197334,0.62492144,-4.35011283])

    def segment_image(self, img):
        img=img.reshape(800*1200,3)
        img=img/255
        x=np.column_stack((img,np.ones(1200*800)))
        s=np.dot(x,self.weights)
        predict=[]
        for i in range(800*1200):
            if (s[i])>=-4.5:
                predict.append(1)
            else:
                predict.append(0)
        predict=np.array(predict)
        mask_img=predict.reshape(800,1200)
        return mask_img
    
    
if __name__ == '__main__':
    folder = "trainset"
    my_detector = BarrelDetector()
    
    
    


