import os, cv2
from skimage import data, util
from skimage.measure import label, regionprops, moments_hu
from matplotlib import pyplot as plt
import numpy as np


class BarrelDetector():
    def __init__(self):
        self.weights=np.array([7.17373143,-11.80197334,0.62492144,-4.35011283])

    def segment_image(self, img):
        plt.imshow(img)
        img=img.reshape(800*1200,3)
        img=img/255
        x=np.column_stack((img,np.ones(1200*800)))
        s=np.dot(x,self.weights)
        predict=[]
        for i in range(800*1200):
            if (s[i])>=-4:
                predict.append(1)
            else:
                predict.append(0)
        predict=np.array(predict)
        mask_img=predict.reshape(800,1200)
        return mask_img
    
    def get_bounding_box(self, img):
        img=BarrelDetector().segment_image(img)
        label_img = label(img)
        props = regionprops(label_img)
        boxes=[]
        for p in props:
            minr,minc,maxr,maxc=p.bbox
            r=(maxr-minr)/(maxc-minc)
            a=p['area']
            if r>1.5 and a>500:   
                b=[minc,minr,maxc,maxr]
                boxes.append(b)
        return boxes
        
    
if __name__ == '__main__':
    folder = "trainset"
    my_detector = BarrelDetector()
    img=cv2.imread('trainset/41.png')
    mask_img=my_detector.segment_image(img)
    boxes=my_detector.get_bounding_box(img)
    print(boxes)
    
    
    


