import os, cv2
from skimage import data, util
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
    
    def get_bounding_box(self, img):
        img=BarrelDetector().segment_image(img)
        img = util.img_as_ubyte(data.coins()) > 110
        label_img = label(img, connectivity=img.ndim)
        props = regionprops(label_img)
        r=1.0
        x1=0.0
        y1=0.0
        x2=0.0
        y2=0.0
        for p in props:
            minx, miny, maxx, maxy = p.bbox
            newr=(maxy-miny)/(maxx-minx)
            if newr>r:
                r=newr
                x1=minx
                y1=miny
                x2=maxx
                y2=maxy
                b=[minx, maxy, maxx, miny]
#                 boxes.append(b)
#         boxes=np.array(boxes)
        return b
            
                
                
        
    
if __name__ == '__main__':
    folder = "trainset"
    my_detector = BarrelDetector()
    img=cv2.imread('trainset/44.png')
    boxes=my_detector.get_bounding_box(img)
    print(boxes)
    
    
    


