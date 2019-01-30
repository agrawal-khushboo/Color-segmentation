'''
ECE276A WI19 HW1
Blue Barrel Detector
'''

import os, cv2
from skimage.measure import label, regionprops
import numpy as np

# weights=[7.17373143,-11.80197334,0.62492144,-4.35011283]
# weights=(np.matrix(weights)).T

class BarrelDetector():
    def __init__(self):
        self.weights=[7.17373143,-11.80197334,0.62492144,-4.35011283]
#         self.weights=np.matrix(self.weights)
#         self.weights=(self.weights).T
        

    def sigmoid(self, score):
        return 1/(1+np.exp(-score))

    def segment_image(self, img):
        img=img.reshape(800*1200,3)
        img=img/255
        x=np.column_stack((img,np.ones(1200*800)))
        s=np.dot(x,self.weights)
        predict=[]
        for i in range(800*1200):
            if (np.float(s[i]))>=-4.5:
                predict.append(1)
            else:
                predict.append(-1)
        mask_img=predict
        return mask_img

#     def get_bounding_box(self, img):
# 		'''
# 			Find the bounding box of the blue barrel
# 			call other functions in this class if needed
			
# 			Inputs:
# 				img - original image
# 			Outputs:
# 				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
# 				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
# 				is from left to right in the image.
				
# 			Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
# 		'''
# 		# YOUR CODE HERE
# 		raise NotImplementedError
# # 		return boxes
if __name__ == '__main__':
    folder = "trainset"
    my_detector = BarrelDetector()
    img = cv2.imread('trainset/44.png')
    mask_img=my_detector.segment_image(img)
    
#         cv2.imshow('image', img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()


		#Display results:
		#(1) Segmented images
		#	 mask_img = my_detector.segment_image(img)
		#(2) Barrel bounding box
		#    boxes = my_detector.get_bounding_box(img)
		#The autograder checks your answers to the functions segment_image() and get_bounding_box()
		#Make sure your code runs as expected on the testset before submitting to Gradescope



