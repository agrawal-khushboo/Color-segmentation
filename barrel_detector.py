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
        self.weights=(np.matrix(self.weights)).T
#         weights=weights.T
        

    def sigmoid(score):
        return 1/(1+np.exp(score))

    def segment_image(self, img):
        x=[]
        for i in range(3):
            col=img[:,:,i]
            col=col.flatten()
            col=col/255         
            x.append(col)
        x.append(np.ones(800*1200))
        x=np.matrix(x)
        x=x.T
        
        score=np.matmul(x,self.weights)
        predict=[]
        for i in range(score.shape[0]):
            if (np.float(score[i]))>=-4.5:
                predict.append(1)
            else:
                predict.append(-1)
        mask_img=predict
#         raise NotImplementedError
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



		#Display results:
		#(1) Segmented images
		#	 mask_img = my_detector.segment_image(img)
		#(2) Barrel bounding box
		#    boxes = my_detector.get_bounding_box(img)
		#The autograder checks your answers to the functions segment_image() and get_bounding_box()
		#Make sure your code runs as expected on the testset before submitting to Gradescope



