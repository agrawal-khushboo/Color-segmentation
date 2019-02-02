import pickle
import time
label=[]
with open("label.txt","wb") as f:
    pickle.dump(label, f)
    
    
    from matplotlib import pyplot as plt
import matplotlib.image as mpimg
%matplotlib notebook
from roipoly import *
img=mpimg.imread('trainset/1.png')
plt.imshow(img)


# Show the image
fig = plt.figure()
plt.imshow(img, interpolation='nearest')
# plt.colorbar()
# plt.title("left click: line segment         right click: close region")

# Let user draw first ROI
roi1 = RoiPoly(color='r', fig=fig)

# Show the image with the first ROI
fig = plt.figure()
plt.imshow(img, interpolation='nearest')
plt.colorbar()
roi1.display_roi()


mask = roi1.get_mask(img[:,:,2])    
plt.imshow(mask)
plt.title('ROI masks of the two ROIs')
plt.show()

mask=mask.flatten()
y=[]
for i in mask:
    if i ==False:
        y.append(-1)with open("label.txt","rb") as f:
    label = pickle.load(f)
    print(label)
label.append(np.array(y))
with open("label.txt","wb") as f:
    pickle.dump(label,f)


    with open("label.txt","rb") as f:
    label = pickle.load(f)
    print(len(label))
    else:
        y.append(1)

