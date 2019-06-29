import cv2
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
#   =======================================================
#                   Task1
#   =======================================================

# GEt the grayscale and the normal images
img1 = cv2.imread('../images/building.jpeg',0) # Grayscale image
img2 = cv2.imread('../images/building.jpeg')   # Normal image

# compute structural tensor
# Use sobel filter to find the gradients along x and y direction, using a kernal of size 3
ix = cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=3)
iy = cv2.Sobel(img1,cv2.CV_64F,0,1,ksize=3)
# Compute the differential operators
ix2 = ix * ix
iy2 = iy * iy
ixy = ix * iy
# Compute the summation, using a gaussian filter with a kernal size 3x3
ksize = (3,3)
ix2 = cv2.GaussianBlur(ix2,ksize,2)
iy2 = cv2.GaussianBlur(iy2,ksize,2)
ixy = cv2.GaussianBlur(ixy,ksize,2)

#Harris Corner Detection

#calculate the deteminant and the trace
D = (ix2*iy2)-(ixy*ixy)
T = (ix2+iy2)

#Compute the response function
R = D - 0.04*(T**2)

#Do non-maxima suppresion
rmax = np.max(R)
kernel = np.ones((3,3),np.uint8)
m = cv2.dilate(R,kernel) # This step finds the local maxima, using a 3x3 kernel
pc, pr = np.where(m >= (0.05*rmax)) # This step finds the thresholded values and their co-ordinates

# Plot the detected corner points and the image ... Maximize the result image for a better view
plt.scatter(pr, pc,c="r", s=0.45, marker='o')
plt.imshow(img2)
plt.title('Harris Corner Detection ( click exit to continue )')
plt.show()

#Forstner Corner Detection
w = D/T
q = (4*D)/(T**2)
w = np.nan_to_num(w,np.inf)
q = np.nan_to_num(q,np.inf)
wmin = np.max(w)
qmin = np.max(q)
#thresholding 
w = w >  0.1*wmin
q = q >  0.1*qmin
r = np.logical_and(w,q) 
# Display the corner points on the image
pc, pr = np.where(r==1)
plt.scatter(pr, pc,c="r", s=0.45, marker='o')
plt.imshow(img2)
plt.title('Forstner Corner Detection ( click exit to continue )')
plt.show()


#   =======================================================
#                   Task2
#   =======================================================

img1 = cv2.imread('../images/mountain1.png')
img2 = cv2.imread('../images/mountain2.png')
sift = cv2.xfeatures2d.SIFT_create()
#extract sift keypoints and descriptors
(kp1, des1) = sift.detectAndCompute(img1, None)
(kp2, des2) = sift.detectAndCompute(img2, None)
# show keypoints
out1 = cv2.drawKeypoints(img1, kp1, None)
out2 = cv2.drawKeypoints(img2, kp2, None)
cv2.imshow("Keypoints for Image-1 ( press enter to continue )", out1)
cv2.imshow("Keypoints for Image-2 ( press enter to continue )", out2)
#print(des1,des2)
cv2.waitKey(0)
cv2.destroyAllWindows()  

bf = cv2.BFMatcher()
matches12 = bf.knnMatch(des1,des2, k=2)
matches21 = bf.knnMatch(des2,des1, k=2)
#print(matches12.shape,matches21.shape)
# own implementation of matching
# filter matches by ratio test

t = 0.4
validratio12 = {}
for p in matches12:
    if (p[0].distance/p[1].distance) <= t:
        validratio12[p[0].queryIdx] = (p[0].trainIdx,p)
validratio21 = {}  
for p in matches21:
    if (p[0].distance/p[1].distance) <= t:
       validratio21[p[0].queryIdx] = (p[0].trainIdx,p)

#print(validratio12)
#print(validratio21)

# determine two-way matches
good_matches =[]
for k,v in validratio12.items():
    #print(k,v[0],v[1])
    #print(k, v, (validratio21.get(v[0]))[0] )
    key = validratio21.get(v[0])
    if key != None:
       if key[0] == k:
          good_matches.append( v[1])
       
#print(good_matches)       
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
cv2.imshow("Valid Matches for images based on SIFT keypoints ( press enter to continue )", img3)
cv2.waitKey(0)
cv2.destroyAllWindows() 
# display matched keypoints



#  =======================================================
#                          Task-3                         
#  =======================================================

nSamples = 4
nIterations = 50
thresh = 0.1
minSamples = 4
best_mse = sys.float_info.max
#  /// RANSAC loop
print(' Starting RANSAC algorithm....')
for i in range(nIterations):

    print('iteration '+str(i)) #,len(good_matches))
    
    #randomly select 4 pairs of keypoints
    kpts1 = []
    kpts2 = []
    for i in range(0,nSamples):
        index = random.randint(0,len(good_matches)-1)
        #print(index)
        kpts1.append( kp1[good_matches[index][0].queryIdx].pt )
        kpts2.append( kp2[good_matches[index][0].trainIdx].pt )
   
    kpts1 = np.array(kpts1,np.float32)
    kpts2 = np.array(kpts2,np.float32)
    #print(kpts1)    
    #print(kpts2)
    
    #apply homography and procure it
    hom = cv2.getPerspectiveTransform(kpts2, kpts1)
    #print(hom.shape)
    #print(img1.shape)
    warppedImage2 = cv2.warpPerspective(img2,  hom, (img1.shape[1],img1.shape[0]))
    #compute transofrmation and warp img2 using it
    #print( warppedImage2.shape)
    #cv2.imshow("Wrapped Image-2", warppedImage2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    no_inliers = 0
    total_mse = 0.0
    for j in range(0,len(kp1)):
        #print(kp1[j])
        size = kp1[j].size
        if size < 1:
           continue
        x = int(kp1[j].pt[1] - size / 2)
        y = int(kp1[j].pt[0] - size / 2)
        size=int(size)
        #print(len(kp1),j,x,y,x+size,y+size)
        patch1 = img1[y: y + size, x: x + size]
        patch2 = warppedImage2[y: y + size, x: x + size]
        #print(patch1.shape, patch2.shape)
        #if patch1.shape==patch2.shape: 
        diff = (cv2.absdiff(patch1, patch2))
        diff = np.float32(diff)
        diff = diff**2
        #print(diff.shape)
        mse = np.sum(diff) / (size*size*3*255*255)
        #print(mse)  
        if mse < thresh:
           no_inliers += 1
           total_mse += mse
    
    #print( total_mse,no_inliers)
    total_mse /= no_inliers
    if no_inliers > minSamples and total_mse < best_mse:
       best_mse = total_mse
       best_hom = hom.copy()
    #count inliers and keep transformation if it is better than the best so far
    #print ( best_mse)
print('RANSAC algorithm done....')

#apply best transformation to transform img2 
finalWarppedImage2 = cv2.warpPerspective(img2 , best_hom, (2*img1.shape[1], img1.shape[0]) )
cv2.imshow("Image-1", img1)
cv2.imshow("Wrapped Image-2", finalWarppedImage2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#display stitched images
#Merge images
mask = cv2.cvtColor(finalWarppedImage2, cv2.COLOR_BGR2GRAY)
ret,mask = cv2.threshold( mask, 0, 255,cv2.THRESH_BINARY_INV)
mask = mask[ 0:int(img1.shape[0]) , 0:int(img1.shape[1]) ]
#cv2.imshow("Mask", mask)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

mask = np.stack([mask,mask,mask],axis=2)
maskedImg = cv2.bitwise_and(img1, mask)
#cv2.imshow("Masked Image", maskedImg)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
finalWarppedImage2 = cv2.add( maskedImg, finalWarppedImage2[ 0:int(img1.shape[0]) , 0:int(img1.shape[1]) ])
cv2.imshow("Stitched Image", finalWarppedImage2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("Stitched_Image.jpg",finalWarppedImage2)