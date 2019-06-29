import numpy as np
import os
import cv2 as cv


MAX_ITERATIONS = 3000 # maximum number of iterations allowed until convergence of the Horn-Schuck algorithm
EPSILON = 0.002 # the stopping criterion for the difference when performing the Horn-Schuck algorithm
EIGEN_THRESHOLD = 0.01 # use as threshold for determining if the optical flow is valid when performing Lucas-Kanade

def load_FLO_file(filename):
    
    if os.path.isfile(filename) is False:
        print("file does not exist %r" % str(filename))
    flo_file = open(filename,'rb')
    magic = np.fromfile(flo_file, np.float32, count=1)
    if magic != 202021.25:
        print('Magic number incorrect. .flo file is invalid')
    w = np.fromfile(flo_file, np.int32, count=1)
    h = np.fromfile(flo_file, np.int32, count=1)
    #the float values for u and v are interleaved in row order, i.e., u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...,
    # in total, there are 2*w*h flow values
    data = np.fromfile(flo_file, np.float32, count=2*w[0]*h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    flo_file.close()
    return flow

#***********************************************************************************
#implement Lucas-Kanade Optical Flow 
# Parameters:
# frames: the two consecutive frames
# Ix: Image gradient in the x direction
# Iy: Image gradient in the y direction
# It: Image gradient with respect to time
# window_size: the number of points taken in the neighborhood of each pixel
# returns the Optical flow based on the Lucas-Kanade algorithm
def Lucas_Kanade_flow(ix, iy, it, window_size):
    # Compute the differential operators
    ix2 = ix * ix
    iy2 = iy * iy
    ixy = ix * iy
   
    ixt = ix * it 
    iyt = iy * it
    
    #print( ix2.shape, iy2.shape, ixy.shape, it.shape, ixt.shape, iyt.shape)
    # Sum them over a window size using box filter
    sigma_ixix = cv.boxFilter(ix2,-1,window_size,None,(-1,-1),False)
    sigma_iyiy = cv.boxFilter(iy2,-1,window_size,None,(-1,-1),False)
    sigma_ixiy = cv.boxFilter(ixy,-1,window_size,None,(-1,-1),False)
    sigma_ixt = cv.boxFilter(ixt,-1,window_size,None,(-1,-1),False)
    sigma_iyt = cv.boxFilter(iyt,-1,window_size,None,(-1,-1),False)
    
    #print( sigma_ixix.shape, sigma_iyiy.shape, sigma_ixiy.shape, sigma_ixt.shape, sigma_iyt.shape)
    
    #Finding the eigenvalues of the second moment matrix, A, where our equation will be of the form Ax=B, using quadratic solution of SriDhar Acharaya formula
    c = sigma_ixix + sigma_iyiy
    S = sigma_ixix**2 + sigma_iyiy**2 -(2*sigma_ixix*sigma_iyiy) +(4*sigma_ixiy*sigma_ixiy)
    e1 = ( c + np.sqrt(S) )/2
    e2 = ( c - np.sqrt(S) )/2
       
    #Determinant value of coefficiant matrix
    D = (sigma_ixix*sigma_iyiy) - (sigma_ixiy**2)
    #print(D.shape)
    # Solution of the matrix equation Ax=B format, x = inv(A) . B 
    # D = Determinant(A) 
    u = -( sigma_iyiy*sigma_ixt - sigma_ixiy*sigma_iyt ) / D
    v = -( sigma_ixix*sigma_iyt - sigma_ixiy*sigma_ixt ) / D
    
    flow =  np.stack((u,v),axis=2)
    eigen = np.stack((e1,e2),axis=2)
    #print(flow.shape)
    return flow,eigen
    pass


#***********************************************************************************
#implement Horn-Schunck Optical Flow 
# Parameters:
# frames: the two consecutive frames
# Ix: Image gradient in the x direction
# Iy: Image gradient in the y direction
# It: Image gradient with respect to time
# window_size: the number of points taken in the neighborhood of each pixel
# returns the Optical flow based on the Horn-Schunck algorithm

def Horn_Schunck_flow(frames, Ix, Iy, It):
    i = 0
    diff = 1
    u = np.zeros((frames.shape[0],frames.shape[1]),dtype = np.float64)
    v = np.zeros((frames.shape[0],frames.shape[1]),dtype = np.float64)
    D = ( 1.0 + Ix*Ix + Iy*Iy)
    while i<MAX_ITERATIONS and diff > EPSILON: #Iterate until the max number of iterations is reached or the difference is less than epsilon
        i += 1
        uk_ = u + 0.05*cv.Laplacian(u,cv.CV_64F)
        vk_ = v + 0.05*cv.Laplacian(v,cv.CV_64F)
        #print( uk_,vk_)
        coefficient = ( Ix*uk_ + Iy*vk_ + It )/D
        u_new = uk_ - Ix*coefficient
        v_new = vk_ - Iy*coefficient
        diff = (np.sum( np.abs(u_new-u) + np.abs(v_new-v) ))
        #print(diff)
        u = np.copy(u_new)
        v = np.copy(v_new)
    
    flow = np.stack((u,v),axis=2)
    #print(flow.shape)
    return flow
         
#calculate the angular error here
def calculate_angular_error(estimated_flow, groundtruth_flow):
    d1 = np.sum(groundtruth_flow*groundtruth_flow,axis=2) + 1.0
    d2 = np.sum(estimated_flow*estimated_flow,axis=2) + 1.0
    n = np.sum(estimated_flow*groundtruth_flow,axis=2) + 1.0
    c = np.arccos(n / ((d1*d2)**0.5))
    c = np.sum(c) / (c.shape[0] * c.shape[1])
    return c*(180/np.pi)
    pass

def flow_to_bgr(img,flow):
    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros(img.shape, dtype=np.uint8)
    hsv[..., 1] = 255
    #get the angle and magnitude
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    #convert thr angle, and normalize the magnitude
    hsv[..., 0] = angle * 90 / np.pi 
    hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    #convert the hsv to bgr space
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr

if __name__ == "__main__":
    # read your data here and then call the different algorithms, then visualise your results
    window_size = (15, 15)  #the number of points taken in the neighborhood of each pixel when applying Lucas-Kanade
    gt_flow = load_FLO_file('../data/groundTruthOF.flo')
    img1_gray = cv.imread('../data/frame1.png',0) # Grayscale image
    img1 = cv.imread('../data/frame1.png')   # Normal image
    img2_gray = cv.imread('../data/frame2.png',0) # Grayscale image
    img2 = cv.imread('../data/frame2.png')   # Normal image
    #print (gt_flow.shape)
 
    #Sobel filters for computing gradients of the images
    ix = cv.Sobel(img1_gray,cv.CV_64F,1,0,ksize=3)
    iy = cv.Sobel(img1_gray,cv.CV_64F,0,1,ksize=3)
    it = img2_gray - img1_gray
    ix = ix.astype(np.float64)
    iy = iy.astype(np.float64)
    it = it.astype(np.float64)
                   
    #Get the ground truth flow
    gt = flow_to_bgr(img1,gt_flow)
    
    #Lucas Kanade algorithm
    flow,eigen = Lucas_Kanade_flow(ix, iy, it, window_size)
    angular_error1 = calculate_angular_error(flow,gt_flow)
      
    vizual1 = flow_to_bgr(img1,flow)
    cv.imshow("Ground-Truth Flow ",gt)
    cv.imshow("Lucas-Kanade Flow ::", vizual1)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    
    #Horn Schunck algorithm
    flow = Horn_Schunck_flow(img1_gray,ix, iy, it)
    angular_error2 = calculate_angular_error(flow,gt_flow)
    
    vizual2 = flow_to_bgr(img1,flow)
    cv.imshow("Horn_Schunck_flow for Image-1", vizual2)
    cv.imshow("Ground-Truth Flow ",gt)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    valid = eigen > EIGEN_THRESHOLD
    c = np.sum(valid)
    print('No of valid eigen values ::',c)
    print('Total No of eigen values ::', valid.shape[0] * valid.shape[1] * valid.shape[2])
    print("Proves that all the second-moment matrics have valid solutions")
    
    print('Average Angular Error after Lucas Kanade = ',angular_error1)
    print('Average Angular Error after Horn Schunck = ',angular_error2)
    
    
    
    