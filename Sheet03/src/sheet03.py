import numpy as np
import cv2 as cv
import random
import matplotlib.pyplot as mp
import math


##############################################
#     Task 1        ##########################
##############################################
kernel_bandwidth = 4  # Kernel parameter.
def EuclideanDistance(X,Y):
    return (( (X[:,0]-Y[:,0])**2 + (X[:,1]-Y[:,1])**2)**0.5 )


def euclid_distance(x, xi):
    return np.sqrt(np.sum((x - xi)**2))

def maximum(a,b):
    if a>b:
       return a
    else:
       return b
        

def neighbourhood_points(X, x_centroid, distance = 5):
    eligible_X = []
    for x in X:
        distance_between = euclid_distance(x, x_centroid)
        if distance_between <= distance:
            eligible_X.append(x)
    return eligible_X

def gaussian_kernel(distance, bandwidth):
    val = (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((distance / bandwidth))**2)
    return val


def task_1_a():
    print("Task 1 (a) ...")
    img = cv.imread('../images/shapes.png')
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,150,apertureSize = 3)
    #cv.imwrite('edges.jpg',edges)
    

    lines = cv.HoughLines(edges,1,np.pi/180,15,30,10)
    print(lines[0])
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    cv.imwrite('houghlines_opencv.jpg',img)
 


def myHoughLines(img_edges, d_resolution, theta_step_sz, threshold):
    """
    Your implementation of HoughLines
    :param img_edges: single-channel binary source image (e.g: edges)
    :param d_resolution: the resolution for the distance parameter
    :param theta_step_sz: the resolution for the angle parameter
    :param threshold: minimum number of votes to consider a detection
    :return: list of detected lines as (d, theta) pairs and the accumulator
    """
    accumulator = np.zeros((int(180 / theta_step_sz), int(np.linalg.norm(img_edges.shape) / d_resolution)))
    detected_lines = []
 
    thetas = np.deg2rad(np.arange(0, 180, theta_step_sz))

    y_idxs, x_idxs = np.nonzero(img_edges) # find all edge (nonzero) pixel indexes
    
    xp=[]
    yp=[]
    points=[]
    for i in range(len(x_idxs)): # cycle through edge points
        x = x_idxs[i]
        y = y_idxs[i]

        for j in range(len(thetas)): # cycle through thetas and calc rho
            d = int( x * np.cos(thetas[j]) +  y * np.sin(thetas[j]) )
            accumulator[j][d] += 1
            points.append((d,j))
            xp.append(d)
            yp.append(j)

    mp.scatter(xp,yp,alpha=0.3,c='r',s=0.15)
    mp.title("Vizualization for Accumulator from Hough Transform ")
    mp.xlabel('d')
    mp.ylabel('theta')
    mp.show()
    #Voting on the accumulator 
    t,d = np.nonzero(accumulator > threshold)
    #print(t,d)
    for i in range(len(t)):
        detected_lines.append((d[i],t[i]))
 
    #print (detected_lines)
    return detected_lines, accumulator,points


def task_1_b():
    print("Task 1 (b) ...")
    img = cv.imread('../images/shapes.png')
    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(img_gray,50,150,apertureSize = 3)
    detected_lines, accumulator, points = myHoughLines(edges, 1, 2, 150)
    print(detected_lines)
    for rho,theta in detected_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    cv.imwrite('houghlines_customized.jpg',img)


##############################################
#     Task 2        ##########################
##############################################


def task_2():
    print("Task 2 ...")
    img = cv.imread('../images/line.png')
    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) # convert the image into grayscale
    edges = cv.Canny(img_gray,50,150,apertureSize = 3) # detect the edges
    theta_res = 2 # set the resolution of theta
    d_res = 1 # set the distance resolution
    _, accumulator,original_X = myHoughLines(edges, d_res, theta_res, 50)
    
    
    X = np.copy(original_X)
    print(X)
    past_X = []
    n_iterations = 5
    figure = mp.figure(1)
    figure.set_size_inches((10, 16))
    mp.subplot(n_iterations + 2, 1, 1)
    mp.title('Initial state')
    #mp.plot(original_X[:,0], original_X[:,1], 'bo')
    #mp.plot(original_X[:,0], original_X[:,1], 'ro')
    
    xp=[]
    yp=[]
    for it in range(n_iterations):
        for i, x in enumerate(X):
            ### Step 1. For each datapoint x ∈ X, find the neighbouring points N(x) of x.
            neighbours = neighbourhood_points(X, x)
        
            ### Step 2. For each datapoint x ∈ X, calculate the mean shift m(x).
            numerator = 0
            denominator = 0
            for neighbour in neighbours:
                distance = euclid_distance(neighbour, x) #maximum(accumulator[neighbour[0]][neighbour[1]], accumulator[x[0]][x[1]] )
                weight = gaussian_kernel(distance, kernel_bandwidth)
                numerator += (weight * neighbour)
                denominator += weight
        
            new_x = numerator / denominator
        
             ### Step 3. For each datapoint x ∈ X, update x ← m(x).
            #accumulator[ X[i][0] ][ X[i][1] ] = new_x
            X[i] = new_x
        past_X.append(np.copy(X))   
        
        #figure_index = i + 2
        #mp.subplot(n_iterations + 2, 1, figure_index)
        #mp.title('Iteration: %d' % (figure_index - 1))
        #mp.plot(original_X[:,0], original_X[:,1], 'bo')
        #mp.plot(past_X[i][:,0], past_X[i][:,1], 'ro')
        xp.append(X[i][0])
        yp.append(X[i][1])
        print (xp,yp)

    mp.scatter(xp,yp,alpha=0.3,c='r',s=0.15)
    mp.title("Vizualization for Accumulator from Meanshift ")
    mp.xlabel('d')
    mp.ylabel('theta')
    mp.show()
    #print(accumulator)
 
    #past_X.append(np.copy(X))
    #accumulatornew = np.zeros((int(180 / theta_step_sz), int(np.linalg.norm(img_edges.shape) / d_resolution)))
    
    
   

##############################################
#     Task 3        ##########################
##############################################


def myKmeans(data, k):
    """
    Your implementation of k-means algorithm
    :param data: list of data points to cluster
    :param k: number of clusters
    :return: centers and list of indices that store the cluster index for each data point
    """
    t = 0
    done = False
    tmax = 1000
    epsilon = 0.0001
    #meu_t is the set of centroids of meu1,meu2,meu3,....meuk
    start = tm.clock()
    meu_t = np.reshape(data[np.random.choice( range(0,data.shape[0]),k,replace=False),:],(k,2))
    Ct = np.hstack((data,np.reshape(np.argsort(np.linalg.norm(data[:, None, :] - meu_t[None, :, :], axis=2), axis=1)[:, 0],(data.shape[0], 1))))
    while not done:
          #Ct is the previous Cluster set
          # meu_tplus1 is the set of updated centroids
          meu_tplus1 = (npi.group_by(Ct[:, 2]).mean(Ct))[1][:,0:2]
          # Ct is the updated cluster set
          Ctnew = np.hstack((data, np.reshape(np.argsort(np.linalg.norm(data[:, None, :] - meu_tplus1[None, :, :], axis=2), axis=1)[:, 0],(data.shape[0], 1))))
          #Tests for convergence
          #Test 1. Check if the previous and current cluster sets intersect
          Ct = np.array(sorted(Ct,key=itemgetter(2)))
          Ctnew = np.array(sorted(Ctnew,key=itemgetter(2)))
          if np.array_equal(Ct,Ctnew) == True:
             done = True
          #Test 2. Check if distance between previous and current centroid(s) is less than epsilon
          if np.reshape(EuclideanDistance(meu_t,meu_tplus1),(k,1)).all() <= epsilon:
             done = True
          #Test 3. Number of iterations exceed maximum limit
          t = t+1
          if t > tmax:
             done = True
          Ct = Ctnew
          meu_t = meu_tplus1
          end =  tm.clock()
    return Ct,np.hstack((meu_t,np.reshape( np.array(list(range(k))),(k,1)))),t,end-start
    #return index, centers


def task_3_a():
    print("Task 3 (a) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''


def task_3_b():
    print("Task 3 (b) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''


def task_3_c():
    print("Task 3 (c) ...")
    img = cv.imread('../images/flower.png')
    '''
    ...
    your code ...
    ...
    '''


##############################################
#     Task 4        ##########################
##############################################


def task_4_a():
    print("Task 4 (a) ...")
    D = None  # construct the D matrix
    W = None  # construct the W matrix
    '''
    ...
    your code ...
    ...
    '''


##############################################
##############################################
##############################################


task_1_a()
task_1_b()
#task_2()
#task_3_a()
#task_3_b()
#task_3_c()
#task_4_a()

