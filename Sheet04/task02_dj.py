import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import math
#rc('text', usetex=True)  # if you do not have latex installed simply uncomment this line + line 75


def load_data():
    """ loads the data for this task
    :return:
    """
    fpath = '../images/ball.png'
    radius = 70
    Im = cv2.imread(fpath, 0).astype(np.float64)/255  # 0 .. 1
    # we resize the image to speed-up the level set method
    Im = cv2.resize(Im, dsize=(0, 0), fx=0.5, fy=0.5)

    height, width = Im.shape

    centre = (width // 2, height // 2)
    Y, X = np.ogrid[:height, :width]
    phi = radius - np.sqrt((X - centre[0]) ** 2 + (Y - centre[1]) ** 2)

    return Im, phi


def get_contour(phi):
    """ get all points on the contour
    :param phi:
    :return: [(x, y), (x, y), ....]  points on contour
    """
    eps = 1
    A = (phi > -eps) * 1
    B = (phi < eps) * 1
    D = (A - B).astype(np.int32)
    D = (D == 0) * 1
    Y, X = np.nonzero(D)
    return np.array([X, Y]).transpose()

# ===========================================
# RUNNING
# ===========================================

# FUNCTIONS
# ------------------------
# your implementation here

def frontProp(gradientwX,gradientwY,phi):
    result=np.zeros_like(gradientwX)
    for r in range(0,result.shape[0]):
        for c in range(0,result.shape[1]):
            weightX=gradientwX[r,c]
            weightY=gradientwY[r,c]
            p=phi[r,c]
            result[r,c] = max(weightX,0.0) *(phi[r,min(c+1,result.shape[1]-1)]-p)+max(weightY,0.0) *(phi[min(r+1,result.shape[0]-1),c]-p) 
            + min(weightX,0.0) * (p-phi[r,max(c-1,0)]) + min(weightY,0.0) *(p-phi[max(r-1,0),c])

    return result


def levelSetContours(Im,phi,temp):
    #make kernels
    image=cv2.imread('../images/ball.png', 0)
    dx=np.array([-0.5,0.0,0.5])
    dxx=np.array([1.0,-2.0,1.0])
    gray=Im.copy()
    gradientX=cv2.sepFilter2D(gray,-1,dx,1.0)
    gradientY=cv2.sepFilter2D(gray,-1,1.0,dx)

    weight = gradientX*gradientX + gradientY * gradientY
    np.sqrt(weight)
    weight = 1.0 /( weight + 1.0).astype(np.float64)
    minVal,maxVal,_,_=cv2.minMaxLoc(weight)
    weight=(weight - minVal) * (maxVal/(maxVal - minVal))
    gradientwX=cv2.sepFilter2D(weight,-1,dx,1.0)
    gradientwY=cv2.sepFilter2D(weight,-1,1.0,dx)

    tau= 1.0/(4.0 * maxVal)
    eps = 0.0001

    phi_x = cv2.sepFilter2D(phi,-1,dx,1.0)
    phi_y = cv2.sepFilter2D(phi,-1,1.0,dx)
    phi_xy = cv2.sepFilter2D(phi_x,-1,1.0,dx)
    phi_xx = cv2.sepFilter2D(phi,-1,dxx,1.0)
    phi_yy = cv2.sepFilter2D(phi,-1,1.0,dxx)
    phi_x2 = cv2.pow(phi_x,2)
    phi_y2=cv2.pow(phi_y,2)

    curvature = (phi_xx * phi_y2 - 2.0*(phi_x*(phi_y*phi_xy)) + (phi_yy*phi_x2))*(1.0/(phi_x2 + phi_y2 + eps))
    curvature=curvature.astype(np.float64)
    curvature=cv2.multiply(curvature,weight,tau)
    edges=frontProp(gradientwX,gradientwY,phi)
    phi = phi + curvature + edges

    t = 1
    plot_every_n_step = 100
    while (t < 20000):
        phi_x = cv2.sepFilter2D(phi,-1,dx,1.0)
        phi_y = cv2.sepFilter2D(phi,-1,1.0,dx)
        phi_xy = cv2.sepFilter2D(phi_x,-1,1.0,dx)
        phi_xx = cv2.sepFilter2D(phi,-1,dxx,1.0)
        phi_yy = cv2.sepFilter2D(phi,-1,1.0,dxx)
        phi_x2 = cv2.pow(phi_x,2)
        phi_y2=cv2.pow(phi_y,2)
        curvature = (phi_xx * phi_y2 - 2.0*(phi_x*(phi_y*phi_xy)) + (phi_yy*phi_x2))*(1.0//(phi_x2 + phi_y2 + eps))
        curvature=cv2.multiply(curvature,weight,tau)
        edges=frontProp(gradientwX,gradientwY,phi)
        phi = phi + curvature + edges
        if t % plot_every_n_step == 0:
            ax1.clear()
            ax1.imshow(Im, cmap='gray')
            ax1.set_title('frame ' + str(t))
            contour= get_contour(phi)
            if len(contour) > 0:
                ax1.scatter(contour[:, 0], contour[:, 1], color='red', s=1)
            ax2.clear()
            ax2.imshow(phi)
            ax2.set_title(r'$\phi$', fontsize=22)
            plt.pause(0.01)
        t=t+1

    plt.show()      

# ------------------------


if __name__ == '__main__':

    n_steps = 20000
    plot_every_n_step = 100

    Im, phi = load_data()
    phi=phi.astype(np.float32)
    temp=phi.copy()
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    levelSetContours(Im,phi,temp)

    # ------------------------
    # your implementation here

    # ------------------------

