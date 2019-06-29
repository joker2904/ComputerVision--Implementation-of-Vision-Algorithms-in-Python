import numpy as np
import utils
import cv2
import matplotlib.pyplot as plt



def visualize_hands2(kpts, title):
      
    fig = plt.figure(figsize=(5, 4))
    fig.canvas.set_window_title(title)
    ax = fig.add_subplot(111)
    ax.clear()
    ax.invert_yaxis()
    plt.axis('off')
      
    for sample_idx in range(0,kpts.shape[1]):
        ax.plot(kpts[0: int(kpts.shape[0]/2), sample_idx], kpts[int(kpts.shape[0]/2) : int(kpts.shape[0]), sample_idx])
        plt.pause(0.6)
    return ax


# ======================= PCA =======================
def pca(covariance, preservation_ratio=0.9):
    # Happy Coding! :)
    eigenValues, eigenVectors = np.linalg.eigh(covariance)
    print (eigenValues.shape, eigenVectors.shape)
    a = np.cumsum(eigenValues,axis=0) / np.sum(eigenValues,axis=0)
    dimension = np.count_nonzero(a < preservation_ratio)
    pcomp = ( eigenVectors[:, np.argsort(-eigenValues)[0:dimension] ] )
    pval  = ( eigenValues[ np.argsort(-eigenValues)[0:dimension] ] )
    print(pcomp.shape,pval.shape)
    return pcomp,pval
    pass


# ======================= Covariance =======================

def create_covariance_matrix(kpts, mean_shape):
    # ToDO
    c = kpts - mean_shape
    print(c.shape)
    cov = np.dot( c, c.transpose() ) / c.shape[0]
    print(cov.shape) 
    return cov
    pass


# ======================= Visualization =======================

def visualize_impact_of_pcs(mean, pcs, pc_weights,dev ):
    # your part here
    temp = mean.copy()
    print(pc_weights)
    c = np.reshape(dev*np.sqrt(np.abs(pc_weights)), (pc_weights.shape[0],1))
    #print(pcs.shape,c.shape)
    temp += np.dot(pcs, c)
    return temp
    #print('Vizualize-',temp)
    #visualize_hands2(temp,'PCA vizualize, with weight deviation'+str(dev))
    pass



# ======================= Training =======================
def train_statistical_shape_model(kpts):
    # Your code here
    mean_shape = cv2.reduce(kpts,1,cv2.REDUCE_AVG)
    print(mean_shape.shape)
    cov = create_covariance_matrix(kpts, mean_shape)
    pcomp,pval = pca(cov)
    for dev in np.arange(-2,2,0.1):
        t = visualize_impact_of_pcs(mean_shape,pcomp,pval,dev)
        if dev == -2:
           temp = t
        else:
           temp = np.hstack((temp,t))
    visualize_hands2(temp,'PCA vizualize, with weight deviations')
    return mean_shape,pcomp,pval    
    pass



# ======================= Reconstruct =======================
def reconstruct_test_shape(kpts, mean, pcs, pc_weight):
    #ToDo
    print('Reconstruct :')
    #decomposition using the principle components
    d = np.dot(pcs.transpose(),kpts-mean)
    #print(d.shape)
    #reconstruction phase
    reconTest = mean.copy()
    reconTest += np.dot(pcs,d)
    #print(reconTest.shape)
    visualize_hands2(kpts,'Original test shape')
    visualize_hands2(reconTest,'Reconstructed test shape from PCA ')
    rms = np.sqrt( np.sum((kpts-reconTest)**2)/kpts.shape[0])
    print('RMS between original and reconstructed values = ',rms)
    return reconTest
    pass
