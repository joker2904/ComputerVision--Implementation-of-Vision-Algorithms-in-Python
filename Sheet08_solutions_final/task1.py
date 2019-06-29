import numpy as np
import utils
import cv2
import cv2.cv2 as cv
import matplotlib.pyplot as plt


def meanvizualize(x,y,title):
    plt.figure()
    plt.title(title)
    plt.plot(x,y)
    plt.show()


def visualize_hands2(kpts, title):
      
    fig = plt.figure(figsize=(5, 4))
    fig.canvas.set_window_title(title)
    ax = fig.add_subplot(111)
    ax.clear()
    ax.invert_yaxis()
    plt.axis('off')
      
    for sample_idx in range(1,kpts.shape[1]):
        ax.plot(kpts[0: int(kpts.shape[0]/2), sample_idx], kpts[int(kpts.shape[0]/2) : int(kpts.shape[0]), sample_idx])
        plt.pause(0.6)
    return ax


# ========================== Mean =============================
def calculate_mean_shape(kpts):
    # ToDO
    #print(kpts.shape)
    shapemu = cv2.reduce(kpts,1,cv2.REDUCE_AVG)
    #print(shapemu.shape)
    
    x = shapemu[0:int(shapemu.shape[0]/2),:]
    y = shapemu[int(shapemu.shape[0]/2):int(shapemu.shape[0]),:]
    #displayShape(x,y,"Shape Vizualization")
    #print(x.shape,y.shape)
    
    x = x * ( 1 / np.sqrt(np.var(x)) )
    y = y * ( 1 / np.sqrt(np.var(y)) )
    #print(x.shape,y.shape)
    return x,y
    pass



# ====================== Main Step ===========================
def procrustres_analysis_step(kpts, reference_meanx,reference_meany):
    # Happy Coding
    
    x = kpts[0:int(kpts.shape[0]/2),:]
    y = kpts[int(kpts.shape[0]/2):int(kpts.shape[0]),:]
    
   
    
    #print(x.shape,y.shape)
    #print(reference_meanx.shape,reference_meany.shape)
    refvarx = np.var(reference_meanx)
    refvary = np.var(reference_meany)
    
    
    varx = np.var(x,axis=0)
    vary = np.var(y,axis=0)
    varx = np.reshape( varx, (1,varx.shape[0]))
    vary = np.reshape( vary, (1,vary.shape[0]))
    #print(varx.shape,vary.shape)
    
    scalex = np.sqrt(refvarx / varx)
    scaley = np.sqrt(refvary / vary)
    
    #print(scalex.shape,scaley.shape)
    x = x * scalex
    y = y * scaley
    #print(x.shape,y.shape)
    
    svdx = np.dot(x.transpose(), reference_meanx)
    svdy = np.dot(y.transpose(), reference_meany)
    
    #print(svdx.shape,svdy.shape)
    ux,sx,vhx = np.linalg.svd(svdx)
    uy,sy,vhy = np.linalg.svd(svdy)
    
    #print(ux.shape,sx.shape,vhx.shape)
    #print(uy.shape,sy.shape,vhy.shape)
    
    rotx =  np.dot(vhx*ux.transpose(),x.transpose()).transpose()
    roty =  np.dot(vhy*uy.transpose(),y.transpose()).transpose()
    
    rotx = reference_meanx - rotx
    roty = reference_meany - roty
    nd = np.vstack((rotx,roty))
    
    #print(rotx.shape,roty.shape)
    #print(nd.shape)
    
    
    return nd
    
    pass



# =========================== Error ====================================

def compute_avg_error(kpts, reference_meanx,reference_meany):
    # ToDo
    st = np.vstack((reference_meanx,reference_meany))
    e = (kpts - st)**2
    s = np.sqrt( np.sum(e)) / (e.shape[0]*e.shape[1]) 
    return s
    pass




# ============================ Procrustres ===============================

def procrustres_analysis(kpts, max_iter=int(1e3), min_error=1e-5):

    aligned_kpts = kpts.copy()
    visualize_hands2(aligned_kpts,'original')
    reference_meanx,reference_meany = calculate_mean_shape(aligned_kpts)
    meanvizualize(reference_meanx,reference_meany,'mean of original dataset')
    
    for iter in range(max_iter):

        reference_meanx,reference_meany = calculate_mean_shape(aligned_kpts)
        
        # align shapes to mean shape
        aligned_kpts = procrustres_analysis_step(aligned_kpts, reference_meanx,reference_meany)
        print(aligned_kpts)
        
        error = compute_avg_error(aligned_kpts, reference_meanx,reference_meany)
        print(iter,' iteration, has error ::',error)
        if error < min_error:
           break
        ##################### Your Part Here #####################

        ##########################################################
 

    # visualize
    # visualize mean shape
    visualize_hands2(aligned_kpts,'aligned dataset')
    meanvizualize(reference_meanx,reference_meany,'mean dataset')
    return aligned_kpts
