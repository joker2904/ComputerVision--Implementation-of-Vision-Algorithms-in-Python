#!/usr/bin/python3.5

import numpy as np
from scipy import misc
import math
import sys


'''
    read the usps digit data
    returns a python dict with entries for each digit (0, ..., 9)
    dict[digit] contains a list of 256-dimensional feature vectores (i.e. the gray scale values of the 16x16 digit image)
'''
def read_usps(filename):
    data = dict()
    with open(filename, 'r') as f:
        N = int( np.fromfile(f, dtype = np.uint32, count = 1, sep = ' ') )
        for n in range(N):
            c = int( np.fromfile(f, dtype = np.uint32, count = 1, sep = ' ') )
            tmp = np.fromfile(f, dtype = np.float64, count = 256, sep = ' ') / 1000.0
            data[c] = data.get(c, []) + [tmp]
    for c in range(len(data)):
        data[c] = np.stack(data[c])
    return data

'''
    load the face image and foreground/background parts
    image: the original image
    foreground/background: numpy arrays of size (n_pixels, 3) (3 for RGB values), i.e. the data you need to train the GMM
'''
def read_face_image(filename):
    image = misc.imread(filename) / 255.0
    bounding_box = np.zeros(image.shape)
    bounding_box[50:100, 60:120, :] = 1
    foreground = image[bounding_box == 1].reshape((50 * 60, 3))
    background = image[bounding_box == 0].reshape((40000 - 50 * 60, 3))
    return image, foreground, background



'''
    implement your GMM and EM algorithm here
'''

def normpdf(Data, Mu, Sigma):
    realmin = sys.float_info[3]
    nbVar, nbData = np.shape(Data)
    Data = np.transpose(Data) - np.tile(np.transpose(Mu), (nbData, 1))
    prob = np.sum(np.dot(Data, np.linalg.inv(Sigma))*Data, 1)
    prob = np.exp(-0.5*prob)/np.sqrt((np.power((2*math.pi), nbVar))*np.absolute(np.linalg.det(Sigma))+realmin)
    return prob

class GMM(object):

    '''
        fit a single gaussian to the data
        @data: an N x D numpy array, where N is the number of observations and D is the dimension (256 for usps digits, 3 for the skin color RGB pixels)
    '''
    def fit_single_gaussian(self, data):
        #TODO
        meu = np.mean(data,axis=0)
        sigma2 = np.mean(data**2,axis=0) - meu
        self.centres = []
        l =[]
        l.append(1.0)
        l.append(meu)
        l.append(sigma2)
        self.centres.append(l)
        self.K = len( self.centres ) 
        

    '''
        implement the em algorithm here
        @data: an N x D numpy array, where N is the number of observations and D is the dimension (256 for usps digits, 3 for the skin color RGB pixels)
        @n_iterations: the number of em iterations
    '''
    def em_algorithm(self, data, n_iterations = 10):
        #TODO
        l = []
        for itr in range(0,n_iterations):
            normalizer = 0.0
            rik = np.zeros( ( data.shape[0], len(self.centres) , data.shape[1] ) ) 
            for i in range(0,data.shape[0]):
                for k in range( 0,len(self.centres) ):
                    rik[i][k] = self.centres[k][0] * normpdf(data[i,:], self.centres[k][1] , self.centres[k][2] )
                   
      
            srik = np.sum(rik,axis = 0)
            srijk = np.sum(srik,axis=1)
            #calculate lambda
            slambda = srik / srijk
            
                   
            # calculating meu
            smeurik = np.zeros( ( data.shape[0], len(self.centres) , data.shape[1] ) ) 
            p = np.reshape( data, ( data.shape[0],1,data.shape[1] ) )
            smeurik = np.reshape( np.sum(rik * p,axis=0), ( 1, len(self.centres), data.shape[1]  ) ) / srik
            print( smeurik.shape)
            
            
                        
            # calculating sigma
            ssigmarik = np.zeros( ( data.shape[0], len(self.centres) , data.shape[1] ) ) 
            for i in range(0,data.shape[0]):
                for k in range( 0,len(self.centres) ):
                    ssigmarik[i][k] = rik[i][k] * np.dot(data[i,:] - smeurik[0][k],data[i,:] - smeurik[0][k])
                    #print( (rik[i][k]).shape, data[i,:].shape)
            ssigmarik = ssigmarik / srik
            print (ssigmarik.shape, srik.shape)
            
        
    '''
        implement the split function here
        generates an initialization for a GMM with 2K components out of the current (already trained) GMM with K components
        @epsilon: small perturbation value for the mean
    '''
    def split(self, epsilon = 0.1):
        #TODO
        
        for i in range(0,len(self.centres)):
            mean1 = self.centres[i][1] - epsilon * self.centres[i][2] 
            l1 = self.centres[i][0]/2.0
            sigma21 = self.centres[i][2]
            mean2 = self.centres[i][1] + epsilon * self.centres[i][2] 
            l2 = self.centres[i][0]/2.0
            sigma22 = self.centres[i][2]
            if i==0:
               gmmK =  [ [l1,mean1,sigma21],[l2,mean2,sigma22] ]
            else:
               gmmk = gmmK.append( [l1,mean1,sigma21] )
               gmmk = gmmK.append( [l2,mean2,sigma22] )
        self.centres = gmmK
        self.K = len( self.centres )
    
    '''
        sample a D-dimensional feature vector from the GMM
    '''
    #def sample(self):
    #    #TODO
        



'''
    Task 2d: synthesizing handwritten digits
    if you implemeted the code in the GMM class correctly, you should not need to change anything here
'''
data = read_usps('usps.txt')
gmm = [ GMM() for _ in range(10) ] # 10 GMMs (one for each digit)
for split in [0, 1, 2]:
    result_image = np.zeros((160, 160))
    print (split)
    for digit in range(1):
        # train the model
        if split == 0:
            gmm[digit].fit_single_gaussian(data[digit])
        else:
            gmm[digit].em_algorithm(data[digit])
        # sample 10 images for this digit
        '''
        for i in range(10):
            x = gmm[digit].sample()
            x = x.reshape((16, 16))
            x = np.clip(x, 0, 1)
            result_image[digit*16:(digit+1)*16, i*16:(i+1)*16] = x
        # save image
        misc.imsave('digits.' + str(2 ** split) + 'components.png', result_image)
        '''
        # split the components to have twice as many in the next run
        gmm[digit].split(epsilon = 0.1)


'''
    Task 2e: skin color model
'''
#image, foreground, background = read_face_image('face.jpg')

'''
    TODO: compute p(x|w=foreground) / p(x|w=background) for each image pixel and manipulate image such that everything below the threshold is black, display the resulting image
'''
