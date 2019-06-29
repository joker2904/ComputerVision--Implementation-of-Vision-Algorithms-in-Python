#!/usr/bin/python3.5

import numpy as np
from scipy import misc
from scipy.stats import multivariate_normal
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

def normpdf(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*np.pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = np.matrix(x - mu)
        inv = np.linalg.inv(sigma)    
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")


'''
    implement your GMM and EM algorithm here
'''
class GMM(object):

    '''
        fit a single gaussian to the data
        @data: an N x D numpy array, where N is the number of observations and D is the dimension (256 for usps digits, 3 for the skin color RGB pixels)
    '''    
    def probability(self,x):
        prob = 0.0
        for j in range(self.k):
            prob += self.lambdak[j] * normpdf(x, self.mean_arr[j, :].A1, self.sigma_arr[j, :]) 
            return prob
    
    def loglikelihood(self):
        vloglikelihood = 0
        
        for i in range(self.m):
            x = 0
            for j in range(self.k):
                x += self.lambdak[j] * normpdf(self.data[i, :], self.mean_arr[j, :].A1, self.sigma_arr[j, :]) 
                #print ('log ::', x) 
            vloglikelihood += np.log(x) 
        return vloglikelihood
    
    def em_fit(self,niter):
        self.e_step()
        self.m_step(niter)
        
    def e_step(self):
        for i in range(self.m):
            denominator = 0
            for j in range(self.k):
                numerator = self.lambdak[j] * normpdf(self.data[i, :],  self.mean_arr[j, :].A1, self.sigma_arr[j,:]) 
                denominator += numerator
                self.rik[i, j] = numerator
                #print('e-step::',self.rik[i,j])
            self.rik[i, :] /= denominator
       
            
    def m_step(self,niter):
        for j in range(self.k):
            const = self.rik[:, j].sum()
            self.lambdak[j] = 1/self.m * const
            _mu_j = np.zeros(self.n)
            _sigma_j = np.zeros((self.n, self.n))
            for i in range(self.m):
                _mu_j += (self.data[i, :] * self.rik[i, j])
                _sigma_j += self.rik[i, j] * ((self.data[i, :] - self.mean_arr[j, :]).T * (self.data[i, :] - self.mean_arr[j, :]))
            self.mean_arr[j] = _mu_j / const
            self.sigma_arr[j] = _sigma_j / const 
            # The next line is commented. It added a regularization factor to the diagonal elements of the variance matrix
            # This was still unable to address the problem of singularity, so we tool another approach
            #self.sigma_arr[j] = self.sigma_arr[j]  +  self.reg_cov
            if( np.linalg.det(self.sigma_arr[j]) == 0.0): # approach to fix singularity issue
               self.sigma_arr[j] = np.identity(self.n)    # reset the covariance matrix to identity matrix 
            
    
    def fit_single_gaussian(self, data):
        #TODO
        # Our logic :
        # Initialize the mean of the multivariate gaussian from the dataset
        # Inititialize the co-variance matrix as a identity matrix, which is the diagonal matrix
        # It addresses the non-singularity problem 
        self.k = 1
        self.data = data
        self.lambdak = np.ones(self.k)/self.k
        self.m, self.n = data.shape
        self.reg_cov = 1e-6*np.identity(self.n) # not used in code. It is regularization parameter 
        self.rik = np.asmatrix(np.empty((self.m, self.k), dtype=float))
        self.mean_arr = np.asmatrix(np.empty((self.k, self.n), dtype=float))
        self.sigma_arr = np.array([np.asmatrix(np.identity(self.n), dtype=float) for i in range(self.k)])
        self.mean_arr[0] = np.reshape(np.mean(data, axis=0),(1,self.n))           
             

    '''
        implement the em algorithm here
        @data: an N x D numpy array, where N is the number of observations and D is the dimension (256 for usps digits, 3 for the skin color RGB pixels)
        @n_iterations: the number of em iterations
    '''
    def em_algorithm(self, data, n_iterations = 10, tolerance=1e-4):
        #TODO
        num_iters = 0
        value_loglikelihood = 1
        previous_value_loglikelihood = 0
        while( (value_loglikelihood - previous_value_loglikelihood) > tolerance and num_iters <= n_iterations ):
            previous_value_loglikelihood = self.loglikelihood()
            self.em_fit(num_iters)
            num_iters += 1
            value_loglikelihood = self.loglikelihood()
            print('Iteration %d: log-likelihood is %.6f'%(num_iters, value_loglikelihood))
        print('Terminate at %d-th iteration:log-likelihood is %.6f'%(num_iters, value_loglikelihood))

    '''
        implement the split function here
        generates an initialization for a GMM with 2K components out of the current (already trained) GMM with K components
        @epsilon: small perturbation value for the mean
    '''
    def split(self, epsilon = 0.1):
        #TODO     
        temp_lambdak = np.empty(2*self.k)
        temp_mean_arr = np.asmatrix(np.empty((2*self.k, self.n), dtype=float))
        temp_sigma_arr = np.array([np.asmatrix(np.identity(self.n), dtype=float) for i in range(2*self.k)])
        for i in range(self.k):
            temp_mean_arr[2*i]   = self.mean_arr[i] - epsilon * np.sqrt(np.linalg.det(self.sigma_arr[i]))
            temp_mean_arr[2*i+1] = self.mean_arr[i] + epsilon * np.sqrt(np.linalg.det(self.sigma_arr[i]))
            temp_sigma_arr[2*i]   = self.sigma_arr[i]
            temp_sigma_arr[2*i+1] = self.sigma_arr[i]
            temp_lambdak[2*i]   = self.lambdak[i] / 2.0
            temp_lambdak[2*i+1] = self.lambdak[i] / 2.0
        
        self.lambdak = temp_lambdak
        self.mean_arr = temp_mean_arr
        self.sigma_arr = temp_sigma_arr
        self.k = 2*self.k
        self.rik = np.asmatrix(np.empty((self.m, self.k), dtype=float))
        
    
    
    '''
        sample a D-dimensional feature vector from the GMM
    '''
    def sample(self):
        #TODO
        # return one of the mean from the split 
        p = np.random.randint(self.k, size=1)[0]
        return self.mean_arr[p,:]
    
      
        



'''
    Task 2d: synthesizing handwritten digits
    if you implemeted the code in the GMM class correctly, you should not need to change anything here

data = read_usps('usps.txt')

gmm = [ GMM() for _ in range(10) ] # 10 GMMs (one for each digit)
for split in [0, 1, 2, 3]:
    result_image = np.zeros((160, 160))
    for digit in range(10):
        # train the model
        print("Digit :",digit)
        if split == 0:
            gmm[digit].fit_single_gaussian(data[digit])
        else:
            print("Run em algorithm for split :",split)
            gmm[digit].em_algorithm(data[digit],100)
        # sample 10 images for this digit
        for i in range(10):
            x = gmm[digit].sample()
            x = x.reshape((16, 16))
            x = np.clip(x, 0, 1)
            result_image[digit*16:(digit+1)*16, i*16:(i+1)*16] = x
        # save image
        misc.imsave('digits.' + str(2 ** split) + 'components.png', result_image)
        # split the components to have twice as many in the next run
        gmm[digit].split(epsilon = 0.1)



    Task 2e: skin color model
'''
image, foreground, background = read_face_image('face.jpg')

'''
    TODO: compute p(x|w=foreground) / p(x|w=background) for each image pixel and manipulate image such that everything below the threshold is black, display the resulting image
'''
#print( image.shape)
#print( foreground )
#print( background )

gmm_foreground = GMM()
gmm_background = GMM()
for split in [0, 1, 2]: #, 3]:
    if split == 0:
       gmm_foreground.fit_single_gaussian(foreground)
       gmm_background.fit_single_gaussian(background)
    else:
       gmm_foreground.em_algorithm(foreground,10)
       gmm_background.em_algorithm(background,10)
    gmm_foreground.split(epsilon = 0.1)
    gmm_background.split(epsilon = 0.1)
 
result_image_face = np.zeros((200, 200))
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        #print('Image-' ,image[i][j])
        thresh = gmm_foreground.probability(image[i][j]) / gmm_background.probability(image[i][j])
        if thresh > 0.1:
           result_image_face[i][j] = 1
        else:
           result_image_face[i][j] = 0
misc.imsave('threshhold_image_2.png', result_image_face)           
    



