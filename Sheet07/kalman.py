import numpy as np
import matplotlib.pylab as plt

observations = np.load('observations.npy')


def get_observation(t):
    return observations[t]


class KalmanFilter(object):
    def __init__(self, Lambda, meu_p, sigma_p, Phi, meu_m, sigma_m):
        self.Lambda = Lambda
        
        self.sigma_p = sigma_p
        self.meu_p = meu_p
        
        self.Phi = Phi
        
        self.sigma_m = sigma_m
        self.meu_m = meu_m
        
        self.state = None
        self.convariance = None
        
        
        

    def init(self, init_state):
        self.state = init_state
        self.convariance = np.eye(init_state.shape[0]) * 0.01

    def track(self, xt):
        # implement here
        # State prediction
        meu_plus = self.meu_p + np.dot(self.Lambda,self.state)           
        
        # covariance prediction
        sigma_plus = self.sigma_p +  np.dot( np.dot(self.Lambda , self.convariance ), (self.Lambda).transpose() )
        
        #compute kalman gain
        temp = self.sigma_m +  np.dot( np.dot( self.Phi, sigma_plus ), (self.Phi).transpose() )
        temp = np.linalg.inv(temp)
        K = np.dot( np.dot( sigma_plus, (self.Phi).transpose() ), temp)
        
        #state update
        temp = xt - self.meu_m - np.dot( self.Phi, meu_plus)
        self.state = meu_plus + np.dot( K , temp )
        
        #covariance update..
        self.covariance = np.dot( ( np.identity(4) - np.dot(K,self.Phi) ),sigma_plus)
        
        pass

    def get_current_location(self):
        return self.Phi @ self.state


def main():
    
    init_state = np.array([0, 1, 0, 0])
    meu_p = np.array([0.00001, 0.00000001, 0.0001, 0.0000001])
    meu_m = np.array([0.00001, 0.00000001])
    #for t in range(len(observations)):
    #    print(get_observation(t))
        
    Lambda = np.identity(init_state.shape[0]) #np.array([?])

    sp = 0.01
    sigma_p = np.array([[sp, 0, 0, 0],
                        [0, sp, 0, 0],
                        [0, 0, sp * 4, 0],
                        [0, 0, 0, sp * 4]])
    #Phi = np.zeros((2,4)) # np.array([])
    Phi = 2.5*np.array([[1,0,0,0],
                    [0,1,0,0]])

    sm = 0.05
    sigma_m = np.array([[sm, 0], [sm, 0]])

    tracker = KalmanFilter(Lambda, meu_p, sigma_p, Phi, meu_m, sigma_m)
    tracker.init(init_state)

    track = []
    for t in range(len(observations)):
        tracker.track(get_observation(t))
        track.append(tracker.get_current_location())
    
    plt.figure()
    plt.plot([x[0] for x in observations], [x[1] for x in observations])
    plt.plot([x[0] for x in track], [x[1] for x in track])
    plt.show()
    

if __name__ == "__main__":
    main()
