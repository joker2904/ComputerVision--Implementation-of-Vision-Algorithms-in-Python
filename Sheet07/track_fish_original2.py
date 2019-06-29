import cv2
import numpy as np
import os

IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')

INIT_X = 448
INIT_Y = 191
INIT_WIDTH = 38 
INIT_HEIGHT = 33 

INIT_BBOX = [INIT_X, INIT_Y, INIT_WIDTH, INIT_HEIGHT]


def load_frame(frame_number):
    """
    :param frame_number: which frame number, [1, 32]
    :return: the image
    """
    image = cv2.imread(os.path.join(IMAGES_FOLDER, '%02d.png' % frame_number))
    return image


def crop_image(image, bbox):
    """
    crops an image to the bounding box
    """
    x, y, w, h = tuple(bbox)
    return image[y: y + h, x: x + w]


def draw_bbox(image, bbox, thickness=2, no_copy=False):
    """
    (optionally) makes a copy of the image and draws the bbox as a black rectangle.
    """
    x, y, w, h = tuple(bbox)
    if not no_copy:
        image = image.copy()
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), thickness)

    return image

'''
def compute_histogram(image):
    # implement here


def compare_histogram(hist1, hist2):
    # implement here

    hist_comp_val = ?

    likelihood = np.exp(-hist_comp_val * 20.0)
    return likelihood
'''


class Position(object):
    """
    A general class to represent position of tracked object.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_bbox(self):
        """
        since the width and height are fixed, we can do such a thing.
        """
        return [self.x, self.y, INIT_WIDTH, INIT_HEIGHT]

    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Position(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return Position(self.x * other, self.y * other)

    def __repr__(self):
        return "[%d %d]" % (self.x, self.y)

    def make_ready(self, image_width, image_height):
        # convert to int
        self.x = int(round(self.x))
        self.y = int(round(self.y))

        # make sure inside the frame
        self.x = max(self.x, 0)
        self.y = max(self.y, 0)
        self.x = min(self.x, image_width)
        self.y = min(self.y, image_height)


class Particle(object):

    def __init__(self):
        self.fitness = 0
      
    def __init__(self,img, inbb, refhist):
        self.bb = inbb;
        # calculate histogram        
        self.hist = cv2.calcHist([crop_image(img,inbb)], [0], None, [64], [0, 256],True,False)
        #hist = cv2.normalize(hist,hist).flatten()
        # calculate fitness
        self.fitness = np.exp(-cv2.compareHist(refhist,self.hist,cv2.HISTCMP_BHATTACHARYYA)*20.0)
 
    def get_bbox(self):
        return self.bb
    
    def get_hist(self):
        return self.hist
    
   


class ParticleFilter(object):
    def __init__(self, du=0, sigma=20, num_particles=500):
        self.template = None  # the template (histogram) of the object that is being tracked.
        self.position = None  # we don't know the initial position still!
        self.particles = []  # we will store list of particles at each step here for displaying.
        self.fitness = []  # particle's fitness values
        self.cumulFit = []
        self.du = du
        self.sigma = sigma
        self.num_particles = num_particles

    def init(self, frame, bbox):
        self.position = Position(x=bbox[0], y=bbox[1])  # initializing the position
        # implement here ...
        self.min_width  = 0
        self.min_height = 0 
        self.max_width  = frame.shape[1] - 1 - int(INIT_WIDTH)
        self.max_height = frame.shape[0] - 1 - int(INIT_HEIGHT)
        
         
        # calculate histogram        
        self.ref_hist = cv2.calcHist([crop_image(frame,bbox)], [0], None, [64], [0, 256],True,False)
        #self.ref_hist = cv2.normalize(self.ref_hist,self.ref_hist).flatten()
            
        # initialize particles
        for pid in range(0,self.num_particles):
            temp_particle = Particle(frame,bbox,self.ref_hist)
            self.particles.append(temp_particle)
    
    def track(self, new_frame):
        # implement here ...
        new_particles = []
        for pid in range(0,self.num_particles):
            sid=self.sampleParticle()
            new_particles.append( self.applyMotionModel(self.particles[sid].bb,new_frame) )   
        self.particles=new_particles
        maxsid=self.sampleParticle()
        self.ref_hist = self.particles[maxsid].get_hist()
        #add this new bounding box to the current frame for display

    def display(self, current_frame,frame_number):
        cv2.imshow('frame', current_frame)

        maxfit = 0.0
        p = -1
        frame_copy = current_frame.copy()
        for i in range(len(self.particles)):
            if self.particles[i].fitness > maxfit:
               p = i
        draw_bbox(frame_copy, self.particles[p].get_bbox(), thickness=3, no_copy=True)

        cv2.imshow('particles '+str(frame_number), frame_copy)
        #cv2.waitKey(100)
        cv2.waitKey(0)
        cv2.destroyAllWindows()  

    # Sample a particle, with the minimum distance from reference histogram
    def sampleParticle(self):
        maxfit = 0.0
        index = -1
        for pid in range(0,self.num_particles):
            if self.particles[pid].fitness > maxfit:
               maxfit = self.particles[pid].fitness
               index = pid
        return index
 
    def applyMotionModel(self,bb,new_frame):
        new_position = Position(x=bb[0], y=bb[1])
        max_P = Particle(new_frame, new_position.get_bbox() ,self.ref_hist)
        maxfit = max_P.fitness
        # Generate 50 samples , normally distributed with the current mean and sigma
        for i in range(0,50):
            cond = True
            while(cond):
               x = new_position.x + np.random.normal(self.du,self.sigma,1)
               y = new_position.y + np.random.normal(self.du,self.sigma,1)  
               if (x > self.min_width and x < self.max_width) and (y > self.min_height and y < self.max_height):
                   new_position.x = int(x)
                   new_position.y = int(y)
                   newP = Particle(new_frame, new_position.get_bbox() ,self.ref_hist)
                   if newP.fitness > maxfit:
                      max_P = newP
                   cond = False
        return max_P 
    






def main():
    np.random.seed(0)
    DU = 0
    SIGMA = 5

    cv2.namedWindow('particles')
    cv2.namedWindow('frame')
    frame_number = 1
    frame = load_frame(frame_number)

    tracker = ParticleFilter(du=DU, sigma=SIGMA)
    tracker.init(frame, INIT_BBOX)
    tracker.display(frame,1)

    for frame_number in range(2, 33):
        frame = load_frame(frame_number)
        tracker.track(frame)
        tracker.display(frame,frame_number)


if __name__ == "__main__":
    main()
