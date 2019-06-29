import numpy as np
import time
import cv2
import utils
import task1
import task2

hands_orig_train = 'data/hands_orig_train.txt.new'
hands_aligned_test = 'data/hands_aligned_test.txt.new'
hands_aligned_train = 'data/hands_aligned_train.txt.new'

def get_keypoints(path):
    data_info = utils.load_data(path)
    kpts = (data_info['samples'])
    #print(data_info['data_dim'])
    #print(data_info['num_data'])
    # Your part here
    #print (kpts.shape)
    return kpts

def task_1():
    # Loading Trainig Data
    kpts = get_keypoints(hands_orig_train)
    kpts = kpts.transpose()
    #print(kpts.shape)
    # calculate mean
    # ToDO
    # we want to visualize the data first
    # ToDO

    task1.procrustres_analysis(kpts,100)


def task_2_1():
    # ============= Load Data =================
    kpts = get_keypoints(hands_aligned_train).transpose()
    print(kpts.shape)
    ### Your part here ##
    #kpts = utils.convert_samples_to_xy(kpts)
    #print(kpts.shape)
    #task2.visualize_hands2(kpts,'Raw data')
    #####################

    mean, pcs, pc_weights = task2.train_statistical_shape_model(kpts)

    return mean, pcs, pc_weights

def task_2_2(mean, pcs, pc_weights):
    # ============= Load Data =================
    kpts = get_keypoints(hands_aligned_test).transpose()
    print(kpts.shape)
    # Your part here

    pointsrecon = task2.reconstruct_test_shape(kpts, mean, pcs, pc_weights)

    time.sleep(20)

if __name__ == '__main__':
    print("Running Task 1")
    task_1()

    print("Running Task 2.1")
    mean, pcs, pc_weights = task_2_1()

    print("Running Task 2.2")
    task_2_2(mean, pcs, pc_weights)
