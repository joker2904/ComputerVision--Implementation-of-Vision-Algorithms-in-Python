import cv2
import numpy as np
import time
'''
def get_convolution_using_fourier_transform(image, kernel):
	return None

def task1():
	image = cv2.imread('../data/einstein.jpeg', 0)
	kernel = None #calculate kernel

	conv_result = None #calculate convolution of image and kernel
	fft_result = get_convolution_using_fourier_transform(image, kernel)
	
	#compare results
def normalized_cross_correlation(image, template):
	return None

def task2():
	image = cv2.imread('../data/lena.png', 0)
	template = cv2.imread('../data/eye.png', 0)

	result_ncc = normalized_cross_correlation(image, template)

	#draw rectangle around found location in all four results
	#show the results

def build_gaussian_pyramid_opencv(image, num_levels):
	return None

def build_gaussian_pyramid(image, num_levels, sigma):
	return None

def template_matching_multiple_scales(pyramid_image, pyramid_template, threshold):
	return None

def task3():
	image = cv2.imread('../data/traffic.jpg', 0)
	template = cv2.imread('../data/traffic-template.jpg', 0)

	cv_pyramid = build_gaussian_pyramid_opencv(image, 4)
	mine_pyramid = build_gaussian_pyramid(image, 4)

	#compare and print mean absolute difference at each level
	
	pyramid_template = build_gaussian_pyramid(template, 4)
	result = template_matching_multiple_scales(pyramid_mine, pyramid_template, 0.7)

	#show result

def get_derivative_of_gaussian_kernel(size, sigma):
	return None, None

def task4():
	image = cv2.imread('../data/einstein.jpeg', 0)

	kernel_x, kernel_y = get_derivative_of_gaussian_kernel(5, 0.6)

	edges_x = None #convolve with kernel_x
	edges_y = None #convolve with kernel_y

	magnitude = None #compute edge magnitude
	direction = None #compute edge direction

	cv2.imshow('Magnitude', magnitude)
	cv2.imshow('Direction', direction)

'''
def l2_distance_transform_2D(edge_function, positive_inf, negative_inf):
	return None

def task5():
    image = cv2.imread('../data/traffic.jpg', 0)
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image,100,200)
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  

    edge_function = None #prepare edges for distance transform
    #dist_transfom_mine = l2_distance_transform_2D(edge_function, positive_inf, negative_inf)
    #dist_transfom_cv = None #compute using opencv
    #compare and print mean absolute difference


#task1()
#task2()
#task3()
#task4()
task5()




