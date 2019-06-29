import cv2
import numpy as np
import time
from scipy.signal import fftconvolve

def get_convolution_using_fourier_transform(image, kernel):
	f = np.fft.fft2(image)
	return image


def window_sum_2d(image, window_shape):

    window_sum = np.cumsum(image, axis=0)
    window_sum = (window_sum[window_shape[0]:-1]
                  - window_sum[:-window_shape[0] - 1])

    window_sum = np.cumsum(window_sum, axis=1)
    window_sum = (window_sum[:, window_shape[1]:-1]
                  - window_sum[:, :-window_shape[1] - 1])

    return window_sum

def task1():
	image = cv2.imread('../data/einstein.jpeg', 0)
	x=cv2.getGaussianKernel(ksize=7, sigma=1)
	kernel = x*x.T #calculate kernel
	conv_result = cv2.filter2D(image,-1,kernel) #calculate convolution of image and kernel
	fft_result = get_convolution_using_fourier_transform(image, kernel)
	cv2.imshow("original image",image)
	cv2.imshow("Blurred image",conv_result)
	cv2.imshow("Fourier",fft_result)
	
	#compare results
def normalized_cross_correlation(image, template):
	pad_input=False
	imageShape = image.shape
	image = np.array(image, dtype=np.float64, copy=False)
	padWidth = tuple((width, width) for width in template.shape)
	mode='constant'
	constant_values=0
	if mode == 'constant':
		image = np.pad(image, pad_width=padWidth, mode=mode,
                       constant_values=constant_values)
	else:
		image = np.pad(image, pad_width=padWidth, mode=mode)
	imageWindowSum = window_sum_2d(image, template.shape)
	imageWindowSum2 = window_sum_2d(image ** 2, template.shape)
	templateMean = template.mean()
	templateVolume = np.prod(template.shape)
	templateSsd = np.sum((template - templateMean) ** 2)
	xcorr = fftconvolve(image, template[::-1, ::-1],
                            mode="valid")[1:-1, 1:-1]
	numerator = xcorr - imageWindowSum * templateMean
	denominator = imageWindowSum2
	np.multiply(imageWindowSum, imageWindowSum, out=imageWindowSum)
	np.divide(imageWindowSum, templateVolume, out=imageWindowSum)
	denominator -= imageWindowSum
	denominator *= templateSsd
	np.maximum(denominator, 0, out=denominator)  # sqrt of negative number not allowed
	np.sqrt(denominator, out=denominator)
	result = np.zeros_like(xcorr, dtype=np.float64)
	mask = denominator > np.finfo(np.float64).eps
	result[mask] = numerator[mask] / denominator[mask]
	parts = []
	for i in range(template.ndim):
		if pad_input:
			d0 = (template.shape[i] - 1) // 2
			d1 = d0 + imageShape[i]
		else:
			d0 = template.shape[i] - 1
			d1 = d0 + imageShape[i] - template.shape[i] + 1
		parts.append(slice(d0, d1))

	return result[tuple(parts)]

def task2():
	image = cv2.imread('../data/lena.png', 0)
	template = cv2.imread('../data/eye.png', 0)

	result_ncc = normalized_cross_correlation(image, template)
	loc = np.where( result_ncc >= 0.7)
	w, h = template.shape[::-1]
	for pt in zip(*loc[::-1]):
		cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h
			), (0,255,255), 2)

	cv2.imshow("Detected area",image)
	#draw rectangle around found location in all four results
	#show the results

def build_gaussian_pyramid_opencv(image, num_levels):
	imgpyr = [image]
	tempImg = image
	for i in range(0,num_levels):
		tempImg = cv2.pyrDown(tempImg)
		imgpyr.append(tempImg)
	imgpyr.reverse()
	return imgpyr

def build_gaussian_pyramid(image, num_levels, sigma):
	selfGaussianPyramid=[image]
	out = image.copy()
	for i in range(0,num_levels):
		kernel = cv2.getGaussianKernel(5, sigma)
		kernel = kernel*kernel.T
		conv_result = cv2.filter2D(out,-1,kernel)
		out = conv_result[::2,::2]
		selfGaussianPyramid.append(out)
		cv2.imshow("pyramid images",out)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	selfGaussianPyramid.reverse()	
	return selfGaussianPyramid

def template_matching_multiple_scales(pyramid_image, pyramid_template, threshold):
	image = cv2.imread('../data/traffic.jpg', 0)
	template = cv2.imread('../data/traffic-template.png', 0)
	result_images=[]
	for i in range(len(pyramid_image)):
		image=pyramid_image[i]
		tpl=pyramid_template[i]
		if i==0:
			result=normalized_cross_correlation(image,tpl)

		else:
			masking=cv2.pyrUp(threshimg)
			mask8u = cv2.inRange(masking, 0, 255)
			_,contours,_ = cv2.findContours(mask8u, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_NONE)
			tplH, tplW = tpl.shape[:2]
			for cnt in contours:
				x, y, w, h = cv2.boundingRect(cnt)
				src = image[y:y+h+tplH, x:x+w+tplW]
				result = cv2.matchTemplate(src, tpl, cv2.TM_CCORR_NORMED)

		T, threshimg = cv2.threshold(result, threshold, 1., cv2.THRESH_TOZERO)
		result_images.append(threshimg)

	return threshimg

def task3():
	image = cv2.imread('../data/traffic.jpg', 0)
	template = cv2.imread('../data/traffic-template.png', 0)

	cv_pyramid = build_gaussian_pyramid_opencv(image, 4)
	mine_pyramid = build_gaussian_pyramid(image, 4, 1)
    # ABSOLUTE MEAN VALUE DIFFERNECE HERE DO NOT FORGET

	pyramid_template = build_gaussian_pyramid(template, 4,1)
	#compare and print mean absolute difference at each level
	start=time.clock()
	template_matching_ncc = normalized_cross_correlation(image,template)	
	print(time.clock()-start)
	startFast = time.clock()
	result = template_matching_multiple_scales(mine_pyramid, pyramid_template, 0.7)
	print(time.clock()-startFast)

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

def l2_distance_transform_2D(edge_function, positive_inf, negative_inf):
	return None

def task5():
	image = cv2.imread('../data/traffic.jpg', 0)

	edges = None #compute edges
	edge_function = None #prepare edges for distance transform

	dist_transfom_mine = l2_distance_transform_2D(edge_function, positive_inf, negative_inf)
	dist_transfom_cv = None #compute using opencv

	#compare and print mean absolute difference


task1()
task2()
task3()
task4()
task5()




