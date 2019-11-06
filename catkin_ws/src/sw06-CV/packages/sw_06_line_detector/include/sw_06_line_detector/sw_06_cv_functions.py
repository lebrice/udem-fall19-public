import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided

# import rospy

## Software Exercise 6: Choose your category (1 or 2) and replace the cv2 code by your own!

## CATEGORY 1
def inRange(hsv_image, low_range, high_range):
	return cv2.inRange(hsv_image, low_range, high_range)

def bitwise_or(bitwise1, bitwise2):
	return cv2.bitwise_or(bitwise1, bitwise2)

def bitwise_and(bitwise1, bitwise2):
	return cv2.bitwise_and(bitwise1, bitwise2)

def getStructuringElement(shape, size):
	return cv2.getStructuringElement(shape, size)

def dilate(bitwise, kernel):
	return cv2.dilate(bitwise, kernel)


## CATEGORY 2
def Canny(image, threshold1, threshold2, apertureSize=3):
	smoothed_image = gaussian_blurring(image, std=1, kernel_size=5)
	dx, dy = image_derivatives(smoothed_image)
	edge_gradients = np.sqrt(dx ** 2 + dy ** 2)
	gradient_angles = snap_angles(np.arctan2(dy / dx))
	
	mask = non_maximum_suppression(smoothed_image, edge_gradients, gradient_angles)

	return cv2.Canny(image, threshold1, threshold2, apertureSize=3)


def non_maximum_suppression(image, edge_gradients, gradient_directions):
	"""

	Arguments:
		image {[type]} -- [description]
		edge_gradients {[type]} -- [description]
		gradient_directions {[type]} -- [description]
	"""
	# 	After getting gradient magnitude and direction, a full scan of image is done to remove any unwanted pixels which may not constitute the edge. For this, at every pixel, pixel is checked if it is a local maximum in its neighborhood in the direction of gradient. Check the image below:

	# Point A is on the edge ( in vertical direction). Gradient direction is normal to the edge. Point B and C are in gradient directions. So point A is checked with point B and C to see if it forms a local maximum. If so, it is considered for next stage, otherwise, it is suppressed ( put to zero).

	# In short, the result you get is a binary image with “thin edges”.
	
	# Get where to check depending on the "direction"
	direction_offset_x = np.cos(gradient_directions)
	direction_offset_y = np.sin(gradient_directions)
	print(direction_offset_x)
	print(direction_offset_y)
			
	for i, row in enumerate(image):
		for j, pixel in enumerate(row):
			pass

def snap_angles(angles):
	"""Snaps a given set of angles to one of the horizontal, vertical, or one of the two diagonal orientations.
	Arguments:
		angles -- an array of anges in radians
	"""
	pi_over_four = np.pi / 4
	return np.round(angles / pi_over_four) * pi_over_four 



def image_derivatives(image):
	""" Computes the Sobel X and Y operators for this image.
	Loosely based on https://en.wikipedia.org/wiki/Sobel_operator

	Arguments:
		image {[type]} -- [description]
	
	Returns:
		[type] -- [description]
	"""
	sobel_sign = np.array([[-1, 0, 1]])
	sobel_mag = np.array([[1, 2, 1]])

	temp = conv2d(image, sobel_sign)
	image_dx = conv2d(temp, sobel_mag.T)

	temp = conv2d(image, sobel_mag)
	image_dy = conv2d(temp, sobel_sign.T)

	return image_dx, image_dy

def conv2d(x, kernel, stride=1, padding="auto", padding_mode="edge"):
	"""
	TAKEN AND ADAPTED FROM https://stackoverflow.com/questions/54962004/implement-max-mean-poolingwith-stride-with-numpy
	ALSO INSPIRED FROM https://cs231n.github.io/convolutional-networks/
	
	2D Pooling 

	Parameters:
		A: input 2D array
		kernel: int, the size of the window
		stride: int, the stride of the window
		padding: int or string, implicit zero paddings on both sides of the input
	"""
	# Padding
	assert len(kernel.shape) == 2, "kernel should be 2d."
	assert kernel.shape[0] % 2 == 1 and kernel.shape[1] % 2 == 1, "only odd-sized kernels are allowed"
	kernel_size = kernel.shape[0]

	if padding == "auto":
		padding = np.array(kernel.shape) // 2
	
	x = np.pad(x, padding, mode=padding_mode)
	# Window view of X
	output_shape = ((x.shape[0] - kernel.shape[0])//stride + 1,
					(x.shape[1] - kernel.shape[1])//stride + 1)
	x_w = as_strided(
		x,
		shape=output_shape + kernel.shape,
		strides = (
			stride*x.strides[0],
			stride*x.strides[1]
		) + x.strides
	)
	# ADAPTATION BELOW:
	# patches is [#patches, k, k]
	patches = x_w.reshape(-1, *kernel.shape)
	flattened_kernel = kernel.flat
	flattened_patches = patches.reshape([patches.shape[0], -1])
	return np.dot(flattened_patches, flattened_kernel).reshape(output_shape)


def separable_conv2d(x, kernel_1d, stride=1, padding="auto"):
	assert len(kernel_1d.shape) == 1, kernel_1d
	k = kernel_1d.shape[0]
	k1 = kernel_1d.reshape([1, k])
	result_1 = conv2d(x, k1, stride, padding)
	k2 = k1.T
	result_2 = conv2d(result_1, k2, stride, padding)
	return result_2
	

def gaussian_kernel_1d(std=1, kernel_size=5):
	x = np.arange(-(kernel_size//2), (kernel_size//2)+1, dtype=float)
	g = np.exp(- (x**2 / (2 * std**2))) / (np.sqrt(2 * np.pi) * std)
	# normalize the sum to 1
	g /= np.sum(g)
	return g


def gaussian_blurring(image, std=1, kernel_size=5):
	kernel = gaussian_kernel_1d(std, kernel_size)
	return separable_conv2d(image, kernel)


def gaussian_derivative_filtering(image, std=1, kernel_size=5):
	kernel = gaussian_derivative_1d(std, kernel_size)
	return separable_conv2d(image, kernel)


def gaussian_derivative_1d(sigma, kernel_size):
	"""
	ADAPTED FROM https://github.com/scipy/scipy/blob/5681835ec51b728fa0ea6237d46aa8032b9e1400/scipy/ndimage/filters.py#L136
	Computes a 1D Gaussian derivative kernel's
	"""
	# #0th order kernel.
	phi_x = gaussian_kernel_1d(sigma, kernel_size)
		
	# f(x) = q(x) * phi(x) = q(x) * exp(p(x))
	# f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
	# p'(x) = -1 / sigma ** 2
	# Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
	# coefficients of q(x)
	q = np.zeros(2)
	q[0] = 1
	D = np.diag([1], 1)  # D @ q(x) = q'(x)
	sigma2 = sigma * sigma
	P = np.diag(np.ones(1)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
	Q_deriv = D + P
	q = Q_deriv.dot(q)
	x = np.arange(-(kernel_size//2), kernel_size//2 + 1)
	exponent_range = np.arange(2)
	q = (x[:, None] ** exponent_range).dot(q)
	return q * phi_x



## CATEGORY 3 (This is a bonus!)
def HoughLinesP(image, rho, theta, threshold, lines, minLineLength, maxLineGap):
	return cv2.HoughLinesP(image, rho, theta, threshold, lines, minLineLength, maxLineGap)



X = np.array([
	[1, 1, 2, 4, 5],
	[5, 6, 7, 8, 6],
	[3, 2, 1, 0, 7],
	[5, 1, 9, 3, 6],
	[1, 2, 3, 4, 8],
])
bob = Canny(X, None, None)
print(bob)
