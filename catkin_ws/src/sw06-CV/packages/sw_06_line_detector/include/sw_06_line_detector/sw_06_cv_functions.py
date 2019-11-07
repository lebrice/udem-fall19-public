import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
import traceback
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

	if apertureSize != 3:
		import warnings
		warnings.warn(UserWarning("Using apertureSize of 3."))
		apertureSize = 3
	
	num_channels = image.shape[-1] if len(image.shape) == 3 else 1
	try:
		outputs = []
		edge_gradients 	= np.zeros_like(image)
		dxs 			= np.zeros_like(image)
		dys 			= np.zeros_like(image)

		for channel in range(num_channels):
			smoothed_image_channel = gaussian_blurring(image[..., channel], std=1, kernel_size=5)
			dx, dy = image_derivatives(smoothed_image_channel)
			dxs[..., channel] = dx
			dys[..., channel] = dy
		
		edge_gradients = np.sqrt(dxs ** 2 + dys ** 2)
		
		# TODO: we take the mean over the channels axis. :(
		edge_gradients = np.mean(edge_gradients, axis=-1)
		dx = np.mean(dxs, axis=-1)
		dy = np.mean(dys, axis=-1)
		
		# TODO: use the grad magnitude and dx and dy's from the channel with the highest gradient.
		# max_grad_indices = np.argmax(edge_gradients, axis=-1)
		# edge_gradients = edge_gradients[max_grad_indices]
		# dx = dxs[max_grad_indices]
		# dy = dys[max_grad_indices]

		
		gradient_directions = snap_angles(np.arctan2(dy, dx))
		filtered_image = non_maximum_suppression(image, edge_gradients, gradient_directions)
		result = hysteresis_thresholding(filtered_image, edge_gradients, threshold1, threshold2)

		# expected = cv2.Canny(image, threshold1, threshold2, apertureSize=3)
		# print(result.shape, expected.shape)
		# print(np.mean(result), np.mean(expected))
		# print(np.count_nonzero(result), np.count_nonzero(expected))
		# print(np.median(result), np.median(expected))
		# print(np.max(result), np.max(expected))
		# return expected
		return result
	except Exception as e:
		traceback.print_exc()


def hysteresis_thresholding(image, image_gradients, min_val, max_val):
	
	
	strong_indices  = np.where(image_gradients >= max_val)
	off_indices 	= np.where(image_gradients < min_val)
	weak_indices 	= np.where((min_val <= image_gradients) & (image_gradients < max_val))
	
	# the set of all 'strong' indices.
	# strong_indices = np.dstack(strong_indices)
	image_h = image.shape[0]
	image_w = image.shape[1]
	
	def neighbourhood(index):
		i, j = index
		# TODO: maybe use pure numpy if there is time left.
		# row, col = np.indices([3,3]) - 1
		# print(image[row, col])
		min_x = max(i-1, 0)
		max_x = min(i+1, image_w)
		min_y = max(j-1, 0)
		max_y = min(j+1, image_h)
		indices = set(
			(x, y)
			for x in range(min_x, max_x)
			for y in range(min_y, max_y)
		)
		indices.discard((i, j))
		return indices
		# return np.array(indices)
	
	strong_indices_set = set(zip(*strong_indices))
	weak_indices_set = set(zip(*weak_indices))
	
	unexplored = set(strong_indices_set)
	while unexplored:
		index = unexplored.pop()
		# print(index)
		neighbours = neighbourhood(index)
		# print(neighbours)
		# print("strong indices", strong_indices_set)
		# print("weak indices", weak_indices_set)
		weak_neighbours = neighbours & weak_indices_set
		# print("weak neighbours:", weak_neighbours)
		# explore the weak neighbours.
		unexplored.update(weak_neighbours)
		# store the (newly) strong neighbours.
		strong_indices_set.update(weak_neighbours)
		weak_indices_set.difference_update(weak_neighbours)
	
	mask = np.zeros_like(image, dtype=bool)
	kept_indices = (
		np.array([x for (x, y) in strong_indices_set], dtype=int),
		np.array([y for (x, y) in strong_indices_set], dtype=int)
	)
	out = np.copy(image)
	mask[kept_indices] = True
	out[~mask] = 0.
	return out

def non_maximum_suppression(image, image_gradients, gradient_directions):
	"""Non-maximum suppression
	
	To be honest, I'm very proud of this piece of code. No for-loops were needed.

	Arguments:
		image {[type]} -- the image to preform non-maximum suppresion on.
		gradient_directions {[type]} -- the gradient directions
	"""	
	# Get where to check depending on the "direction"
	direction_offset_x = np.round(np.cos(gradient_directions)).astype(int)
	direction_offset_y = np.round(np.sin(gradient_directions)).astype(int)
	direction_offset = np.dstack((direction_offset_x, direction_offset_y))
		
	# the (i, j) indices of all points in the image.
	row, col = np.indices(image_gradients.shape)
	# in order not to cause any indexing errors, we create a
	# padded version of the image with the edge values duplicated.
	# a pixel at (row, col) in the image is located at (row+1, col+1) in the padded image.
	image_ = np.pad(image_gradients, 1, mode="edge")
	row_, col_ = row + 1, col + 1
	# get the image pixels before and after each pixel in the image. 
	pixel_middle = image_gradients[row, col]
	pixel_forward  = image_[row_ + direction_offset_x, col_ + direction_offset_y]
	pixel_backward = image_[row_ - direction_offset_x, col_ - direction_offset_y]

	higher_than_forward  = pixel_middle > pixel_forward
	higher_than_backward = pixel_middle > pixel_backward
	is_local_maximum = higher_than_backward & higher_than_forward
	out = np.copy(image)
	out[~is_local_maximum] = 0
	return out
			

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

# X = np.random.rand(80, 160, 3)
# print(X.shape)
# # X = np.array([
# # 	[1, 1, 2, 4, 5],
# # 	[5, 6, 12, 8, 6],
# # 	[3, 2, 15, 123, 7],
# # 	[5, 1, 23, 3, 6],
# # 	[1, 2, 3, 4, 8],
# # ])

# # X = np.tile(X[..., np.newaxis], (1, 1, 3))
# bob = Canny(X, 0.1, 25)
# print(bob.shape)
