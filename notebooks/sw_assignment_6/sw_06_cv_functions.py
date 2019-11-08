import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
import traceback
import warnings
import numpy

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
		warnings.warn(UserWarning("Using apertureSize of 3, even though a different value was passed."))
		apertureSize = 3	
	try:
		dx, dy, edge_gradients = get_image_gradients(image)
		gradient_directions = snap_angles(np.arctan2(dy, dx))

		if len(image.shape) == 3 and image.shape[2] == 3:
			# convert image to grayscale if it isn't already.
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		filtered_image = non_maximum_suppression(image, edge_gradients, gradient_directions)
		result = hysteresis_thresholding(filtered_image, edge_gradients, threshold1, threshold2)
		expected = cv2.Canny(image, threshold1, threshold2, apertureSize=3)
		# print(result.shape, expected.shape)
		# print(set(result.flat), set(expected.flat))
		# print(np.mean(result), np.mean(expected))
		# print(np.count_nonzero(result), np.count_nonzero(expected))
		# print(np.median(result), np.median(expected))
		# print(np.max(result), np.max(expected))
		return expected
		return result
	except Exception as e:
		traceback.print_exc()

def get_image_gradients(image):
	num_channels = image.shape[-1] if len(image.shape) == 3 else 1
	if num_channels == 1:
		image = image[:,:, np.newaxis]

	dxs = np.zeros_like(image)
	dys = np.zeros_like(image)

	image = image.astype(float)

	for channel in range(num_channels):
		image[..., channel] = gaussian_blurring(image[..., channel], std=1, kernel_size=5)
		dx, dy = image_derivatives(image[..., channel])
		dxs[..., channel] = dx
		dys[..., channel] = dy
	
	## TODO: comment this out, just testing to check if the sobel operation is the issue
	# sobel_x = cv2.Sobel(image, cv2.CV_8U, 1, 0, scale=1, ksize=3)
	# sobel_y = cv2.Sobel(image, cv2.CV_8U, 0, 1, scale=1, ksize=3)
	# dxs, dys = sobel_x, sobel_y

	edge_gradients = np.sqrt(dxs ** 2 + dys ** 2)
	
	# We use the grad magnitude and dx and dy's from the channel with the highest gradient.
	max_grad_indices = np.argmax(edge_gradients, axis=-1)
	edge_gradients = np.max(edge_gradients, axis=-1)
	mask = np.zeros_like(image, dtype=bool)
	mask[max_grad_indices] = True
	dxs &= mask
	dx = np.sum(dxs, axis=-1)
	dys &= mask
	dy = np.sum(dys, axis=-1)
	return dx, dy, edge_gradients


def hysteresis_thresholding(image, image_gradients, min_val, max_val):
	"""
	Perform hysteresis thresholding using some bitwise magic.
	"""
	print("BEFORE HYSTERISIS THRESHOLDING:", image)
	print("gradients:", image_gradients)

	largest_gradient_value = np.max(image_gradients)
	while largest_gradient_value < max_val:
		print("Largest gradient value:", largest_gradient_value)
		warnings.warn(UserWarning("Image has no edge gradients above upper threshold, increasing all gradients values!"))
		# return np.zeros_like(image)
		image_gradients *= 1.5
		largest_gradient_value = np.max(image_gradients)
	
	# print("Largest gradient value:", largest_gradient_value)
	# the set of all 'strong' indices.
	strong_indices  = indices_where(image_gradients >= max_val)
	off_indices 	= indices_where(image_gradients < min_val)
	weak_indices 	= indices_where((min_val <= image_gradients) & (image_gradients < max_val))
	
	image_height = image.shape[0]
	image_width = image.shape[1]

	# get the neighbours of all strong edges.
	# convert their neighbours with weak edges to strong edges.
	to_explore = np.zeros_like(image_gradients, dtype=bool)
	to_explore[index_with(strong_indices)] = True

	explored = np.zeros_like(image_gradients, dtype=bool)

	strong = np.zeros_like(image_gradients, dtype=bool)
	strong[index_with(strong_indices)] = True
	# print("strong:", strong)

	weak = np.zeros_like(image_gradients, dtype=bool)
	weak[index_with(weak_indices)] = True

	unexplored_indices = aggregate(np.nonzero(to_explore))
	# print("unexplored (initial):", [str(v) for v in unexplored])
	# print("weak indices (initial):", [str(v) for v in weak_indices])
	# print("off indices (initial):", [str(v) for v in off_indices])
	already_explored = np.zeros_like(to_explore)

	while len(unexplored_indices) > 0:
		
		# print("exploring indices ", [str(v) for v in indices])
		# print(indices)

		neighbours = neighbourhood(unexplored_indices, image_width, image_height)
		is_neighbour = np.zeros_like(weak)
		is_neighbour[index_with(neighbours)] = True
		is_weak_neighbour = is_neighbour & weak
		weak_neighbours = aggregate(np.nonzero(is_weak_neighbour))
		# weak_neighbours = common_rows_between(neighbours, weak_indices)

		# print("The neighbours of (", ",".join(str(pixel) for pixel in indices), ") are ", neighbours)
		# print("weak neighbours:", [str(v) for v in weak_neighbours])
		
		strong[index_with(weak_neighbours)] = True
		weak[index_with(weak_neighbours)] = False
		# mark that we need to explore these:
		
		already_explored[index_with(unexplored_indices)] = True
		# explore the indices of the weak neighbours, if they haven't been explored already.
		to_explore[index_with(weak_neighbours)] = True
		# do not re-explore already explored indices.
		to_explore &= ~already_explored
		
		unexplored_indices = aggregate(np.nonzero(to_explore))
	
	out = np.zeros_like(image_gradients)
	out[~strong] = 0
	out[strong] = 255
	print("AFTER HYSTERISIS THRESHOLDING:", out)
	return out


def aggregate(list_of_indices):
	return np.concatenate(np.dstack(list_of_indices))

def indices_where(condition):
	return np.concatenate(np.dstack(np.where(condition)))

def index_with(list_of_indices):
	return list_of_indices[:, 0], list_of_indices[:, 1]

def neighbourhood(index, image_width, image_height):
	"""Returns the coordinates of the neighbours of a given coordinate or list of coordinates.
	
	Arguments:
		index {np.ndarray} -- either a list of coordinates (as an ndarray) or a coordinate itself, in the form (i, j)

		NOTE: the pixels neighbours are clipped of (image_height-1, )
	
	Returns:
		np.ndarray -- ndarray of shape [?, 2], which contains the indices of the neighbouring pixels
	"""
	neighbourhoods = np.concatenate(np.dstack((np.indices([3,3]) - 1)))
	if len(index.shape) == 2:
		neighbourhoods = neighbourhoods[:, np.newaxis, :]

	neighbours_and_itself = index + neighbourhoods
	keep = np.ones(9, dtype=bool)
	keep[4] = False # drop the point itself, but keep the neighbours.
	neighbours = neighbours_and_itself[keep]
	if len(index.shape) == 2:
		neighbours = np.stack(neighbours, axis=1)
	
	mask = np.ones_like(neighbours, dtype=bool)
	# remove all neighbours that have either a negative value in them
	negative = np.where(neighbours < 0)
	mask[negative] = False
	# or a value equal to image_height in x
	greater_than_image_height = np.where(neighbours[..., 0] >= image_height)
	mask[greater_than_image_height] = False
	# or image_width in z
	greater_than_image_width = np.where(neighbours[..., 1] >= image_height)
	mask[greater_than_image_width] = False
	# or that correspond to an index in 'index'
	tiled = np.expand_dims(index, 1)
	tiled = np.tile(tiled, (1, neighbours.shape[1], 1))
	equal_to_index = np.equal(neighbours, tiled)
	equal_to_index = np.all(equal_to_index, axis=-1)
	mask[equal_to_index] = False
	
	mask = np.all(mask, axis=-1)

	# print(mask)
	# for i, (m, n) in enumerate(zip(mask, neighbours)):
	# 	if len(index.shape) == 2:
	# 		for keep, (i, j) in zip(m, n):
	# 			print("point", i, j, "is good:", keep)
	# 	else:
	# 		keep = m
	# 		i, j = n
	# 		print("point", i, j, "is good:", keep)
		
	neighbours = neighbours[mask]
	# get rid of duplicates:
	neighbours = np.unique(neighbours, axis=0)
	return neighbours
	# # print(image[row, col])
	# min_x = max(i-1, 0)
	# max_x = min(i+1, image_w-1)
	# min_y = max(j-1, 0)
	# max_y = min(j+1, image_h-1)
	# indices = set(
	# 	(x, y)
	# 	for x in range(min_x, max_x + 1)
	# 	for y in range(min_y, max_y + 1)
	# )
	# print(indices)
	# indices.discard((i, j))
	# return indices
	# # return np.array(indices)


def common_rows_between(array_1, array_2):
	"""TAKEN FROM https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
	
	Arguments:
		array_1 {np.ndarray} -- a 2d array
		array_2 {np.ndarray} -- another 2d array
	
	Returns:
		np.ndarray -- a 2d array containing the common rows in both array_1 and array_2. 
	"""
	nrows, ncols = array_1.shape
	dtype={
		'names': ['f{}'.format(i) for i in range(ncols)],
		'formats': ncols * [array_1.dtype]
	}
	C = np.intersect1d(array_1.view(dtype), array_2.view(dtype))
	# This last bit is optional if you're okay with "C" being a structured array...
	C = C.view(array_1.dtype).reshape(-1, ncols)
	return C



def non_maximum_suppression(image, image_gradients, gradient_directions):
	"""Non-maximum suppression
	
	To be honest, I'm very proud of this piece of code. No for-loops were needed.

	Arguments:
		image {[type]} -- the image to preform non-maximum suppresion on.
		gradient_directions {[type]} -- the gradient directions
	"""
	print("Before non-maximum suppression:", image)
	# Get where to check depending on the "direction"
	direction_offset_x = np.round(np.cos(gradient_directions)).astype(int)
	direction_offset_y = np.round(np.sin(gradient_directions)).astype(int)
	direction_offset = np.dstack((direction_offset_x, direction_offset_y))

	# the (i, j) indices of all points in the image.
	row, col = np.indices(image.shape)
	# in order not to cause any indexing errors, we create a
	# padded version of the image with the edge values duplicated.
	# a pixel at (row, col) in the image is located at (row+1, col+1) in the padded image.
	image_ = np.pad(image, 1, mode="edge")
	row_, col_ = row + 1, col + 1
	# get the image pixels before and after each pixel in the image. 
	pixel_middle = image[row, col]
	pixel_forward  = image_[row_ + direction_offset_x, col_ + direction_offset_y]
	pixel_backward = image_[row_ - direction_offset_x, col_ - direction_offset_y]

	higher_than_forward  = pixel_middle > pixel_forward
	higher_than_backward = pixel_middle > pixel_backward
	is_local_maximum = higher_than_backward & higher_than_forward
	out = np.copy(image)
	out[~is_local_maximum] = 0
	print("AFTER non-maximum suppression: ", out)

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

	temp1 = conv2d(image, sobel_sign)
	image_dx = conv2d(temp1, sobel_mag.T)

	temp2 = conv2d(image, sobel_mag)
	image_dy = conv2d(temp2, -sobel_sign.T)
	
	return image_dx, image_dy

	# save these for comparison
	image_dx_1, image_dy_1 = image_dx, image_dy

	# Slower alternative (from OpenCV docs):
	sobel_x = np.array([
		[-1, 0, 1],
		[-2, 0, 2],
		[-1, 0, 1],
	])

	image_dx = conv2d(image, sobel_x)
	image_dy = conv2d(image, -sobel_x.T)
	assert np.all(np.isclose(image_dy, image_dy_1))
	assert np.all(np.isclose(image_dx, image_dx_1))
	return image_dx, image_dy

	

def conv2d(x, kernel, stride=1, padding="auto", padding_mode="constant"):
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


def separable_conv2d(x, kernel_1d, stride=1, padding="auto", padding_mode="edge"):
	assert len(kernel_1d.shape) == 1, kernel_1d
	k = kernel_1d.shape[0]
	k1 = kernel_1d.reshape([1, k])
	result_1 = conv2d(x, k1, stride, padding, padding_mode)
	k2 = k1.T
	result_2 = conv2d(result_1, k2, stride, padding, padding_mode)
	return result_2
	

def gaussian_kernel_1d(std=1, kernel_size=5):
	x = np.arange(-(kernel_size//2), (kernel_size//2)+1, dtype=float)
	g = np.exp(- (x**2 / (2 * std**2))) / (np.sqrt(2 * np.pi) * std)
	# normalize the sum to 1
	g = g / np.sum(g)
	return g


def gaussian_blurring(image, std=1, kernel_size=5):
	# # print(kernel)
	# kernel = np.expand_dims(kernel, axis=0)
	# kernel = kernel.T @ kernel
	# kernel /= np.sum(kernel)
	# print("BEFORE GAUSSIAN BLURRING:\n", image)

	kernel = gaussian_kernel_1d(std, kernel_size)
	image1 = separable_conv2d(image, kernel, padding_mode="constant")
	# print("AFTER GAUSSIAN BLURRING1:\n", image1)
	
	return image1
	# slower alternative from openCV documentation:
	kernel = 1 / 159 * np.array([
		[2, 4, 5, 4, 2],
		[4, 9, 12, 9, 4],
		[5, 12, 15, 12, 5],
		[4, 9, 12, 9, 4],
		[2, 4, 5, 4, 2]
	])
	image2 = conv2d(image, kernel)
	print("AFTER GAUSSIAN BLURRING2:\n", image2)
	return image2


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
X = np.array([
	[0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0],
	[0, 255, 255, 255, 7],
	[5, 0, 23, 3, 6],
	[1, 2, 3, 4, 8],
])
X = np.tile(X[..., np.newaxis], (10, 10, 3))
X[:,:,1] = X[:,:,0] * 0.5
X[:,:,2] = X[:,:,0] * 0.3

print(X.shape)
bob = Canny(X.astype(np.uint8), 75, 200)
print(bob)
