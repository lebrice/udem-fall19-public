import cv2
import sw_06_cv_functions as sw06
import duckietown_utils as dtu

import numpy as np
import rospy
from .line_detector_interface import Detections, LineDetectorInterface
import time
        

class LineDetectorHSV(dtu.Configurable, LineDetectorInterface):
    """ LineDetectorHSV """

    def __init__(self, configuration):
        # Images to be processed
        self.bgr = np.empty(0)
        self.hsv = np.empty(0)
        self.edges = np.empty(0)

        param_names = [
            'hsv_white1',
            'hsv_white2',
            'hsv_yellow1',
            'hsv_yellow2',
            'hsv_red1',
            'hsv_red2',
            'hsv_red3',
            'hsv_red4',
            'dilation_kernel_size',
            'canny_thresholds',
            'hough_threshold',
            'hough_min_line_length',
            'hough_max_line_gap',
        ]

        dtu.Configurable.__init__(self, param_names, configuration)

    def _colorFilter(self, color):
        # threshold colors in HSV space
        if color == 'white':
            bw = sw06.inRange(self.hsv, self.hsv_white1, self.hsv_white2)
        elif color == 'yellow':
            bw = sw06.inRange(self.hsv, self.hsv_yellow1, self.hsv_yellow2)
        elif color == 'red':
            bw1 = sw06.inRange(self.hsv, self.hsv_red1, self.hsv_red2)
            bw2 = sw06.inRange(self.hsv, self.hsv_red3, self.hsv_red4)
            bw = sw06.bitwise_or(bw1, bw2)
        else:
            raise Exception('Error: Undefined color strings...')

        # binary dilation
        kernel = sw06.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (self.dilation_kernel_size, self.dilation_kernel_size))
        bw = sw06.dilate(bw, kernel)

        # refine edge for certain color
        edge_color = sw06.bitwise_and(bw, self.edges)

        return bw, edge_color

    def _findEdge(self, gray):
        start = time.time()
        edges = sw06.Canny(gray, self.canny_thresholds[0], self.canny_thresholds[1], apertureSize=3)
        end = time.time()
        print("CANNY TOOK", end - start, "seconds.") 
        return edges

    def _HoughLine(self, edge):
        lines = sw06.HoughLinesP(edge, 1, np.pi / 180, self.hough_threshold, np.empty(1),
                                self.hough_min_line_length, self.hough_max_line_gap)
        if lines is not None:
            lines = np.array(lines[:, 0])
        else:
            lines = []
        return lines

    def _checkBounds(self, val, bound):
        val[val < 0] = 0
        val[val >= bound] = bound - 1
        return val

    def _correctPixelOrdering(self, lines, normals):
        flag = ((lines[:, 2] - lines[:, 0]) * normals[:, 1] - (lines[:, 3] - lines[:, 1]) * normals[:, 0]) > 0
        for i in range(len(lines)):
            if flag[i]:
                x1, y1, x2, y2 = lines[i, :]
                lines[i, :] = [x2, y2, x1, y1]

    def _findNormal(self, bw, lines):
        normals = []
        centers = []
        if len(lines) > 0:
            length = np.sum((lines[:, 0:2] - lines[:, 2:4]) ** 2, axis=1, keepdims=True) ** 0.5
            dx = 1.* (lines[:, 3:4] - lines[:, 1:2]) / length
            dy = 1.* (lines[:, 0:1] - lines[:, 2:3]) / length

            centers = np.hstack([(lines[:, 0:1] + lines[:, 2:3]) / 2, (lines[:, 1:2] + lines[:, 3:4]) / 2])
            x3 = (centers[:, 0:1] - 3.*dx).astype('int')
            y3 = (centers[:, 1:2] - 3.*dy).astype('int')
            x4 = (centers[:, 0:1] + 3.*dx).astype('int')
            y4 = (centers[:, 1:2] + 3.*dy).astype('int')
            x3 = self._checkBounds(x3, bw.shape[1])
            y3 = self._checkBounds(y3, bw.shape[0])
            x4 = self._checkBounds(x4, bw.shape[1])
            y4 = self._checkBounds(y4, bw.shape[0])
            flag_signs = (np.logical_and(bw[y3, x3] > 0, bw[y4, x4] == 0)).astype('int') * 2 - 1
            normals = np.hstack([dx, dy]) * flag_signs

            self._correctPixelOrdering(lines, normals)
        return centers, normals

    def detectLines(self, color):
        with dtu.timeit_clock('_colorFilter'):
            bw, edge_color = self._colorFilter(color)
        with dtu.timeit_clock('_HoughLine'):
            lines = self._HoughLine(edge_color)
        with dtu.timeit_clock('_findNormal'):
            centers, normals = self._findNormal(bw, lines)
        return Detections(lines=lines, normals=normals, area=bw, centers=centers)

    def setImage(self, bgr):

        with dtu.timeit_clock('np.copy'):
            self.bgr = np.copy(bgr)
        with dtu.timeit_clock('cvtColor COLOR_BGR2HSV'):
            self.hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        with dtu.timeit_clock('_findEdge'):
            self.edges = self._findEdge(self.bgr)

    def getImage(self):
        return self.bgr
