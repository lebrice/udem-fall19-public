
from collections import OrderedDict
from math import floor

from numpy.testing.utils import assert_almost_equal
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import multivariate_normal, entropy

from duckietown_msgs.msg import SegmentList
import duckietown_utils as dtu

from duckietown_utils.parameters import Configurable

import numpy as np

from .lane_filter_interface import LaneFilterInterface

from scipy.stats import multivariate_normal
from scipy.ndimage.filters import gaussian_filter
from math import floor, sqrt
import copy

import rospy

def wrap(angle):
    """Takes in any angle (in radians), and expresses it inside the [-np.pi, np.pi] range.
    
    Arguments:
        angle {float} -- An angle in radians
    
    Returns:
        float -- The angle projected within the range (-np.pi, np.pi]
    """
    two_pi = 2 * np.pi
    angle %= two_pi
    if angle < 0:
        angle += two_pi
    # angle is now inside [0, 2pi]
    if angle > np.pi:
        angle -= two_pi
    assert (- np.pi) < angle <= np.pi
    return angle

def R(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])




def cos_and_sin(x):
    return np.array([
        np.cos(x),
        np.sin(x),
    ])


def TR(theta_rad, Ax, Ay):
    return np.array([
        [np.cos(theta_rad), -np.sin(theta_rad), Ax],
        [np.sin(theta_rad),  np.cos(theta_rad), Ay],
        [0, 0, 1],
    ])


class LaneFitlerParticle(Configurable, LaneFilterInterface):

    class Particle():
        def __init__(self, d, phi, config):
            self.d = d
            self.phi = phi
            self.weight = 1

            self.d_max = config['d_max']
            self.d_min = config['d_min']
            self.phi_max = config['phi_max']
            self.phi_min = config['phi_min']
            self.sigma_d = config['sigma_d']
            self.sigma_phi = config['sigma_phi']

        def predict(self, dt, v, w):
            """update d and phi depending on dt, v and w
            
            Arguments:
                dt {float} -- [the time delay elapsed since the last prediction.]
                v {[type]} -- [the commanded linear (tangential) velocity of the robot]
                w {[type]} -- [the commanded angular velocity of the robot]
            
             lets's consider the displacement with respect to the previous position:
                  ^(x)
            (y)   |
             <----|
            
            
            The displacement is (y, x), with:
            - X is the displacement in direction parallel to the lane
            - Y is the displacement in the direction perpendicular to the lane.
            
            """
            # DT bug.
            dt *= 0.15

            if w == 0:
                # going in a straight line.
                displacement = dt * v * cos_and_sin(self.phi)
                # d is displaced in 'y', and phi stays the same.
                self.d += displacement[1]
            else:
                # calculate the displacement due to omega
                angle_performed_along_arc = dt * w
                tangential_velocity = v
                angular_velocity = w
                radius_of_curvature = np.abs(tangential_velocity / angular_velocity)

                # get the projections of the distance traveled relative to current heading phi.
                dx = radius_of_curvature * np.sin(angle_performed_along_arc)
                dy = radius_of_curvature * (1 - np.cos(angle_performed_along_arc))
                
                
                # This displacement is in the frame with heading phi though.
                # Therefore we need to correct for that as well by rotating it by -phi.
                rotation = R(-self.phi)
                displacement = np.matmul(np.array([dy, dx]), rotation)
                
                # The orientation changed since we moved along the arc.
                # It can easily be shown that the change in phi due to the arc motion
                # is equal to the angle performed along the arc.
                change_in_phi = angle_performed_along_arc
                self.phi += change_in_phi
                self.d += displacement[1]

            self.d += np.random.normal(0, self.sigma_d)
            self.phi += np.random.normal(0, self.sigma_phi)

            # We clip d and phi to stay within the desired maximal range
            self.d = np.clip(self.d, self.d_min, self.d_max)
            self.phi = np.clip(self.phi, self.phi_min, self.phi_max)

        def update(self, ds, phis):
            # Change weight depending on the likelihood of ds and phis from the measurements

            ########
            # Your code here
            # TODO: How can you estimate the likelihood of the measurements given this particular particle?
            # Here, the d and phi values for a given segment can be recovered using function self.process(segment).
            # Suggestion: remember how it was done in the histogram filter.
            # Maybe you can compute a distance from your particle to each measured pair of (d,phi)
            # and compute a score based on the quantity of pairs that is not further than a given threshold? Other ideas are welcome too!
            ########
            average_distance = 0
            if len(ds) > 0 and len(phis) > 0: 
                normalized_d_distances = ((ds - self.d) / (self.d_max - self.d_min)) ** 2
                normalized_phi_distances = ((phis - self.phi) / (self.phi_max - self.phi_min)) ** 2
                # this is the normalized average distance (between 0 and 1)
                average_distance = np.mean(normalized_d_distances + normalized_phi_distances) / 2
            self.weight = 1 - average_distance

        def perturb(self, dd, dphi):
            self.d += dd
            self.phi += dphi
            # We clip d and phi to stay within the desired maximal range
            self.d = np.clip(self.d, self.d_min, self.d_max)
            self.phi = np.clip(self.phi, self.phi_min, self.phi_max)


    def __init__(self):
        # Parameters
        self.nb_particles = 500

        ## Initialization 
        self.mean_d_0 = 0       # Expected value of d at initialization
        self.mean_phi_0 = 0     # Expected value of phi at initialization
        self.sigma_d_0 = 0.1    # Standard deviation of d at initialization
        self.sigma_phi_0 = 0.1  # Standard deviation of phi at initialization

        ## Prediction step noise
        self.sigma_d = 0.010   # 0.001
        self.sigma_phi = 0.010 # 0.002

        ## Roughening
        self.rough_d =  0.010 # 0.001
        self.rough_phi = 0.020 # 0.002

        ## Environment parameters
        self.linewidth_white = 0.05
        self.linewidth_yellow = 0.025
        self.lanewidth = 0.23

        ## Limits
        self.d_max = 0.3        
        self.d_min = -0.15
        self.phi_min = -1.5
        self.phi_max = 1.5
        self.range_est = 0.33   # Maximum distance for a segment to be considered
        self.delta_d = 0.02     # Maximum error on d for a segment to be considered as an inlier
        self.delta_phi = 0.1    # Maximum error on phi for a segment to be considered as an inlier
        self.min_max = 0.1      # Minimum maximal weight 

        # Attributes
        self.particles = []
        self.particle_config = {
            'd_max': self.d_max,
            'd_min': self.d_min,
            'phi_max': self.phi_max,
            'phi_min': self.phi_min,  
            'sigma_d': self.sigma_d,      
            'sigma_phi': self.sigma_phi,   
        }

        # Initialization
        self.initialize()

    def initialize(self):
        # Initialize the particle filter
        initial_particles = []

        # Parameters to be passed
        config = self.particle_config

        ds = np.random.uniform(self.d_min, self.d_max, size=self.nb_particles)
        phis = np.random.uniform(self.phi_min, self.phi_max, size=self.nb_particles)
        for i in range(self.nb_particles):
            d = 0
            phi = 0

        ########
        # Your code here
        # TODO: Initialize the particle set using a given distribution.
        # You can use the initialization parameters.
        # Would sampling from a Gaussian distribution be a good idea?
        # Could you also want to sample from an uniform distribution, in order to be able to recover from an initial state that is far from the Gaussian center?
        ########
            initial_particles.append(self.Particle(ds[i], phis[i], config))
        self.particles = initial_particles

    def predict(self, dt, v, w):
        # Prediction step for the particle filter
        for particle in self.particles:
            particle.predict(dt, v, w)

    def update(self, segment_list):
        # Measurement update state for the particle filter
        segmentArray = self.prepareSegments(segment_list)
        self.updateWeights(segment_list)
        self.resample()
        self.roughen()

    def updateWeights(self, segment_list):
        ds = []
        phis = []
        # Compute the ds and phis from the segments
        for segment in segment_list:
            d, phi, _ = self.process(segment)
            ds.append(d)
            phis.append(phi)
        # Update the particle weights
        for particle in self.particles:
            particle.update(ds, phis) 
        
    def resample(self):
        # Sample a new set of particles
        weights = np.array([p.weight for p in self.particles], dtype=np.float32)
        # for some reason, we sometiems start with negative weights? 
        weights -= np.min(weights)
        if np.sum(weights) == 0:
            # sometimes weights is all zeroes, for some reason.
            # In this case, I guess we set the weights as all equal:
            weights = np.ones_like(weights, dtype=np.float32)            
        probabilities = weights / np.sum(weights)
        
        assert np.all(probabilities >= 0), probabilities
        assert np.isclose(np.sum(probabilities), 1.0), probabilities
        # print("probabilities:", probabilities)
        indices_to_keep = np.random.choice(self.nb_particles, size=self.nb_particles, p=probabilities)

        indices_kept = set()
        new_particles = []
        for index in indices_to_keep:
            particle = self.particles[index]

            if index in indices_kept:
                # if we already have this particle, make a new copy:
                particle = self.Particle(particle.d, particle.phi, self.particle_config)
                new_particles.append(particle)
            else:
                # if we haven't sampled this particle already, just add it:
                new_particles.append(particle)
                indices_kept.add(index)
        self.particles = new_particles

    def roughen(self):
        # Roughen the particles set to avoid sample impoverishment
        for particle in self.particles:
            dd = np.random.normal(loc = 0.0, scale = self.rough_d)
            dphi = np.random.normal(loc = 0.0, scale = self.rough_phi)
            particle.perturb(dd, dphi)

    def getEstimate(self):
        # Get the estimate of d and phi
        d = 0
        phi = 0
        ########
        # Your code here
        # TODO: What is the best way to give an estimate of the state given a set of particles?
        # Would it be a random sampling? An average? Putting them in bins and chosing the most populated one?
        ########

        # Here we perform a weighted average of the particles.
        ds = np.array([p.d for p in self.particles])
        phis = np.array([p.phi for p in self.particles])
        weights = np.array([p.weight for p in self.particles])
        weights /= np.sum(weights)

        d = np.dot(weights, ds)
        phi = np.dot(weights, phis)

        # print("Estimate: d:", d, " phi:", phi)
        return [d, phi]

    
    def isInLane(self):
        # Test to know if the bot is in the lane
        ########
        # Your code here
        # TODO: Remember the way the histogram filter was determining if the robot is or is not in the lane.
        # What was the idea behind it? How could this be applied to a particle filter?
        ########
        return  max(p.weight for p in self.particles) > self.min_max

### Other functions - no modification needed ###
    def process(self, segment):
        # Returns   d_i the distance from the middle of the lane
        #           phi_i the angle from the lane direction
        #           l_i the distance from the bot in perpendicular projection
        p1 = np.array([segment.points[0].x, segment.points[0].y])
        p2 = np.array([segment.points[1].x, segment.points[1].y])
        t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)

        n_hat = np.array([-t_hat[1], t_hat[0]])
        d1 = np.inner(n_hat, p1)
        d2 = np.inner(n_hat, p2)
        l1 = np.inner(t_hat, p1)
        l2 = np.inner(t_hat, p2)
        if (l1 < 0):
            l1 = -l1
        if (l2 < 0):
            l2 = -l2

        l_i = (l1 + l2) / 2
        d_i = (d1 + d2) / 2
        phi_i = np.arcsin(t_hat[1])
        if segment.color == segment.WHITE:  # right lane is white
            if(p1[0] > p2[0]):  # right edge of white lane
                d_i = d_i - self.linewidth_white
            else:  # left edge of white lane
                d_i = - d_i
                phi_i = -phi_i
            d_i = d_i - self.lanewidth / 2

        elif segment.color == segment.YELLOW:  # left lane is yellow
            if (p2[0] > p1[0]):  # left edge of yellow lane
                d_i = d_i - self.linewidth_yellow
                phi_i = -phi_i
            else:  # right edge of white lane
                d_i = -d_i
            d_i = self.lanewidth / 2 - d_i

        return d_i, phi_i, l_i

    def getStatus(self):
        return LaneFilterInterface.GOOD

    def get_inlier_segments(self, segments, d, phi):
        inlier_segments = []
        for segment in segments:
            d_s, phi_s, l = self.process(segment)
            if abs(d_s - d) < self.delta_d and abs(phi_s - phi)<self.delta_phi:
                inlier_segments.append(segment)
        return inlier_segments

    def prepareSegments(self, segments):
    # Filter out segments
        segmentsRangeArray = []
        for segment in segments:
            # Remove RED segments 
            if segment.color != segment.WHITE and segment.color != segment.YELLOW:
                continue
            # Remove segments that are behind us
            elif segment.points[0].x < 0 or segment.points[1].x < 0:
                continue
            # Remove segments that are too far from the Duckiebot
            elif self.getSegmentDistance(segment) > self.range_est:
                continue
            else:
                segmentsRangeArray.append(segment)
        return segmentsRangeArray

    def getSegmentDistance(self, segment):
        x_c = (segment.points[0].x + segment.points[1].x) / 2
        y_c = (segment.points[0].y + segment.points[1].y) / 2
        return sqrt(x_c**2 + y_c**2)

    def getBeliefArray(self):
        # Returns the representation of the belief as an array (for visualization purposes)
        ds, phis = np.mgrid[self.d_min:self.d_max:self.delta_d, self.phi_min:self.phi_max:self.delta_phi]
        beliefArray = np.zeros(ds.shape)
        # Image of the particle set
        for particle in self.particles:
            d_i = particle.d
            phi_i = particle.phi
            # if the vote lands outside of the histogram discard it
            if d_i > self.d_max or d_i < self.d_min or phi_i < self.phi_min or phi_i > self.phi_max:
                continue
            
            i = int(floor((d_i - self.d_min) / self.delta_d))
            j = int(floor((phi_i - self.phi_min) / self.delta_phi))

            if i == beliefArray.shape[0]:
                i -= 1
            if j == beliefArray.shape[1]:
                j -= 1

            beliefArray[i, j] = beliefArray[i, j] + 1  

        if np.linalg.norm(beliefArray) == 0:
            return beliefArray
        beliefArray = beliefArray / np.sum(beliefArray)
        return beliefArray
 

class LaneFilterHistogram(Configurable, LaneFilterInterface):
    #"""LaneFilterHistogram"""

    def __init__(self, configuration):
        param_names = [
            'mean_d_0',
            'mean_phi_0',
            'sigma_d_0',
            'sigma_phi_0',
            'delta_d',
            'delta_phi',
            'd_max',
            'd_min',
            'phi_max',
            'phi_min',
            'cov_v',
            'linewidth_white',
            'linewidth_yellow',
            'lanewidth',
            'min_max',
            'sigma_d_mask',
            'sigma_phi_mask',
            'curvature_res',
            'range_min',
            'range_est',
            'range_max',
            'curvature_right',
            'curvature_left',
        ]

        configuration = copy.deepcopy(configuration)
        Configurable.__init__(self, param_names, configuration)

        self.d, self.phi = np.mgrid[self.d_min:self.d_max:self.delta_d,
                                    self.phi_min:self.phi_max:self.delta_phi]
        self.d_pcolor, self.phi_pcolor = \
            np.mgrid[self.d_min:(self.d_max + self.delta_d):self.delta_d,
                     self.phi_min:(self.phi_max + self.delta_phi):self.delta_phi]

        self.beliefArray = np.empty(self.d.shape)
        self.range_arr = np.zeros(1)
        self.mean_0 = [self.mean_d_0, self.mean_phi_0]
        self.cov_0 = [[self.sigma_d_0, 0], [0, self.sigma_phi_0]]
        self.cov_mask = [self.sigma_d_mask, self.sigma_phi_mask]

        self.d_med_arr = []
        self.phi_med_arr = []
        self.median_filter_size = 5

        self.initialize()
        self.updateRangeArray()

        # Additional variables
        self.red_to_white = False
        self.use_yellow = True
        self.range_est_min = 0
        self.filtered_segments = []

    def getStatus(self):
        return LaneFilterInterface.GOOD

    def predict(self, dt, v, w):
        delta_t = dt
        d_t = self.d + v * delta_t * np.sin(self.phi)
        phi_t = self.phi + w * delta_t
        p_belief = np.zeros(self.beliefArray.shape)

        for i in range(self.beliefArray.shape[0]):
            for j in range(self.beliefArray.shape[1]):
                if self.beliefArray[i, j] > 0:
                    if d_t[i, j] > self.d_max or d_t[i, j] < self.d_min or phi_t[i, j] < self.phi_min or phi_t[i, j] > self.phi_max:
                        continue
                    i_new = int(
                        floor((d_t[i, j] - self.d_min) / self.delta_d))
                    j_new = int(
                        floor((phi_t[i, j] - self.phi_min) / self.delta_phi))
                    p_belief[i_new, j_new] += self.beliefArray[i, j]
        s_belief = np.zeros(self.beliefArray.shape)
        gaussian_filter(p_belief, self.cov_mask,
                        output=s_belief, mode='constant')

        if np.sum(s_belief) == 0:
            return
        self.beliefArray = s_belief / np.sum(s_belief)

    # prepare the segments for the creation of the belief arrays
    def prepareSegments(self, segments):
        segmentsRangeArray = []
        self.filtered_segments = []
        for segment in segments:
            # Optional transform from RED to WHITE
            if self.red_to_white and segment.color == segment.RED:
                segment.color = segment.WHITE

            # Optional filtering out YELLOW
            if not self.use_yellow and segment.color == segment.YELLOW: continue

            # we don't care about RED ones for now
            if segment.color != segment.WHITE and segment.color != segment.YELLOW:
                continue
            # filter out any segments that are behind us
            if segment.points[0].x < 0 or segment.points[1].x < 0:
                continue

            self.filtered_segments.append(segment)
            # only consider points in a certain range from the Duckiebot for the position estimation
            point_range = self.getSegmentDistance(segment)
            if point_range < self.range_est and point_range > self.range_est_min:
                segmentsRangeArray.append(segment)

        return segmentsRangeArray

    def updateRangeArray(self):
        self.beliefArray = np.empty(self.d.shape)
        self.initialize()

    # generate the belief arrays
    def update(self, segments):
        # prepare the segments for each belief array
        segmentsRangeArray = self.prepareSegments(segments)
        # generate all belief arrays
        measurement_likelihood = self.generate_measurement_likelihood(segmentsRangeArray)

        if measurement_likelihood is not None:
            self.beliefArray = np.multiply(self.beliefArray, measurement_likelihood)
            if np.sum(self.beliefArray) == 0:
                self.beliefArray = measurement_likelihood
            else:
                self.beliefArray = self.beliefArray / np.sum(self.beliefArray)

    def generate_measurement_likelihood(self, segments):
        # initialize measurement likelihood to all zeros
        measurement_likelihood = np.zeros(self.d.shape)
        for segment in segments:
            d_i, phi_i, l_i =  self.generateVote(segment)
            # if the vote lands outside of the histogram discard it
            if d_i > self.d_max or d_i < self.d_min or phi_i < self.phi_min or phi_i > self.phi_max:
                continue
            i = int(floor((d_i - self.d_min) / self.delta_d))
            j = int(floor((phi_i - self.phi_min) / self.delta_phi))
            measurement_likelihood[i, j] += 1
        if np.linalg.norm(measurement_likelihood) == 0:
            return None
        measurement_likelihood = measurement_likelihood / np.sum(measurement_likelihood)
        return measurement_likelihood

    # get the maximal values d_max and phi_max from the belief array. 
    def getEstimate(self):
        maxids = np.unravel_index(self.beliefArray.argmax(), self.beliefArray.shape)
        d_max = self.d_min + (maxids[0] + 0.5) * self.delta_d
        phi_max = self.phi_min + (maxids[1] + 0.5) * self.delta_phi
        return [d_max, phi_max]

    def get_estimate(self):
        d, phi = self.getEstimate()
        res = OrderedDict()
        res['d'] = d
        res['phi'] = phi
        return res

    # return the maximal value of the beliefArray
    def getMax(self):
        return self.beliefArray.max()

    def isInLane(self):
        return self.getMax() > self.min_max

    def initialize(self):
        pos = np.empty(self.d.shape + (2,))
        pos[:, :, 0] = self.d
        pos[:, :, 1] = self.phi
        self.cov_0
        RV = multivariate_normal(self.mean_0, self.cov_0)
        self.beliefArray = RV.pdf(pos)

    # generate a vote for one segment
    def generateVote(self, segment):
        p1 = np.array([segment.points[0].x, segment.points[0].y])
        p2 = np.array([segment.points[1].x, segment.points[1].y])
        t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)
        n_hat = np.array([-t_hat[1], t_hat[0]])
        d1 = np.inner(n_hat, p1)
        d2 = np.inner(n_hat, p2)
        l1 = np.inner(t_hat, p1)
        l2 = np.inner(t_hat, p2)
        if (l1 < 0):
            l1 = -l1
        if (l2 < 0):
            l2 = -l2
        l_i = (l1 + l2) / 2
        d_i = (d1 + d2) / 2
        phi_i = np.arcsin(t_hat[1])
        if segment.color == segment.WHITE:  # right lane is white
            if(p1[0] > p2[0]):  # right edge of white lane
                d_i = d_i - self.linewidth_white
            else:  # left edge of white lane
                d_i = - d_i
                phi_i = -phi_i
            d_i = d_i - self.lanewidth / 2
        elif segment.color == segment.YELLOW:  # left lane is yellow
            if (p2[0] > p1[0]):  # left edge of yellow lane
                d_i = d_i - self.linewidth_yellow
                phi_i = -phi_i
            else:  # right edge of white lane
                d_i = -d_i
            d_i = self.lanewidth / 2 - d_i
        return d_i, phi_i, l_i

    def get_inlier_segments(self, segments, d_max, phi_max):
        inlier_segments = []
        for segment in segments:
            d_s, phi_s, l = self.generateVote(segment)
            if abs(d_s - d_max) < self.delta_d and abs(phi_s - phi_max)<self.delta_phi:
                inlier_segments.append(segment)
        return inlier_segments

    # get the distance from the center of the Duckiebot to the center point of a segment
    def getSegmentDistance(self, segment):
        x_c = (segment.points[0].x + segment.points[1].x) / 2
        y_c = (segment.points[0].y + segment.points[1].y) / 2
        return sqrt(x_c**2 + y_c**2)
