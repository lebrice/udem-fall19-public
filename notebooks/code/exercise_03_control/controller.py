import numpy as np


class Controller():
    def __init__(self, k_p_dist=10.0, k_d_dist=0., k_p_angle=10.0, k_d_angle=0):
        self.gain = 2.0
        pass
        self.last_dist: float = None
        self.last_angle: float = None
        
        self.k_p_angle = k_p_angle
        self.k_d_angle = k_d_angle
        self.k_p_dist = k_p_dist
        self.k_d_dist = k_d_dist
#         print(self.k_p_dist, self.k_d_dist, self.k_p_angle, self.k_d_angle)

    def angle_control_commands(self, dist, angle):
        # Return the angular velocity in order to control the Duckiebot so that it follows the lane.
        # Parameters:
        #     dist: distance from the center of the lane. Left is negative, right is positive.
        #     angle: angle from the lane direction, in rad. Left is negative, right is positive.
        # Outputs:
        #     omega: angular velocity, in rad/sec. Right is negative, left is positive.
        
        omega = 0. 
        
        #######
        #
        # MODIFY ANGULAR VELOCITY
        #
        # YOUR CODE HERE
        #
        #######
        if self.last_angle is None:
            self.last_angle = angle
        if self.last_dist is None:
            self.last_dist = dist
        
        e_angle = angle
        d_angle = (self.last_angle - angle)

        e_dist = dist
        d_dist = (self.last_dist - dist)
        
        self.last_angle = angle
        self.last_dist = dist
        
        omega += self.k_p_angle * e_angle
        omega += self.k_d_angle * d_angle
        
        omega += self.k_p_dist * e_dist
        omega += self.k_d_dist * d_dist
        return  omega

    def pure_pursuit(self, env, pos, angle, follow_dist=0.25):
        # Return the angular velocity in order to control the Duckiebot using a pure pursuit algorithm.
        # Parameters:
        #     env: Duckietown simulator
        #     pos: global position of the Duckiebot
        #     angle: global angle of the Duckiebot
        # Outputs:
        #     v: linear veloicy in m/s.
        #     omega: angular velocity, in rad/sec. Right is negative, left is positive.
        
        
        closest_curve_point = env.unwrapped.closest_curve_point
        
        # Find the curve point closest to the agent, and the tangent at that point
        closest_point, closest_tangent = closest_curve_point(pos, angle)

        iterations = 0
        
        lookup_distance = follow_dist
        multipler = 0.5
        curve_point = None
        
        while iterations < 10:            
            ########
            #
            #TODO 1: Modify follow_point so that it is a function of closest_point, closest_tangent, and lookup_distance
            #
            ########
            follow_point = closest_point
            
            curve_point, _ = closest_curve_point(follow_point, angle)

            # If we have a valid point on the curve, stop
            if curve_point is not None:
                break

            iterations += 1
            lookup_distance *= multiplier
        ########
        #
        #TODO 2: Modify omega
        #
        ########
        omega = 0.
        v = 0.5

        return v, omega