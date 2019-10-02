import numpy as np

def wrap(angle: float) -> float:
    """Takes in any angle (in radians), and expresses it inside the [-np.pi, np.pi] range.
    """
    two_pi = 2 * np.pi
    angle %= two_pi
    if angle < 0:
        angle += two_pi
    # angle is now inside [0, 2pi]
    if angle > np.pi:
        angle -= two_pi
    return angle


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
        
        self.k_v = 0.5
        import collections
        self.points = collections.defaultdict(list)

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
        multiplier = 0.5
        curve_point = None
        
        while iterations < 10:            
            ########
            #
            #TODO 1: Modify follow_point so that it is a function of closest_point, closest_tangent, and lookup_distance
            #
            ########
            def find_follow_point(closest_point, closest_tangent, lookup_distance):
                follow_point = closest_point + lookup_distance * closest_tangent
                # print("closest_point:", closest_point , "closesest_tangent", closest_tangent, "lookup_distance", lookup_distance)
                return follow_point
                
            follow_point = find_follow_point(closest_point, closest_tangent, lookup_distance)
            
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
        print("curve point:", curve_point, "pos:", pos)
        robot_to_curve_point = curve_point - pos
        # print("local:", curve_point_local)
        # sin_alpha = sin(theta_g - angle)
        # sin(A - B) = sinAcosB-cosAsinB
        # hypothenuse = np.sqrt(robot_to_curve_point[0] ** 2 + robot_to_curve_point[2] ** 2)
        # sin_theta_g = robot_to_curve_point[0] / hypothenuse
        # cos_theta_g = robot_to_curve_point[2] / hypothenuse
        # sin_alpha_1 = sin_theta_g * np.cos(angle) - cos_theta_g * np.sin(angle)
        
        robot_to_curve_point = curve_point - pos
        robot_to_curve_point_angle = np.arctan2(-robot_to_curve_point[2], robot_to_curve_point[0])
        alpha = robot_to_curve_point_angle - angle
        alpha = wrap(alpha)

        sin_alpha = np.sin(alpha)
        
        self.points["alpha"].append(alpha)
        self.points["sin_alpha"].append(sin_alpha)
        # need to tune this.
        self.k_l = 0.5
        self.k_v = 1.0
        
        max_speed = 0.5

        v = min(self.k_v / abs(alpha), max_speed)
        
        # v = abs(self.k_v / alpha)
        lookup_d = self.k_l * v

        omega = sin_alpha / lookup_d 

        print("v", v)
        print("omega", omega)
        print("alpha:", alpha)

        # cross-track error
        cross_track_error = sin_alpha * np.linalg.norm(robot_to_curve_point)
        self.points["cross_track_error"].append(cross_track_error)
        # angle error,
        angle_of_tangent = np.arctan2(-closest_tangent[2], closest_tangent[0])
        angle_error = angle_of_tangent - angle
        angle_error = wrap(angle_error)
        self.points["angle_error"].append(angle_error)
        # commanded linear velocity
        self.points["commanded_linear_velocity"].append(v)
        # commanded angular velocity 
        self.points["commanded_angular_velocity"].append(omega)


        return v, omega
    
    def plot(self):
        import matplotlib.pyplot as plt

        for i, (title, list_of_points) in enumerate(self.points.items()):
            data_np = np.asarray(list_of_points)
            plt.plot(data_np)

        plt.legend(self.points.keys(), loc="upper right")
        plt.title("Fabrice Normandin")


