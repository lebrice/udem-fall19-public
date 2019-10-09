import random
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import *

def wrap(angle: float) -> float:
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

def to_3d(point_2d):
    return np.asarray([point_2d[0], 0, point_2d[1]])

class RRT_planner:
    """
    Rapid Random Tree (RRT) planner
    """

    class Node:
        """
        Node for RRT
        TODO: move this out of the RRT_planner class.
        """
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.parent = None
            self.path_x = []
            self.path_y = []

        def to(self, other: "Node") -> Tuple[float, float]:
            """Returns the distance and angle between this node and `other`.
            
            Arguments:
                other {Node} -- Destination node
            
            Returns:
                Tuple[float, float] -- The distance and angle in radians between `self` and `other`.
            """
            dx = other.x - self.x
            dy = other.y - self.y
            d = math.sqrt(dx ** 2 + dy ** 2)
            angle = math.atan2(dy, dx)
            return d, angle

        def straight_path_to(self: "Node", end_node: "Node", step_size: float) -> Tuple[List[float], List[float]]:
            """Returns the coordinates of the straight line path between `self` and `end_node`.
            
            Arguments:
                start_node {Node} -- The starting node
                end_node {Node} -- The end node
                step_size {float} -- The step size along the path
            
            Returns:
                (path_x, path_y): Tuple[List[float], List[float]] -- the X and Y coordinates of the points along the path.
            """
            dist, angle = self.to(end_node)

            step_x = np.cos(angle) * step_size
            path_x = np.arange(self.x, end_node.x, step_x).tolist()
            path_x.append(end_node.x)

            step_y = np.sin(angle) * step_size
            path_y = np.arange(self.y, end_node.y, step_y).tolist()
            path_y.append(end_node.y)

            return path_x, path_y

        def collides_with(self, obstacleList: List[Tuple[float, float, float]]) -> bool:
            """Crude collision detection. Returns wether the path to get from the parent node to this node collides with any of the given obstacles. 
            
            Arguments:
                obstacleList {List[Tuple[float, float, float]]} -- List of (x, y, size) for each obstacle
            
            Returns:
                bool -- Wether the path to this node collided with any of the obstacles.
            """
            # Returns True if collision between a node and at least one obstacle in obstacleList
            for (ob_x, ob_y, size) in obstacleList:
                dist_x = (ob_x - x for x in self.path_x)
                dist_y = (ob_y - y for y in self.path_y)
                distance_magnitudes = (dx * dx + dy * dy for (dx, dy) in zip(dist_x, dist_y))
                min_distance_magnitude = (size / 2) ** 2
                collided = (d <= min_distance_magnitude for d in distance_magnitudes)
                if any(collided):
                    return True
            return False
        
        def lineage(self):
            """yields all the parents of this node up to the root."""
            node = self
            while node.parent is not None:
                yield node.parent
                node = node.parent

    def __init__(self, start, goal, list_obstacles, rand_area,
                 max_branch_length=0.5, path_res=0.1, goal_sample_rate=5, max_iter=1000):
        """
        Parameters:
            start: Start Position [x,y]
            goal: Goal Position [x,y]
            list_obstacles: obstacle Positions [[x,y,size],...]
            rand_area: random Sampling Area [x_min, x_max, y_min, y_max]
            max_branch_length : maximal extension for one step
            path_res : resolution of obstacle checking in the path
            goal_sample_rate : percentage of samples that are artifically set to the goal
            max_iter: maximal number of iterations

        """
        self.start_node = self.Node(start[0], start[1])
        self.end_node = self.Node(goal[0], goal[1])
        self.rand_area = rand_area
        self.max_branch_length = max_branch_length
        self.path_res = path_res
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.list_obstacles = list_obstacles
        self.list_nodes = []

    def plan(self, show_anim=True):
        """
        Returns the path from goal to start.
        show_anim: flag for show_anim on or off
        """

        self.list_nodes = [self.start_node]
        for it in range(self.max_iter):
            # new_node = self.Node(0, 0)
            ####### 
            # Objective: create a valid new_node according to the RRT algorithm and append it to "self.list_nodes"
            # You can call any of the functions defined lower, or add your own.

            # YOUR CODE HERE
            random_node = self.get_random_node()
            closest_node = self.get_node_closest_to(random_node)
            
            # new_node = self.extend_towards(closest_node, random_node)
            new_node = self.extend(closest_node, random_node)

            collision = new_node.collides_with(self.list_obstacles)
            if collision:
                continue
            else:
                self.list_nodes.append(new_node)
            
            #######

            if show_anim and it % 5 == 0:
                self.draw_graph(random_node)

            if self.distance_to_goal(new_node.x, new_node.y) <= self.max_branch_length:
                print("Reached goal")
                return self.make_final_path(len(self.list_nodes) - 1)

            if show_anim and it % 5:
                self.draw_graph(random_node)

        return None  # cannot find path

    def extend_towards(self, or_node: Node, dest_node: Node) -> Node:
        """Given an origin node and a destination node, this function adds a node in the
        direction of dest_node from origin_node, at most self.path_res away.
        If the distance between the two nodes is less than
        `self.max_branch_length`, joins the two nodes, and returns the dest_node.
        
        Arguments:
            or_node {Node} -- Origin Node
            dest_node {Node} -- Destination node
            step_length {float} -- the maximum length of each step along the path.
        
        Returns:
            Node -- Returns this node in the middle, or dest_node if the
            dest_node can be reached in one step.
        """
        dist, angle = or_node.to(dest_node)
        if dist <= self.max_branch_length:
            # the dest_node can be reached, therefore we will simply join the
            # two nodes, update the path of the dest_node, and return the
            # destination node.
            new_node = self.Node(
                x = dest_node.x,
                y = dest_node.y,
            )
        else:
            # we can't reach the dist_node, so we make a new node in the middle, as close as possible to dest_node.
            new_node = self.Node(
                x = or_node.x + np.cos(angle) * self.max_branch_length,
                y = or_node.y + np.sin(angle) * self.max_branch_length,
            )

        path_x, path_y = or_node.straight_path_to(new_node, self.path_res)
        new_node.path_x = path_x
        new_node.path_y = path_y
        new_node.parent = or_node
        return new_node
    
    def extend(self, or_node, dest_node):
        """
        Returns a new node going from or_node in the direction of dest_node with maximal distance of max_branch_length. New node path goes from parent to new node with steps of path_res.
        """
        new_node = self.Node(or_node.x, or_node.y)
        dist, angle = self.compute_dist_ang(new_node, dest_node)
        dist_extension = self.max_branch_length
        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if dist_extension > dist:
            dist_extension = dist

        n_expand = math.floor(dist_extension / self.path_res)

        for _ in range(n_expand):
            new_node.x += self.path_res * math.cos(angle)
            new_node.y += self.path_res * math.sin(angle)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        dist, _ = self.compute_dist_ang(new_node, dest_node)
        if dist <= self.path_res:
            new_node.x = dest_node.x
            new_node.y = dest_node.y
            new_node.path_x[-1] = dest_node.x
            new_node.path_y[-1] = dest_node.y

        new_node.parent = or_node

        return new_node

    def draw_graph(self, rnd=None, final_path = False):
        # Draw a graph of the path
        plt.clf()
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.list_nodes:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for (ob_x, ob_y, size) in self.list_obstacles:
            plt.plot(ob_x, ob_y, "ok", ms=30 * size)

        if final_path:
            plt.plot([x for (x, y) in final_path], [y for (x, y) in final_path], '-r')
        plt.plot(self.start_node.x, self.start_node.y, "xr")
        plt.plot(self.end_node.x, self.end_node.y, "xr")
        plt.axis([0, 7, 0, 5])
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.pause(0.01)

    def make_final_path(self, goal_ind):
        # Returns the path as the list of all the node positions in node_list, from end to start
        path = [[self.end_node.x, self.end_node.y]]
        node = self.list_nodes[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def distance_to_goal(self, x, y):
        dx = x - self.end_node.x
        dy = y - self.end_node.y
        return math.sqrt(dx ** 2 + dy ** 2)

    def get_random_node(self):
        # Returns a random node within random area, with the goal being sampled with a probability of goal_sample_rate %
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(random.uniform(self.rand_area[0], self.rand_area[1]),
                            random.uniform(self.rand_area[2], self.rand_area[3]))
        else:  # goal point sampling
            rnd = self.Node(self.end_node.x, self.end_node.y)
        return rnd

    @staticmethod
    def collision(node, obstacleList):
        # Returns True if collision between a node and at least one obstacle in obstacleList
        for (ob_x, ob_y, size) in obstacleList:
            list_dist_x = [ob_x - x for x in node.path_x]
            list_dist_y = [ob_y - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(list_dist_x, list_dist_y)]

            if min(d_list) <= (size/2) ** 2:
                return True 

        return False     

    @staticmethod
    def compute_dist_ang(or_node, dest_node):
        # Computes distance and angle between origin and destination nodes
        dx = dest_node.x - or_node.x
        dy = dest_node.y - or_node.y
        d = math.sqrt(dx ** 2 + dy ** 2)
        angle = math.atan2(dy, dx)
        return d, angle

    @staticmethod
    def get_closest_node_id(list_nodes, random_node):
        # Returns index of node in list_nodes that is the closest to random_node
        dist_list = [(node.x - random_node.x) ** 2 + (node.y - random_node.y)
                 ** 2 for node in list_nodes]
        min_id = dist_list.index(min(dist_list))

        return min_id
    
    def get_node_closest_to(self, target: Node) -> Node:
        # Returns index of node in list_nodes that is the closest to random_node
        distances = {
            node: ((node.x - target.x) ** 2 + (node.y - target.y) ** 2)
            for node in self.list_nodes
        }
        return min(distances, key=distances.get)    


class RTT_Path_Follower:
    """
    Follows a path given by RRT_Planner
    """
    def __init__(self, path, local_env):
        self.path = path
        self.path_3d = [to_3d(p) for p in self.path]
        self.forward_path = list(reversed(self.path_3d))
        self.env = local_env
        self.lookup_distance = 0.5
        self.max_v = 0.5
        import collections
        self.points = collections.defaultdict(list)
        print("INIT WAS CALLED")
        self.omega = None

        self.target_index = 1
        self.state = "TURNING"

    def next_action(self):
        # Current position and angle
        cur_pos_x = self.env.cur_pos[0]
        cur_pos_y = self.env.cur_pos[2]
        cur_angle = self.env.cur_angle
        
        v = 0.
        omega = 0.
        
        #######
        #
        # YOUR CODE HERE: change v and omega so that the Duckiebot keeps on following the path
        #
        #######
        def c(angle):
            """
            Taken from https://piazza.com/class/k06skksxrwp18e?cid=90, 
            """
            return np.asarray([
                [np.cos(angle), 0, -np.sin(angle)],
                [0, -1, 0],
                [-np.sin(angle), 0, -np.cos(angle)],    
            ])
        
        target_pos = self.forward_path[self.target_index]

        to_target = target_pos - self.env.cur_pos
        # print("to_target", to_target)
        to_target_relative = c(self.env.cur_angle) @ to_target
        # print("to_target_relative", to_target_relative)
        
        mag = to_target_relative.dot(to_target_relative)
        if mag < 0.05:
            self.target_index += 1
            print("Heading to next point on the path: ", self.forward_path[self.target_index])
            if self.target_index == len(self.forward_path):
                print("REACHED THE GOAL!")
                return 0, 0

        alpha = np.arctan2(to_target_relative[2], to_target_relative[0])
        alpha = wrap(alpha)
        # print("ALPHA:", alpha)
        omega, v = 0, 0
        if abs(alpha) > 0.01:
            if self.state != "TURNING":
                # print("TURNING")
                self.state = "TURNING"
            omega = 1 * alpha
            v = 0
        else:
            if self.state != "STRAIGHT":
                # print("STRAIGHT")
                self.state = "STRAIGHT"
            omega = 0
            v = 0.5 * mag
            v = min(v, 0.05)
            v = max(v, 2.0)
        
        ## useful for debugging
        self.points["mag"].append(mag)
        self.points["alpha"].append(alpha)
        self.points["omega"].append(omega)
        self.points["v"].append(v)

        return v, omega



        mag = to_target.dot(to_target)
        print("current pos:", self.env.cur_pos)
        print("Current angle (calculated):", wrap(np.arctan2(self.env.cur_pos[2], self.env.cur_pos[0])))
        print("Current angle (actual):", wrap(self.env.cur_angle))

        to_target_angle = np.arctan2(to_target[2], to_target[0])
        


        v, omega = self.pure_pursuit(
            self.env,
            self.env.cur_pos,
            self.env.cur_angle,
            follow_dist=0.50,
        )
        return v, omega
    
    def closest_curve_point(self, pos, angle):
        """Gives the curve point closest to the agent, and the tangent at that point
        Arguments:
            self {[type]} -- [description]
            angle {[type]} -- [description]
        """
        closest_node = None
        closest_node_index = None
        min_distance = None

        pos = np.asarray(pos)

        for i, path_node in enumerate(self.path_3d):
            delta = path_node - pos
            dist = delta.dot(delta)
            if min_distance is None or dist < min_distance:
                min_distance = dist
                closest_node = path_node
                closest_node_index = i
        
        if closest_node_index == 0:
            tangent = np.asarray(closest_node) - np.asarray(self.path_3d[1])
        else:
            tangent = np.asarray(self.path_3d[closest_node_index-1]) - np.asarray(self.path_3d[closest_node_index])
        tangent /= np.linalg.norm(tangent)

        return closest_node, tangent


    def pure_pursuit(self, env, pos, angle, follow_dist=0.25):
        # Return the angular velocity in order to control the Duckiebot using a pure pursuit algorithm.
        # Parameters:
        #     env: Duckietown simulator
        #     pos: global position of the Duckiebot
        #     angle: global angle of the Duckiebot
        # Outputs:
        #     v: linear veloicy in m/s.
        #     omega: angular velocity, in rad/sec. Right is negative, left is positive.
        
        # Find the curve point closest to the agent, and the tangent at that point
        closest_point, closest_tangent = self.closest_curve_point(pos, angle)
    
        def find_curve_point_better(follow_dist, step_size=0.05) -> float:
            """
            We take baby steps along the curve, and we return the first point along
            that curve that is at least L distance away from the robot.
            """
            def distance_mag(a, b):
                diff = a-b
                return (diff).dot(diff)
            
            # Start off from closest_point
            curve_point, tangent = closest_point, closest_tangent
            distance_from_robot_mag = distance_mag(pos, curve_point)
            follow_dist_mag = follow_dist**2

            max_iterations = 10
            iterations = 0
            while distance_from_robot_mag < follow_dist_mag and iterations < max_iterations:
                # take a small step forward from the current curve_point
                forward_projection = curve_point + tangent * step_size

                # Find the curve point closest to the forward projection, and the tangent at that point
                next_curve_point, next_tangent = self.closest_curve_point(forward_projection, angle)
                if curve_point is None:
                    break
                curve_point, tangent = next_curve_point, next_tangent
                # recalculate the distance
                distance_from_robot_mag = distance_mag(curve_point, pos)

                iterations += 1

            return curve_point

        # curve_point = find_curve_point_old()
        curve_point = find_curve_point_better(follow_dist=self.lookup_distance, step_size=0.05)

        robot_to_curve_point = curve_point - pos
        robot_to_curve_point_angle = np.arctan2(-robot_to_curve_point[2], robot_to_curve_point[0])
        alpha = robot_to_curve_point_angle - angle
        alpha = wrap(alpha)
        sin_alpha = np.sin(alpha)
        
        # need to tune this.
        self.k_l = 0.25
        self.k_v = 0.25
        
        v = self.max_v
        # v = self.k_v / abs(alpha)
        # v = min(v, self.max_v)
        
        lookup_distance = self.lookup_distance
        # lookup_distance = self.k_l * v
        p = 0.8
        
        # # without smoothing.
        # self.lookup_distance = lookup_distance
        
        # With smoothing (prevents jerky motion)
        self.lookup_distance = p * self.lookup_distance + (1-p) * lookup_distance
        self.points["lookup_distance"].append(self.lookup_distance)

        omega = sin_alpha / self.lookup_distance

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
        import math
        num_plots = len(self.points.keys())
        
        fig = plt.figure(figsize=(15, 12))
        for i, (title, list_of_points) in enumerate(self.points.items()):
            plt.subplot(2, math.ceil(num_plots / 2), i+1)
            data_np = np.asarray(list_of_points)
            plt.plot(data_np)
            plt.title(title)
        # plt.legend(self.points.keys(), loc="upper right")
        plt.suptitle("Fabrice Normandin")
    