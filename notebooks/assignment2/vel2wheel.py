import numpy as np




def vel2wheel(v, omega, wheel_dist, wheel_rad):
    
    gain = 1
    trim = 0
    
    # Maximal speed
    if v > 0.5:
        v = 0.5
    elif v < -0.5:
        v = -0.5
    
    
##### Fill the code here:
    # linear velocities:
    L = wheel_dist / 2
    wheel_circumference = 2 * np.pi * wheel_rad
    turning_left = omega > 0
    moving_forward = v > 0
    if omega == 0:
        # going straight.
        left_rate = right_rate = v / wheel_circumference
    elif v == 0:
        # turning in-place
        faster_wheel = abs(omega) * L / wheel_circumference
        slower_wheel = - faster_wheel

        left_rate = slower_wheel if turning_left else faster_wheel
        right_rate = faster_wheel if turning_left else slower_wheel
    elif v > 0:
        # trace a curve
        d = abs(v / omega) # radius of curvature
        slower_wheel = abs(omega) * (d - L) / wheel_circumference
        faster_wheel = abs(omega) * (d + L) / wheel_circumference

        left_rate = slower_wheel if turning_left else faster_wheel
        right_rate = faster_wheel if turning_left else slower_wheel
    else:
        d = abs(v / omega) # radius of curvature
        slower_wheel = - abs(omega) * (d - L) / wheel_circumference
        faster_wheel = - abs(omega) * (d + L) / wheel_circumference

        left_rate = faster_wheel if turning_left else slower_wheel
        right_rate = slower_wheel if turning_left else faster_wheel


####


    
    
    return left_rate, right_rate