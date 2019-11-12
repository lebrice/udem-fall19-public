import numpy as np

def R(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

def update(d, phi, dt, v, w):
    if w == 0:
        displacement = dt * v * np.array([
            np.sin(phi),
            np.cos(phi),
        ])
    else:
        # calculate the displacement due to omega
        angle_performed_along_circle = dt * w
        tangential_velocity = v
        angular_velocity = w
        radius_of_curvature = np.abs(tangential_velocity / angular_velocity)
        
        dx = radius_of_curvature * (1 - np.cos(angle_performed_along_circle))
        dy = radius_of_curvature * np.sin(angle_performed_along_circle)
        
        # This displacement is in the frame with heading phi though.
        # Therefore we need to correct for that as well.
        rotation = R(-phi)
        displacement = np.array([dx, dy]) @ rotation

    dx = displacement[0]
    dy = displacement[1]

    d += dx
    phi = np.arctan2(dx, dy)
    return d, phi

new_d, new_phi = update(
    d=1.0,
    phi=0,
    dt=1,
    v=1,
    w=0.01,
)
print(new_d, new_phi)