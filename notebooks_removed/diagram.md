# Diagram


## Nodes list

- ### Gymdt Node

    Description:

      - Seems to read from the GYM environment, sending images down the ROS pipeline and getting back actions to take on the environment.

    Inputs:

      - wheels_driver_node/wheels_cmd

    Outputs:

      - corrected_image/compressed
      - camera_info_topic

- ### Line Detector Node

    Description:

      - line detection?

    Inputs:

      - corrected_image/compressed
      - transform
      - switch
      - fsm_mode

    Outputs:

      - segment_list
      - image_with_lines

- ### Ground Projection Node

    Description:

      - ?

    Inputs:

      - lineseglist_in

    Outputs:

      - lineseglist_out

    Services:

      - estimate_homography
      - get_ground_coordinate
      - get_image_coordinate

- ### Lane Filter Node

    Description:

      - ?

    Inputs:

      - segment_list
      - car_cmd
      - change_params

    Outputs:

      - in_lane (boolean)
      - lane_pose (LanePose)
      - belief_img (Image)
      - ml_img (Image)
      - entropy (float)

- ### Lane Controller Node

    Description:

      - ?

    Inputs:

      - lane_pose (LanePose)
      - obstacle_avoidance_pose (LanePose)
      - obstacle_detected (bool)
      - intersection_navigation_pose (LanePose)
      - wheels_cmd_executed (bool)
      - actuator_limits
      - switch (fsm related, bool)
      - stop_line_reading
      - fsm_mode (fsm related, FSMState)

    Outputs:

      - car_cmd
      - actuator_limits_received (bool)
      - radius_limit (bool)
