# !pip3 install gnupg
import roslib
import rospy
import jupyros as jr
import rosbag
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
import matplotlib.pyplot as plt
import numpy as np



bridge = CvBridge()
with rosbag.Bag('images.bag', 'r') as image_bag:
    for topic, image_message, t in image_bag.read_messages():
        arr = np.asarray(image_message.data)
        
        np_arr = np.frombuffer(image_message.data, np.uint8)
#         print(np_arr.shape)
#         break
#         print(np_arr)
#         break
#         image = bridge.imgmsg_to_cv2(image_message, "bgr8")
# #         image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#         cv2.imshow("image", image)
# #         plt.show()
        break
    
