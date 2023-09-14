#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

def depth_callback(data):
    try:
        bridge = CvBridge()
        depth_image = bridge.imgmsg_to_cv2(data, desired_encoding="32FC1")
        print(depth_image[180,320])   # prints distance in meters of center pixel
    except Exception as e:
        print(e)

def main():
    rospy.init_node('zed_depth_node', anonymous=True)
    rospy.Subscriber('/zed/zed_node/depth/depth_registered', Image, depth_callback)
    rospy.spin()

if __name__ == '__main__':
    main()




