#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import rospkg
import cv2
import rospy
import sys, os
from math import atan2, degrees, pi 
from std_msgs.msg import Float64
from sensor_msgs.msg import Image, Joy
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge, CvBridgeError

rospy.init_node('lane_detection')

class BaseClass(object):
    """
    Base class for autonomous driving. 
    This class reads the pre-trained model, fetches the image from the ZED camera rostopic, and predicts the next steering angle.
    Speed is constant. The car will not move until a successful inference result is produced. 
    
    Methods
    -------
    
    zed_callback(data)
        ZED rostopic callback. It is fired when the new image is available.
        
    image_process(img)
        Process the captured image to detect a specific color (red) object and calculate a predicted steering angle
        based on its position in the image.
    
    pipeline()
        Publish the predicted steering angle and speed
       
    """
    def __init__(self):
        # Required for converting the ROS images to OpenCV images
        self.bridge = CvBridge()
        
        # Car steering angle
        self.angle = 0   # in radians
        
        # Car speed
        self.speed = 0.65  #m/s
        
        self.rate = rospy.Rate(20)
        
        # Debug mode. See what the car is seeing
        self.debug = True
        
        # Our ZED Image
        self.image = None
        
        rospack = rospkg.RosPack()
        
        self.finished = False   

        self.img = None

        # Subscriber and Publisher
        rospy.Subscriber("/zed/zed_node/rgb_raw/image_raw_color", Image, self.zed_callback, queue_size=1)
        self.pub = rospy.Publisher('/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=1)
        self.image_pub = rospy.Publisher('/processed_image_output', Image, queue_size=1)


    def image_process(self,img):
        img = cv2.resize(img, (640, 360))
        self.img = img[180:360,:]
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        
        lower_red = np.array([154, 50, 0])
        upper_red = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        
        masked_img = cv2.bitwise_and(self.img, self.img, mask = mask)
        blurred = cv2.GaussianBlur(masked_img, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        _, self.result = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(self.result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours is not None:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
            center = [x+w/2,y+h/2]
            offset = 50
            if center[0]<320-offset:
                theta = degrees(atan2(w,h))
            elif center[0]>=320+offset:
                theta = -degrees(atan2(w,h))
            else:
                theta = 0

        print(theta)

        self.angle = 0.25*(theta)*(pi/180)



    def zed_callback(self, data):
        
        # Convert the image to OpenCV format
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.image_process(self.image)


    def pipeline(self):
        if not self.img is None:
            image_msg = self.bridge.cv2_to_imgmsg(self.img, "bgr8")
            self.image_pub.publish(image_msg)
            
            # cv2.imshow("Image", self.img)
            # if cv2.waitKey(3) & 0xFF == ord('q'):
            #     pass
            
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = self.angle
        msg.drive.speed = self.speed
        self.pub.publish(msg)
        self.rate.sleep()

        
drive = BaseClass()
        
if __name__ == '__main__':
    while not rospy.is_shutdown():
        drive.pipeline()
    drive.finished = True
    rospy.spin()
    sys.exit(0)

