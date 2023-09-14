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

    Controller(error)
    A PID controller for generating the steering angle command.
    
    pipeline()
        Publish the calculated steering angle and speed
       
    """
    def __init__(self):
        # Required for converting the ROS images to OpenCV images
        self.bridge = CvBridge()
        
        # Car steering angle
        self.angle = 0   # in radians
        
        # Car speed
        self.speed = 0.6  #m/s
        
        self.rate = rospy.Rate(20)
        
        # Debug mode. See what the car is seeing
        self.debug = True
        
        # Our ZED Image
        self.image = None
        
        rospack = rospkg.RosPack()
        
        self.finished = False   

        self.img = None

        #Controller Parameters
        self.Kp=0.1
        self.Ki=0
        self.Kd=0.2
        
        self.dt = 1
        self.I = 0
        self.last_error = 0
        # Subscriber and Publisher
        rospy.Subscriber("/zed/zed_node/rgb_raw/image_raw_color", Image, self.zed_callback, queue_size=1)
        self.pub = rospy.Publisher('/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=1)
        self.image_pub = rospy.Publisher('/processed_image_output', Image, queue_size=1)


    def image_process(self,img):
        img = cv2.resize(img, (640, 360))
        self.img = img[200:360,:]
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        
        lower_red = np.array([154, 60, 0])  #s 50 
        upper_red = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        
        masked_img = cv2.bitwise_and(self.img, self.img, mask = mask)
        kernel = np.ones((6,6),np.uint8)
        closing = cv2.morphologyEx(masked_img, cv2.MORPH_CLOSE,kernel)
        blurred = cv2.GaussianBlur(closing, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        _,self.result = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(self.result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) >0:
            largest_contour = max(contours, key=cv2.contourArea)
            #if cv2.contourArea(largest_contour)>50:
            moment = cv2.moments(largest_contour) 
            object_x = int(moment["m10"] / moment["m00"])
            object_y = int(moment["m01"] / moment["m00"])
            object_center = [object_x,object_y]            
            #x, y, w, h = cv2.boundingRect(largest_contour)
            #cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #object_center = [x+w/2,y+h/2]
            cv2.circle(self.img, (object_x,object_y),1,(0,0,255),3)
            cv2.drawContours(self.img,largest_contour,-1, (0, 255, 0), 1)
            frame_center = [320,180]
            cv2.line(self.img,(frame_center[0],frame_center[1]),(object_center[0],object_center[1]),(255,0,0),1)
            error = frame_center[0]-object_center[0]
            self.angle = self.Controller(error)
            #print(str(self.angle) + "  degrees: "+ str(self.angle*180/pi))
            

    def Controller(self,error):
        P = self.Kp*error
        self.I = self.I + self.Ki*(error)*self.dt
        D = self.Kd*(error-self.last_error)/self.dt
        self.last_error = error
        u=P+self.I+D

        u=u*pi/180  # convert from degree to radian 
        return u
        
    def zed_callback(self, data):
        
        # Convert the image to OpenCV format
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        self.image_process(self.image)


    def pipeline(self):
        if not self.img is None:
            image_msg = self.bridge.cv2_to_imgmsg(self.img, "bgr8")
            self.image_pub.publish(image_msg)
            
            # cv2.imshow("Image", self.img)
            # cv2.imshow("Masked", self.result)
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

