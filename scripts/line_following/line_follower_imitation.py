#!/usr/bin/python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import rospkg
from sensor_msgs.msg import Image, Joy
import sys, os
from math import atan2, degrees, pi
from std_msgs.msg import Float64
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge, CvBridgeError
from time import sleep
import time
import tensorflow as tf
 
#import myModel
#import dummyModel

rospy.init_node('line_follower_imitation_learning')


class BaseClass(object):

    def __init__(self):
        super(BaseClass,self).__init__()
        # Required for converting the ROS images to OpenCV images
        self.bridge = CvBridge()
        self.predictions = 0
        # Car steering angle
        self.angle = 0   # in radians

        # Car speed
        self.speed = 0.6  #m/s

        self.rate = rospy.Rate(25)

        # Debug mode. See what the car is seeing
        self.debug = True

        # Our ZED Image
        self.image = None
        self.input_image_for_model = None
        rospack = rospkg.RosPack()

        self.finished = False

        self.img = None

        #self.model = myModel.load_model()
        self.optimized_model = tf.saved_model.load("resnetModel_optimized_trt")

        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        self.frame_count = 0
        self.start_time = time.time()


        # Subscriber and Publisher
        rospy.Subscriber("/zed/zed_node/rgb_raw/image_raw_color", Image, self.zed_callback, queue_size=1)
        self.pub = rospy.Publisher('/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=1)


    
    def inference_image(self, img):
        

        if img is not None:

            #with tf.device('/gpu:0'):

                #img = img.reshape(1,224,224,3)
            self.predictions =  self.optimized_model(img)  #self.optimized_model(img)

            #with tf.device('/cpu:0'):
                #print(self.predictions)
    
    def zed_callback(self, data):

        img_np = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)   # ÇALIŞAN SATIR
            
        img_np = img_np[:, :, :3]  
        img = img_np[180:360,:]
        resized_image = cv2.resize(img, (224, 224))
        resized_image = cv2.cvtColor(resized_image,cv2.COLOR_BGR2RGB)
        resized_image = resized_image.reshape(1,224,224,3)
        self.input_image_for_model = tf.convert_to_tensor(resized_image, dtype=tf.float32)
        #self.input_image_for_model = tf.expand_dims(input_image_for_model, axis=0)  # Add the batch dimension

        #cv2.imshow("Image", resized_image)
        #if cv2.waitKey(3) & 0xFF == ord('q'):
        #    pass
            
       
    def pipeline(self):

        msg = AckermannDriveStamped()
        self.angle = self.predictions
        msg.drive.steering_angle = self.angle
        msg.drive.speed = self.speed
        self.pub.publish(msg)
        self.rate.sleep()



if __name__ == '__main__':
    drive = BaseClass()

    while not rospy.is_shutdown():
        with tf.device("/GPU:0"):
            drive.inference_image(drive.input_image_for_model)
        drive.pipeline()
        
        print(drive.predictions)
        
        # FPS counter 
        drive.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - drive.start_time
        if elapsed_time >= 1.0:
            fps = drive.frame_count / elapsed_time
            print(f"Inference FPS: {fps:.2f}")

            drive.frame_count = 0
            drive.start_time = current_time 

    rospy.spin()
    sys.exit(0)





