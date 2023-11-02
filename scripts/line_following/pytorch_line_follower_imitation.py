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
import onnx
import torchvision.transforms as transforms
import onnxruntime as ort
import torch
import torch.nn as nn
from time import sleep
rospy.init_node('line_follower_imitation_learning')

class BaseClass(nn.Module):

    def __init__(self):
        super(BaseClass,self).__init__()
        # Required for converting the ROS images to OpenCV images
        self.bridge = CvBridge()

        # Car steering angle
        self.angle = 0   # in radians

        # Car speed
        self.speed = 0.6  #m/s

        self.rate = rospy.Rate(25)

        # Debug mode. See what the car is seeing
        self.debug = True

        # Our ZED Image
        self.image = None

        rospack = rospkg.RosPack()

        self.finished = False

        self.img = None


        
        #self.onnx_model_path = "resnet18-92acc.onnx"
        #self.ort_session = ort.InferenceSession(self.onnx_model_path)
        
        self.device = torch.device("cuda")
        self.model_path = "model3.pt"
        #self.model = torch.load(self.model_path)
        self.model = torch.load(self.model_path, map_location=self.device)


        # Subscriber and Publisher
        rospy.Subscriber("/zed/zed_node/rgb_raw/image_raw_color", Image, self.zed_callback, queue_size=1)
        self.pub = rospy.Publisher('/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=1)
        self.image_pub = rospy.Publisher('/processed_image_output', Image, queue_size=1)
        
        #self.model.eval()
        #print(self.model.input_size)
    
    def forward(self,img):
        out = self.model(img)
        return out



    def inference_image(self, img):
        # Convert ROS Image to OpenCV format
        #onnx_model =onnx.load(self.onnx_model_path)
        #input_name = self.ort_session.get_inputs()[0].name
        img = img.astype(np.float32) / 255.0
        img = img.transpose((2, 0, 1))
    
        img = img[np.newaxis,...]
        img_tensor = torch.from_numpy(img).to(self.device)
        
        print(img_tensor.dtype)
        
        output = self(img_tensor)
        print(output.shape)

        
        #ort_inputs = {input_name: img}
        #ort_outs = self.ort_session.run(None, ort_inputs)
        #print(ort_outs)

    
    def zed_callback(self, data):

        img_np = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        img_np = img_np[:, :, :3]  # Keep only the first three channels (R, G, B)
        #img = cv2.resize(img_np, (640, 360))
        img = img_np[180:360,:]
        resized_image = cv2.resize(img, (224, 224))
        
        #cv2.imshow("Image", img_np)
        #if cv2.waitKey(3) & 0xFF == ord('q'):
            #pass

        self.inference_image(resized_image)
        return resized_image
    def pipeline(self):

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