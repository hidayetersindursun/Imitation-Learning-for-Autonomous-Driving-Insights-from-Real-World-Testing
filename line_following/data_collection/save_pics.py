
 #!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import os 
import datetime

folder_path = "/home/nvidia/marc_ws/scripts/line_following/data_collection/imgs"  #where images saved

# Instantiate CvBridge
bridge = CvBridge()

def image_callback(msg):
    # print("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg 
        img = cv2.resize(cv2_img, (320, 180))
        # cv2.imshow("frame",img)
        # cv2.waitKey(3)
      
        date_and_time = datetime.datetime.now()
        file_name = "{}.jpg".format(date_and_time)

        cv2.imwrite(os.path.join(folder_path, file_name), cv2_img)
def main():
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "/zed/zed_node/rgb_raw/image_raw_color"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()
    