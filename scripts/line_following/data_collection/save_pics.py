
 #!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os 
import datetime

date_and_time = datetime.datetime.now()
folder_name = "{}".format(date_and_time)

directory = "/home/nvidia/marc_ws/scripts/line_following/data_collection/imgs/"

# Create the folder
folder_path = os.path.join(directory, folder_name)
os.mkdir(folder_path)

# Instantiate CvBridge
bridge = CvBridge()

def image_callback(msg):

    try:

        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        # Save your OpenCV2 image as a jpg 
        img = cv2.resize(cv2_img, (320, 180))
        # cv2.imshow("frame",img)
        # cv2.waitKey(3)
    
        date_and_time = datetime.datetime.now()
        file_name = "{}.jpg".format(date_and_time)
        cv2.imwrite(os.path.join(folder_path, file_name), cv2_img)
        
        
    except CvBridgeError as e:
        print(e)

def main():

    rospy.init_node('image_listener',anonymous=True)
    
    # Define your image topic
    image_topic = "/zed/zed_node/rgb_raw/image_raw_color"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback,queue_size=10)
    
    #rate = rospy.Rate(30)
    #while not rospy.is_shutdown():
        # Sleep to achieve the desired rate
    #    rate.sleep()

    # Spin until ctrl + c
    
    rospy.spin()

if __name__ == '__main__':
    main()
    