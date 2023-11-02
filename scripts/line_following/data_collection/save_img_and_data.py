
 #!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os 
import datetime
import time 
import message_filters
import csv

date_and_time = datetime.datetime.now()
folder_name = "{}".format(date_and_time)

directory = "/home/nvidia/marc_ws/scripts/line_following/data_collection/imgs/"

# Create the folder
folder_path = os.path.join(directory, folder_name)
os.mkdir(folder_path)
print(folder_path)
#print("Note: If it doesn't work, check zed node and joy node.")
# Instantiate CvBridge
bridge = CvBridge()

# Create a CSV file and writer object
csv_file_path = os.path.join(folder_path, "axes_data.csv")
csv_file = open(csv_file_path, mode='w')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "steering angle", "speed"])  # Write the header


def joy_callback(joy_msg):

    try:
        print("Callback function called")
        steering = joy_msg.drive.steering_angle
        speed = joy_msg.drive.speed
        print "Steering Angle: ", steering
        print "Speed: ", speed
        timestamp = rospy.get_time()
        csv_writer.writerow([timestamp, steering, speed])

    except CvBridgeError as e:
        print(e)

def image_callback(msg):
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        timestamp = rospy.get_time()
        file_name = "{}.jpg".format(timestamp)
        cv2.imwrite(os.path.join(folder_path, file_name), cv2_img)

    except CvBridgeError as e:
        print(e)


def main():

    rospy.init_node('data_synchronizer',anonymous=True)

    # Define your image topic
    image_topic =  "/zed/zed_node/rgb_raw/image_raw_color"  #"/tobik_image"
    joy_topic =  "/ackermann_cmd_mux/input/navigation"  #"/tobik"
    # Set up your subscriber and define its callback
    image_sub = rospy.Subscriber(image_topic, Image, image_callback,queue_size=1)
    joy_sub = rospy.Subscriber(joy_topic, AckermannDriveStamped, joy_callback,queue_size=1)

    # Synchronize the messages based on their timestamps
    

    #rate = rospy.Rate(30)
    #while not rospy.is_shutdown():
        # Sleep to achieve the desired rate
        #rate.sleep()

    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()
    