import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os 
import datetime

bridge = CvBridge()
cv2_img = None  # Define cv2_img as a global variable
steering_angle = 0.0
speed = 0.0

def drive_callback(data):
    global steering_angle, speed
    # Extract the steering_angle and speed from the received message
    steering_angle = data.drive.steering_angle
    speed = data.drive.speed
    return steering_angle, speed

def image_callback(msg):
    global cv2_img  # Access the global cv2_img variable

    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # You can perform additional image processing or saving here if needed

    except CvBridgeError as e:
        print(e)

def main():
    global cv2_img, steering_angle, speed # Access the global cv2_img variable
    rospy.init_node('ackermann_listener', anonymous=True)
    pub = rospy.Publisher('/tobik', AckermannDriveStamped, queue_size=1)
    image_pub = rospy.Publisher('/tobik_image', Image, queue_size=1)
    # Define the topic you want to subscribe to
    topic_name = '/ackermann_cmd_mux/input/navigation'
    
    # Subscribe to the topic and specify the callback function
    rospy.Subscriber(topic_name, AckermannDriveStamped, drive_callback, queue_size=10)
    image_topic = "/zed/zed_node/rgb_raw/image_raw_color"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    # Keep the node alive

    while not rospy.is_shutdown():
        # Create a custom message and publish it
        if cv2_img is not None:
            image_msg = bridge.cv2_to_imgmsg(cv2_img, encoding="bgr8")
            image_pub.publish(image_msg)

        msg = AckermannDriveStamped()
        msg.drive.steering_angle = steering_angle
        msg.drive.speed = speed
        pub.publish(msg)
        rate=rospy.Rate(20)
        rate.sleep()

if __name__ == '__main__':
    main()
