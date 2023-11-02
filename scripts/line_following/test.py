#!/usr/bin/env python

import rospy
from ackermann_msgs.msg import AckermannDriveStamped

def callback(data):
    # Extract the steering_angle and speed from the received message
    steering_angle = data.drive.steering_angle
    speed = data.drive.speed
    timestamp = rospy.get_time()

    # Print the extracted values
    print "timestamp", timestamp
    print "Steering Angle: ", steering_angle
    print "Speed: ", speed

def listener():
    rospy.init_node('ackermann_listener', anonymous=True)
    
    # Define the topic you want to subscribe to
    topic_name = '/ackermann_cmd_mux/input/navigation'
    
    # Subscribe to the topic and specify the callback function
    rospy.Subscriber(topic_name, AckermannDriveStamped, callback, queue_size=1)

    # Keep the node alive
    rospy.spin()

if __name__ == '__main__':
    listener()
