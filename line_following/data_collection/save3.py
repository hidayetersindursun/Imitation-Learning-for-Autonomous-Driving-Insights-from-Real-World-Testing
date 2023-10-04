from __future__ import print_function
import rospy
import cv2 as cv
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
import os
import datetime
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


IMAGE_TOPIC = "/zed/zed_node/rgb_raw/image_raw_color"
JOY_TOPIC = "/ackermann_cmd_mux/input/navigation"

# optional: use http://wiki.ros.org/message_filters#Time_Synchronizer to
# combine both the left and right camera
# e.g., frames = np.hstack([frame_left, frame_right])
class Recorder:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.imcb)
        self.angle = None
        self.speed = None
        self.joy_sub = rospy.Subscriber(JOY_TOPIC, AckermannDriveStamped, self.joycb)

        start = datetime.datetime.now()
        data_folder = "data_{}".format(start.strftime("%m_%d_%H_%M"))
        os.mkdir(data_folder)
        os.chdir(data_folder)
        self.csv_fn = "{}.csv".format(data_folder)

    def joycb(self, msg):
        self.angle = msg.drive.steering_angle
        self.speed = msg.drive.speed

    def imcb(self, data):
        if self.angle is None or self.speed is None:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        img = cv.resize(cv_image, (320, 240))
        now = datetime.datetime.now()
        img_fn = "{}.jpg".format(now.strftime("%m_%d_%H_%M_%S_%f")[:-3])

        with open(self.csv_fn, "a") as fh:
            fh.write("{},{},{}\n".format(img_fn, self.angle, self.speed))
        # if you prefer PNGs:
        # cv.imwrite(img_fn, frames, [int(cv.IMWRITE_PNG_COMPRESSION), 5])
        cv.imwrite(img_fn, img, [int(cv.IMWRITE_JPEG_QUALITY), 85])


def main():
    rospy.init_node("record")
    _ = Recorder()
    rospy.spin()


if __name__ == "__main__":
    main()