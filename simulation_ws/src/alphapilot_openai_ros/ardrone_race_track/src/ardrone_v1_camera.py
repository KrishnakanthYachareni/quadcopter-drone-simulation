# -*- coding: utf-8 -*-
"""ARDrone v1 camera example

This example is used to display the front camera viewport. It can be used to debug an image
processing pipeline, before feeding the images into a model.
"""

__copyright__ = "Copyright 2019, Elvis Dowson"
__license__ = "MIT"
__author__ = "Elvis Dowson <elvis.dowson@gmail.com>"

import cv2
import numpy as np
import sys

import rospy
import rospkg

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ROSCamera(object):

    def __init__(self, topic="", show_image=False, window_name="camera_window"):

        self.topic=topic
        self.show_image = show_image
        self.window_name = window_name

        # camera frame and properties
        self.frame = None
        self.frame_size = None
        self.frame_width = None
        self.frame_height = None

        # processed image
        self.resize_width = 320
        self.resize_height = 180
        self.processed_image = None

        # create the cv_bridge object
        self.bridge = CvBridge()

        # Create the main display window
        if self.show_image:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # subscribe to the image topic and set callback.
        # image topic names can be remapped in a launch file.
        self.image_sub = rospy.Subscriber(self.topic,
                                          Image,
                                          callback=self.camera_callback,
                                          queue_size=1)

    # callback functions
    def camera_callback(self, data):
        """Callback function for processing front camera rgb images.

        :return: None.
        """
        # convert the ros image to bgr8 formats
        frame = self.convert_image(data, "bgr8")
        self.frame = frame

        # store the frame width and height in a pair of global variables
        if self.frame_width is None:
            self.frame_size = (frame.shape[1], frame.shape[0])
            self.frame_width, self.frame_height = self.frame_size

        # process image
        self.processed_image = self.process_image(frame)

    # image processing functions
    def convert_image(self, ros_image, desired_encoding="rgb8"):
        """Convert the ROS image to the required format.

        This function uses a cv_bridge() helper function to convert the ROS image to the required format.
        - For OpenCV use "bgr8" encoding
        - For Python and OpenAI Gym use "rgb8" encoding.

        :param ros_image: Input ROS image.
        :param desired_encoding: String representation for desired encoding format.
        :return: Converted image as an ndarray.
        """
        try:
            image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding=desired_encoding)
            return np.array(image, dtype=np.uint8)
        except CvBridgeError, e:
            print e

    def process_image(self, image):
        """Placeholder function for image processing.

        :param frame: Input image.
        :return: Processed image.
        """
        image = self.resize(image, (self.resize_width, self.resize_height))
        image = self.grayscale(image, code=cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def resize(image, size):
        """Resize image.

        :param image: Source image
        :param size: dim (width, height)
        :return:
        """
        return cv2.resize(src=image, dsize=size)

    @staticmethod
    def grayscale(image, code=cv2.COLOR_BGR2GRAY):
        """Convert to grayscale
        For OpenCV BGR encoded images, use code=cv2.COLOR_BGR2GRAY.
        For RGB images, use code=cv2.COLOR_RGB2GRAY
        :param image: Input image
        :param code: cv2 color conversion code.
        :return:
        """
        return cv2.cvtColor(src=image, code=code)

    @staticmethod
    def imshow_image(window_name, frame):
        """Display an image in a window using OpenCV.

        :param window_name: Window name
        :param frame: Frame
        :return:
        """
        # update the image display
        cv2.imshow(window_name, frame)

        # process any keyboard commands
        keystroke = cv2.waitKey(5)
        if keystroke is not None and keystroke != -1:
            try:
                cc = chr(keystroke & 255).lower()
                if cc == 'q':
                    # The has press the q key, so exit
                    rospy.signal_shutdown("User hit q key to quit.")
            except:
                pass

    def cleanup(self):
        """ ROS shutdown cleanup call back function.
        """
        if self.show_image:
            cv2.destroyAllWindows()


# main entry-point
def main(args):

    # configuration
    node_name = "ardrone_v1_camera"
    camera_topic = "/drone/front_camera/image_raw"
    window_name = 'ardrone_forward_camera_mono'
    output_dir = 'output'

    # ros init node
    rospy.init_node(node_name, anonymous=True, log_level=rospy.INFO)
    rospy.loginfo("Starting node: " + str(node_name))

    # set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('ardrone_race_track')
    output_dir = pkg_path + '/' + output_dir

    try:
        camera = ROSCamera(topic=camera_topic, show_image=True, window_name=window_name)
        while not rospy.is_shutdown():
            if camera.processed_image is not None:
                camera.imshow_image(window_name, camera.processed_image)

    except KeyboardInterrupt:
        print("Shutting down %s node." % node_name)


if __name__ == '__main__':
    main(sys.argv)
