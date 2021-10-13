#!/usr/bin/env python
from __future__ import print_function
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import uuid

class image_converter:
    
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/drone/down_camera/image_raw",Image,self.callback)

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        imageId = str(uuid.uuid1())
        status = cv2.imwrite('/home/user/simulation_ws/droneDownCamData/'+imageId+'.png',cv_image)
        print("Drone captured an Image with id: ",imageId)
            
        # rospy.loginfo(cv_image)
        
def main(args):
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main(sys.argv)