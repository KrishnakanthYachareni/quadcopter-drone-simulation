#!/usr/bin/env python
import rospy
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import Pose

class GazeboLinkPose:
    link_name = ''
    link_pose = Pose()

    def __init__(self, link_name):
        self.link_name = link_name
        self.link_name_rectified = link_name.replace("::", "_")

        if not self.link_name:
            raise ValueError("'link_name' is an empty string")

        #Subscribing to gazebo link states which provides scene states
        self.states_sub = rospy.Subscriber(
            "/gazebo/link_states", LinkStates, self.callback)
        
        # Publishing the gazebo publisher with communication message type as a Pose    
        self.pose_pub = rospy.Publisher(
            "/gazebo/" + self.link_name_rectified, Pose, queue_size=10)

    def callback(self, data):
        try:
            ind = data.name.index(self.link_name)
            self.link_pose = data.pose[ind]
        except ValueError:
            pass


if __name__ == '__main__':
    try:
        # Initializing the new node
        rospy.init_node('gazebo_link_pose', anonymous=True)

        # Scene base link name will be getting from ./launch/model_tracker.launch file
        # Base link name is: sjtu_drone::base_link
        gp = GazeboLinkPose(rospy.get_param('~link_name'))
        publish_rate = rospy.get_param('~publish_rate', 10)

        rate = rospy.Rate(publish_rate)
        while not rospy.is_shutdown():
            # Publising the drone position to the defined topic
            gp.pose_pub.publish(gp.link_pose)
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
