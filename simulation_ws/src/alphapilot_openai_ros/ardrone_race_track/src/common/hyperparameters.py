#!/usr/bin/env python

"""Hyperparameters module.

"""

__copyright__ = "Copyright 2019, Elvis Dowson"
__license__ = "MIT"
__author__ = "Elvis Dowson <elvis.dowson@gmail.com>"


import rospy


class HyperParameters(object):
    """HyperParameters configuration.

       This retrieves the hyperparameters from the ros parameter server.
    """

    def __init__(self, robot_ns):
        # robot ns
        self.robot_ns = robot_ns

        # loads parameters from the ros param server
        # parameters are stored in a yaml file inside the config directory.
        # they are loaded at runtime by the launch file.

        # general parameters
        self.node_name = rospy.get_param("/{0}/node_name".format(robot_ns))
        self.output_dir = rospy.get_param("/{0}/output_dir".format(robot_ns)) + '/' + self.node_name

        self.checkpoint = rospy.get_param("/{0}/checkpoint".format(robot_ns))
        self.checkpoint_interval = rospy.get_param("/{0}/checkpoint_interval".format(robot_ns))

        self.seed = rospy.get_param("/{0}/seed".format(robot_ns))

        self.log_interval = rospy.get_param("/{0}/log_interval".format(robot_ns))

        # screen parameters
        self.screen_height = rospy.get_param("/{0}/screen_height".format(robot_ns))
        self.screen_width = rospy.get_param("/{0}/screen_width".format(robot_ns))

        self.show_image = rospy.get_param("/{0}/show_image".format(robot_ns), True)

        # ddqn parameters
        self.gamma = rospy.get_param("/{0}/gamma".format(robot_ns))

        self.epsilon_start = rospy.get_param("/{0}/epsilon_start".format(robot_ns))
        self.epsilon_end = rospy.get_param("/{0}/epsilon_end".format(robot_ns))
        self.epsilon_decay = rospy.get_param("/{0}/epsilon_decay".format(robot_ns))

        self.batch_size = rospy.get_param("/{0}/batch_size".format(robot_ns))
        self.target_network_update_interval = rospy.get_param("/{0}/target_network_update_interval".format(robot_ns))

        self.nepisodes = rospy.get_param("/{0}/nepisodes".format(robot_ns))
        self.nsteps = rospy.get_param("/{0}/nsteps".format(robot_ns))

        # optimizer parameters
        self.lr = rospy.get_param("/{0}/lr".format(robot_ns))

        self.n_actions = rospy.get_param("/{0}/n_actions".format(robot_ns))
        self.n_observations = rospy.get_param("/{0}/n_observations".format(robot_ns))

        # additional configuration
        self.frames = 0
