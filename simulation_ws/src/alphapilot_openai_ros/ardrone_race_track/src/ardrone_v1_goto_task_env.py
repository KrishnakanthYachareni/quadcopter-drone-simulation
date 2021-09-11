"""ARDroneGoto-v1 task environment definition.

This module is based on the OpenAI ROS ParrotDroneGoto-v0 environment originally developed by
Miguel Angel Rodriguez, The Construct.

url: https://github.com/edowson/openai_ros/blob/kinetic-devel/openai_ros/src/openai_ros/task_envs/parrotdrone/parrotdrone_goto.py
"""

import rospy
import numpy

from geometry_msgs.msg import Point, Vector3
from tf.transformations import euler_from_quaternion

from gazebo_msgs.msg import ContactsState, ContactState

from gym import spaces
from gym.envs.registration import register

from ardrone_v1_robot_env import ARDroneEnv

timestep_limit_per_episode = 10000  # can be any value

register(
        id='ARDroneGoto-v1',
        entry_point='ardrone_v1_goto_task_env:ARDroneGotoEnv',
        timestep_limit=timestep_limit_per_episode,
    )


class ARDroneGotoEnv(ARDroneEnv):

    def __init__(self):
        """
        Make ardrone learn how to navigate to get to a point
        """

        # Only variable needed to be set here
        number_actions = rospy.get_param('/drone/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)

        # Actions and Observations
        self.linear_forward_speed = rospy.get_param('/drone/linear_forward_speed')
        self.angular_turn_speed = rospy.get_param('/drone/angular_turn_speed')
        self.angular_speed = rospy.get_param('/drone/angular_speed')

        self.init_linear_speed_vector = Vector3()
        self.init_linear_speed_vector.x = rospy.get_param('/drone/init_linear_speed_vector/x')
        self.init_linear_speed_vector.y = rospy.get_param('/drone/init_linear_speed_vector/y')
        self.init_linear_speed_vector.z = rospy.get_param('/drone/init_linear_speed_vector/z')

        self.init_angular_turn_speed = rospy.get_param('/drone/init_angular_turn_speed')

        self.min_sonar_value = rospy.get_param('/drone/min_sonar_value')
        self.max_sonar_value = rospy.get_param('/drone/max_sonar_value')

        self.min_collision_value = False
        self.max_collision_value = True

        # Get WorkSpace Cube Dimensions
        self.work_space_x_max = rospy.get_param("/drone/work_space/x_max")
        self.work_space_x_min = rospy.get_param("/drone/work_space/x_min")
        self.work_space_y_max = rospy.get_param("/drone/work_space/y_max")
        self.work_space_y_min = rospy.get_param("/drone/work_space/y_min")
        self.work_space_z_max = rospy.get_param("/drone/work_space/z_max")
        self.work_space_z_min = rospy.get_param("/drone/work_space/z_min")

        # Maximum RPY values
        self.max_roll = rospy.get_param("/drone/max_roll")
        self.max_pitch = rospy.get_param("/drone/max_pitch")
        self.max_yaw = rospy.get_param("/drone/max_yaw")

        # Get Desired Point to Get
        self.desired_point = Point()
        self.desired_point.x = rospy.get_param("/drone/desired_pose/x")
        self.desired_point.y = rospy.get_param("/drone/desired_pose/y")
        self.desired_point.z = rospy.get_param("/drone/desired_pose/z")

        self.desired_point_epsilon = rospy.get_param("/drone/desired_point_epsilon")

        # We place the Maximum and minimum values of the X,Y,Z,R,P,Y of the pose

        high = numpy.array([self.work_space_x_max,
                            self.work_space_y_max,
                            self.work_space_z_max,
                            self.max_roll,
                            self.max_pitch,
                            self.max_yaw,
                            self.max_sonar_value,
                            self.max_collision_value])

        low = numpy.array([ self.work_space_x_min,
                            self.work_space_y_min,
                            self.work_space_z_min,
                            -1*self.max_roll,
                            -1*self.max_pitch,
                            -numpy.inf,
                            self.min_sonar_value,
                            self.min_collision_value])

        self.observation_space = spaces.Box(low, high)

        rospy.logdebug("ACTION SPACES TYPE ===> "+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE ===> "+str(self.observation_space))

        # Rewards
        self.closer_to_point_reward = rospy.get_param("/drone/closer_to_point_reward")
        self.no_collision_reward = rospy.get_param("/drone/no_collision_reward")
        self.end_episode_points = rospy.get_param("/drone/end_episode_points")

        # Penalty
        self.time_step_penalty = rospy.get_param("/drone/time_step_penalty")

        self.accumulated_steps = 0
        self.accumulated_reward = 0.0

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(ARDroneGotoEnv, self).__init__()

    def _set_init_pose(self):
        """
        Sets the Robot in its init linear and angular speeds
        and lands the robot. Its preparing it to be reseted in the world.
        """
        #raw_input("INIT SPEED PRESS")
        self.move_base(self.init_linear_speed_vector,
                       self.init_angular_turn_speed,
                       epsilon=0.05,
                       update_rate=10)
        # We Issue the landing command to be sure it starts landing
        #raw_input("LAND PRESS")
        #self.land()

        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        #raw_input("TakeOFF PRESS")
        # We TakeOff before sending any movement commands
        self.takeoff()

        # For Info Purposes
        self.accumulated_reward = 0.0
        # We get the initial pose to mesure the distance from the desired point.
        gt_pose = self.get_gt_pose()
        self.previous_distance_from_des_point = self.get_distance_from_desired_point(gt_pose.position)

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the drone
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """

        rospy.logdebug("Start Set Action ==> " + str(action))
        # We convert the actions to speed movements to send to the parent class of Parrot AR Drone
        linear_speed_vector = Vector3()
        angular_speed = 0.0

        if action == 0: #FORWARDS
            linear_speed_vector.x = self.linear_forward_speed
            self.last_action = "FORWARDS"
        elif action == 1: #BACKWARDS
            linear_speed_vector.x = -1*self.linear_forward_speed
            self.last_action = "BACKWARDS"
        elif action == 2: #STRAFE_LEFT
            linear_speed_vector.y = self.linear_forward_speed
            self.last_action = "STRAFE_LEFT"
        elif action == 3: #STRAFE_RIGHT
            linear_speed_vector.y = -1*self.linear_forward_speed
            self.last_action = "STRAFE_RIGHT"
        elif action == 4: #UP
            linear_speed_vector.z = self.linear_forward_speed
            self.last_action = "UP"
        elif action == 5: #DOWN
            linear_speed_vector.z = -1*self.linear_forward_speed
            self.last_action = "DOWN"
        elif action == 6: #STOP
            linear_speed_vector.x = 0.0
            linear_speed_vector.y = 0.0
            linear_speed_vector.z = 0.0
            self.last_action = "STOP"

        # We tell drone the linear and angular speed to set to execute
        self.move_base(linear_speed_vector,
                       angular_speed,
                       epsilon=0.05,
                       update_rate=10)

        rospy.logdebug("END Set Action ==> "+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have access to, we need to read the
        droneEnv API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # Get the laser scan data
        gt_pose = self.get_gt_pose()

        # Get the orientation of the cube in RPY
        roll, pitch, yaw = self.get_orientation_euler(gt_pose.orientation)

        # Get the sonar value
        sonar = self.get_sonar()
        sonar_value = sonar.range

        # Get collision information
        collision = self.detect_collision(self.get_contacts())

        # Simplify the spatial grid to make learning go faster
        observations = [int(gt_pose.position.x),
                        int(gt_pose.position.y),
                        int(gt_pose.position.z),
                        round(roll, 1),
                        round(pitch, 1),
                        round(yaw, 1),
                        round(sonar_value, 1),
                        bool(collision)]

        rospy.logdebug("Observations ==> "+str(observations))
        rospy.logdebug("END Get Observation ==>")
        return observations

    def _is_done(self, observations):
        """
        The done can be done due to three reasons:
        1) It went outside the workspace
        2) It collided with a race gate
        3) It detected something with the sonar that is too close
        4) It flipped due to a crash or something
        5) It has reached the desired point
        """

        episode_done = False

        current_position = Point()
        current_position.x = observations[0]
        current_position.y = observations[1]
        current_position.z = observations[2]

        # TODO: See if there is a better class to represent orientation (rpy) than a point (xyq)
        current_orientation = Point()
        current_orientation.x = observations[3]
        current_orientation.y = observations[4]
        current_orientation.z = observations[5]

        sonar_value = observations[6]

        collision = observations[7]

        is_inside_workspace_now = self.is_inside_workspace(current_position)
        sonar_detected_something_too_close_now = self.sonar_detected_something_too_close(sonar_value)
        drone_flipped = self.drone_has_flipped(current_orientation)
        has_reached_des_point = self.is_in_desired_position(current_position, self.desired_point_epsilon)

        rospy.logwarn(">>>>>> DONE RESULTS <<<<<")
        if not is_inside_workspace_now:
            rospy.logwarn("is inside workspace = " + str(is_inside_workspace_now))
        else:
            rospy.loginfo("is inside workspace = " + str(is_inside_workspace_now))

        if collision:
            rospy.logwarn("collision detected = " + str(collision))
        else:
            rospy.loginfo("collision = " + str(collision))

        if sonar_detected_something_too_close_now:
            rospy.logwarn("sonar detected something too close = " + str(sonar_detected_something_too_close_now))
        else:
            rospy.loginfo("sonar detected something too close = " + str(sonar_detected_something_too_close_now))

        if drone_flipped:
            rospy.logwarn("drone has flipped = " + str(drone_flipped))
        else:
            rospy.loginfo("drone has flipped = " + str(drone_flipped))

        if has_reached_des_point:
            rospy.logerr("has reached destination point = " + str(has_reached_des_point))
        else:
            rospy.logwarn("has reached destination point = " + str(has_reached_des_point))

        # We see if we are outside the Learning Space
        episode_done = not(is_inside_workspace_now) or collision or sonar_detected_something_too_close_now or drone_flipped or has_reached_des_point

        if episode_done:
            rospy.logerr("episode done ====> " + str(episode_done))
        else:
            rospy.logwarn("episode done ====> " + str(episode_done))

        return episode_done

    def _compute_reward(self, observations, done):

        current_position = Point()
        current_position.x = observations[0]
        current_position.y = observations[1]
        current_position.z = observations[2]

        collision = observations[7]

        distance_from_des_point = self.get_distance_from_desired_point(current_position)
        distance_difference = distance_from_des_point - self.previous_distance_from_des_point

        if not done:

            # If there has been a decrease in the distance to the desired point, we reward it
            if distance_difference < 0.0:
                rospy.logwarn("DECREASE IN DISTANCE GOOD")
                reward = self.closer_to_point_reward
            else:
                rospy.logwarn("INCREASE IN DISTANCE BAD")
                reward = -1 * self.time_step_penalty

        else:

            if self.is_in_desired_position(current_position, epsilon=0.1) and not collision:
                reward = self.end_episode_points + self.no_collision_reward
            elif self.is_in_desired_position(current_position, epsilon=0.1) and collision:
                reward = -1 * self.no_collision_reward
            else:
                reward = -1 * self.end_episode_points

        self.previous_distance_from_des_point = distance_from_des_point

        rospy.logdebug("reward=" + str(reward))
        self.accumulated_reward += reward
        rospy.logdebug("accumulated_reward = " + str(self.accumulated_reward))
        self.accumulated_steps += 1
        rospy.logdebug("accumulated_steps = " + str(self.accumulated_steps))

        return reward

    # Internal TaskEnv Methods

    def is_in_desired_position(self, current_position, epsilon=0.05):
        """
        It return True if the current position is similar to the desired poistion
        """

        is_in_desired_pos = False

        x_pos_plus = self.desired_point.x + epsilon
        x_pos_minus = self.desired_point.x - epsilon
        y_pos_plus = self.desired_point.y + epsilon
        y_pos_minus = self.desired_point.y - epsilon
        z_pos_plus = self.desired_point.z + epsilon
        z_pos_minus = self.desired_point.z - epsilon

        x_current = current_position.x
        y_current = current_position.y
        z_current = current_position.z

        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)
        z_pos_are_close = (z_current <= z_pos_plus) and (z_current > z_pos_minus)

        is_in_desired_pos = x_pos_are_close and y_pos_are_close and z_pos_are_close

        rospy.logwarn("###### IS IN DESIRED POS ? ######")
        rospy.logwarn("current position: x= %f y= %f z= %f" % (current_position.x, current_position.y, current_position.z))

        rospy.logwarn("x_pos_plus= "+str(x_pos_plus)+",x_pos_minus= "+str(x_pos_minus))
        rospy.logwarn("y_pos_plus= "+str(y_pos_plus)+",y_pos_minus= "+str(y_pos_minus))
        rospy.logwarn("z_pos_plus= " + str(z_pos_plus) + ",z_pos_minus= " + str(z_pos_minus))

        rospy.logwarn("x_pos_are_close "+str(x_pos_are_close))
        rospy.logwarn("y_pos_are_close "+str(y_pos_are_close))
        rospy.logwarn("z_pos_are_close " + str(z_pos_are_close))

        rospy.logwarn("is_in_desired_pos "+str(is_in_desired_pos))
        rospy.logwarn("############")

        return is_in_desired_pos

    def is_inside_workspace(self,current_position):
        """
        Check if the Drone is inside the Workspace defined
        """
        is_inside = False

        rospy.logwarn("##### INSIDE WORKSPACE? #######")
        rospy.logwarn("XYZ current_position: x= %f y= %f z= %f" % (current_position.x, current_position.y, current_position.z))
        rospy.logwarn("workspace_x_max= "+str(self.work_space_x_max)+",workspace_x_min= "+str(self.work_space_x_min))
        rospy.logwarn("workspace_y_max= "+str(self.work_space_y_max)+",workspace_y_min= "+str(self.work_space_y_min))
        rospy.logwarn("workspace_z_max= "+str(self.work_space_z_max)+",workspace_z_min= "+str(self.work_space_z_min))
        rospy.logwarn("############")

        if current_position.x > self.work_space_x_min and current_position.x <= self.work_space_x_max:
            if current_position.y > self.work_space_y_min and current_position.y <= self.work_space_y_max:
                if current_position.z > self.work_space_z_min and current_position.z <= self.work_space_z_max:
                    is_inside = True

        return is_inside

    def detect_collision(self, contacts):
        """Detect collisions.
        This method currently performs a simple collision test. It will detect if the agent collides with
        the ground plane and a race gate.
        :param contacts:
        :return:
        """
        if self.contacts.states:
            return True
        else:
            return False

    def sonar_detected_something_too_close(self, sonar_value):
        """
        Detects if there is something too close to the drone front
        """
        rospy.logwarn("##### SONAR TOO CLOSE? #######")
        rospy.logwarn("sonar_value"+str(sonar_value)+",min_sonar_value="+str(self.min_sonar_value))
        rospy.logwarn("############")

        too_close = sonar_value < self.min_sonar_value

        return too_close

    def drone_has_flipped(self,current_orientation):
        """
        Based on the orientation RPY given states if the drone has flipped
        """
        has_flipped = True

        self.max_roll = rospy.get_param("/drone/max_roll")
        self.max_pitch = rospy.get_param("/drone/max_pitch")

        rospy.logwarn("#### HAS FLIPPED? ########")
        rospy.logwarn("RPY current_orientation: r= %f, p= %f, y= %f" % (current_orientation.x, current_orientation.y, current_orientation.z))
        rospy.logwarn("max_roll"+str(self.max_roll)+",min_roll="+str(-1*self.max_roll))
        rospy.logwarn("max_pitch"+str(self.max_pitch)+",min_pitch="+str(-1*self.max_pitch))
        rospy.logwarn("############")

        if current_orientation.x > -1*self.max_roll and current_orientation.x <= self.max_roll:
            if current_orientation.y > -1*self.max_pitch and current_orientation.y <= self.max_pitch:
                    has_flipped = False

        return has_flipped

    def get_distance_from_desired_point(self, current_position):
        """
        Calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        distance = self.get_distance_from_point(current_position,
                                                self.desired_point)

        return distance

    def get_distance_from_point(self, pstart, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param pstart:
        :param p_end:
        :return:
        """
        a = numpy.array((pstart.x, pstart.y, pstart.z))
        b = numpy.array((p_end.x, p_end.y, p_end.z))

        distance = numpy.linalg.norm(a - b)

        return distance

    def get_orientation_euler(self, quaternion_vector):
        # We convert from quaternions to euler
        orientation_list = [quaternion_vector.x,
                            quaternion_vector.y,
                            quaternion_vector.z,
                            quaternion_vector.w]

        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw
