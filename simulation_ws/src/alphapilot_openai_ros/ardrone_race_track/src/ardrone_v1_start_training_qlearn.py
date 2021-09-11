#!/usr/bin/env python

"""ARDrone qlearn training script.

"""

import gym
import numpy
import time

# ROS packages required
import rospy
import rospkg

# import training environment
from gym import wrappers
from ardrone_v1_goto_task_env import ARDroneGotoEnv

# import model
from model.qlearn.qlearn import QLearn


if __name__ == '__main__':

    rospy.init_node('ardrone_v1_goto_qlearn', anonymous=True, log_level=rospy.WARN)

    # Create the Gym environment
    env = gym.make('ARDroneGoto-v1')
    rospy.loginfo("Gym environment done")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('ardrone_race_track')
    outdir = pkg_path + '/output/ardrone_v1_goto_qlearn'
    env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    robot_ns = "drone"

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("/{0}/alpha".format(robot_ns))
    Epsilon = rospy.get_param("/{0}/epsilon".format(robot_ns))
    Gamma = rospy.get_param("/{0}/gamma".format(robot_ns))
    epsilon_discount = rospy.get_param("/{0}/epsilon_discount".format(robot_ns))
    nepisodes = rospy.get_param("/{0}/nepisodes".format(robot_ns))
    nsteps = rospy.get_param("/{0}/nsteps".format(robot_ns))

    # Initialises the algorithm that we are going to use for learning
    qlearn = QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = 0

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        rospy.logdebug("############### START EPISODE => " + str(x))

        accumulated_reward = 0
        done = False
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        state = ''.join(map(str, observation))

        # Show on screen the actual situation of the robot
        # env.render()
        # for each episode, we test the robot for nsteps
        for i in range(nsteps):
            rospy.logwarn("############### Start Step => " + str(i))
            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            rospy.logwarn("Next action is: %d", action)
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)
            rospy.loginfo(str(observation) + " " + str(reward))
            accumulated_reward += reward
            if highest_reward < accumulated_reward:
                highest_reward = accumulated_reward

            nextState = ''.join(map(str, observation))

            # Make the algorithm learn based on the results
            rospy.logwarn("# state we were => " + str(state))
            rospy.logwarn("# action that we took => " + str(action))
            rospy.logwarn("# reward that action gave => " + str(reward))
            rospy.logwarn("# episode accumulated_reward => " + str(accumulated_reward))
            rospy.logwarn("# State in which we will start next step => " + str(nextState))
            qlearn.learn(state, action, reward, nextState)

            if not done:
                rospy.logwarn("NOT DONE")
                state = nextState
            else:
                rospy.logwarn("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
            rospy.logwarn("############### END Step => " + str(i))
            #raw_input("Next Step...PRESS KEY")
            # rospy.sleep(2.0)
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logerr(("EP: " + str(x + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
            round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
            accumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))

    rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
        initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    # print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
