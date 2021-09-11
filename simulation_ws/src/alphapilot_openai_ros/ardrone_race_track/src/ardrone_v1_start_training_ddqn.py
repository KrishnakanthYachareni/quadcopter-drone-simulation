#!/usr/bin/env python

"""ARDrone ddqn training script.

TODO:
01. [x] DDQN class
02. [x] OpenCV forward camera topic subscription
03. [x] Image pre-processing: resize
04. [x] Feed input to DDQN
05. [x] Model load/save/restore during training
06. [ ] Tensorboard logging
07. [ ] Implement collision checking using tf.transforms for race gates.

ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

__copyright__ = "Copyright 2019, Elvis Dowson"
__license__ = "MIT"
__author__ = "Elvis Dowson <elvis.dowson@gmail.com>"

import argparse
import cv2
import gym
import math
import PIL
import random
import matplotlib.pyplot as plt
import numpy as np

from itertools import count

import rospy
import rospkg

from std_msgs.msg import String
from sensor_msgs.msg import Image, RegionOfInterest, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T

# import our training environment
from gym import wrappers
import ardrone_v1_goto_task_env

# import model
from model.dqn.dqn import DQN
from common.hyperparameters import HyperParameters
from common.replay_memory import ReplayMemory, Transition
from common.util import get_object_attribute_values
from trainer import Trainer
from tester import Tester


class DDQNAgent(object):

    def __init__(self, env, cfg):
        # gym env
        self.env = env

        # hyperparameters
        self.cfg = cfg

        # ros init
        self.node_name = self.cfg.node_name
        rospy.init_node(self.node_name, anonymous=True, log_level=rospy.ERROR)
        rospy.loginfo("Starting node: " + str(self.node_name))
        rospy.on_shutdown(self.cleanup)

        # create the Gym environment
        env = gym.make(self.env)
        env.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        rospy.loginfo("Gym environment done")

        # set the logging system
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('ardrone_race_track')
        output_dir = pkg_path + '/' + self.cfg.output_dir
        self.env = wrappers.Monitor(env, output_dir, force=True)
        rospy.loginfo("Monitor Wrapper started")

        # if gpu is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # observations

        # screen dimensions
        self.screen_height = self.cfg.screen_height
        self.screen_width = self.cfg.screen_width

        # frame
        self.frame = None
        self.frame_size = None
        self.frame_width = None
        self.frame_height = None
        self.image_header = None

        # actions
        self.n_actions = self.cfg.n_actions

        # show image window parameters
        self.show_image = self.cfg.show_image
        self.keystroke = None
        self.resize_window_width = 0
        self.resize_window_height = 0

        # transform pipeline
        self.transform = T.Compose([T.ToPILImage(),
                                    T.Resize(size=(self.screen_height, self.screen_width), interpolation=PIL.Image.CUBIC),
                                    T.ToTensor()])

        # create an opencv image display window
        if self.show_image:
            self.cv_window_name = self.node_name
            cv2.namedWindow(self.cv_window_name, cv2.WINDOW_NORMAL)
            if self.resize_window_height > 0 and self.resize_window_width > 0:
                cv2.ResizeWindow(self.cv_window_name, self.resize_window_width, self.resize_window_height)

        # create the cv_bridge object
        self.bridge = CvBridge()

        # subscribe to the image topic and set callback.
        # image topic names can be remapped in a launch file.
        self.image_sub = rospy.Subscriber("/drone/front_camera/image_raw",
                                          Image,
                                          self.front_camera_image_callback,
                                          queue_size=1)

        # policy network
        self.policy_net = DQN(self.screen_height, self.screen_width, self.n_actions).to(self.device)
        self.target_net = DQN(self.screen_height, self.screen_width, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), self.cfg.lr)

        # experience replay memory
        self.memory = ReplayMemory(10000)

        # training
        self.is_training = True

        # TODO: Remove this
        self.steps_done = 0
        self.episode_durations = []

    def select_action(self, state):

        eps_threshold = self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end) * \
                        math.exp(-1. * self.steps_done / self.cfg.epsilon_decay)
        self.steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.cfg.batch_size:
            return
        transitions = self.memory.sample(self.cfg.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.cfg.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.cfg.gamma) + reward_batch.float()

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self):

        for i_episode in range(self.cfg.nepisodes):
            # Initialize the environment and state
            self.env.reset()
            last_screen = self.process_image(self.frame)
            # convert back to see the processed image
            # torchvision.utils.save_image(last_screen, "%s/processed_image.png" % self.cfg.output_dir)
            current_screen = self.process_image(self.frame)
            state = current_screen - last_screen

            for t in count():
                # select and perform an action
                action = self.select_action(state)
                _, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)

                # observe new state
                last_screen = current_screen
                current_screen = self.process_image(self.frame)
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                # store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # move to the next state
                state = next_state

                # perform one step of the optimization (on the target network)
                self.optimize_model()
                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break

            # update the target network, copying all weights and biases in DQN
            if i_episode % self.cfg.target_network_update_interval == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def front_camera_image_callback(self, data):
        """Callback function for processing front camera rgb images.

        :return: None.
        """
        # store the image header
        self.image_header = data.header

        # convert the ros image to rgb and bgr formats
        frame = self.convert_image(data, "rgb8")

        # store the frame width and height in a pair of global variables
        if self.frame_width is None:
            self.frame_size = (frame.shape[1], frame.shape[0])
            self.frame_width, self.frame_height = self.frame_size

        # copy the current frame
        self.frame = frame.copy()

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

    def process_image(self, frame):
        """Process an image.

        - convert to torch oder (CHW)
        - convert to float
        - rescale i.e. divide by 255
        - convert to torch tensor
        - transform image: resize
        - add a batch dimension (BCHW)

        :type frame: Input image frame.
        :return: Processed image in torch tensor in (BCHW) format
        """
        # convert to torch order (CHW)
        image = frame.transpose((2, 0, 1))

        # convert to float, rescale, convert to torch tensor
        image = np.ascontiguousarray(image, dtype=np.float32) / 255
        image = torch.from_numpy(image)

        # transform image: grayscale, resize
        image = self.transform(image)

        # add a batch dimension (BCHW)
        return image.unsqueeze(0).to(self.device)

    def imshow_image(self, window_name, display_image):
        # update the image display
        if self.show_image:
            cv2.imshow(window_name, display_image)

            self.keystroke = cv2.waitKey(5)

            # Process any keyboard commands
            if self.keystroke is not None and self.keystroke != -1:
                try:
                    cc = chr(self.keystroke & 255).lower()
                    if cc == 'q':
                        # The has press the q key, so exit
                        rospy.signal_shutdown("User hit q key to quit.")
                except:
                    pass

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated

    def load_weights(self, model_path):
        if model_path is None:
            return
        self.policy_net.load_state_dict(torch.load(model_path))

    def save_model(self, output, tag=''):
        torch.save(self.policy_net.state_dict(), '%s/model_%s.pth' % (output, tag))

    def save_config(self, output):
        """Save hyperparamter configuration to a text file,
        :param output: Output folder location.
        :return:
        """
        with open(output + '/config.txt', 'w') as f:
            attr_dict = get_object_attribute_values(self.cfg)
            # get list of object attributes and sort them in alphabetical order
            attr_list = attr_dict.keys()
            attr_list.sort()
            for k in attr_list:
                f.write(str(k) + " = " + str(attr_dict[k]) + "\n")

    def cleanup(self):
        rospy.loginfo("Shutting down node: " + str(self.node_name))
        if self.show_image:
            cv2.destroyAllWindows()

def main():

    """Main entrypoint function.

    :return:
    """

    """
    Example usage:
    python ardrone_v1_start_training_ddqn.py --train --env ARDroneGoto-v1
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', dest='train', action='store_true', help='train model')
    parser.add_argument('--env', default='ARDroneGoto-v1', type=str, help='gym environment')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--model_path', type=str, help='if test, import the model')
    args = parser.parse_args()

    if not args.env:
        print('please specify a gym environment, e.g. --env %s' % args.env)
        exit(0)

    # hyperparameters
    robot_ns = "drone"
    cfg = HyperParameters(robot_ns)

    # agent training loop
    agent = DDQNAgent(env=args.env, cfg=cfg)

    if args.train:
        print("%s DDQN: training ..." % args.env)
        agent.train()

        # trainer = Trainer(agent, args.env, cfg)
        # trainer.train()

        # try:
        #     if agent.show_image:
        #         while not rospy.is_shutdown():
        #             agent.imshow_image(agent.cv_window_name,
        #                                agent.processed_image.cpu().squeeze(0).permute(1, 2, 0).numpy())
        #
        # except KeyboardInterrupt:
        #     print("Shutting down ros node.")
        #     cv2.DestroyAllWindows()

    elif args.test:
        if args.model_path is None:
            print('please specify the model path:', '--model_path %s' % cfg.output_dir)
            exit(0)
        print("%s DDQN: testing ..." % args.env)
        tester = Tester(agent, args.env, args.model_path)
        tester.test()


if __name__ == '__main__':
    main()
