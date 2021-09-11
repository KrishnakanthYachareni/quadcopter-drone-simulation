# -*- coding: utf-8 -*-
"""ARDrone ddqn training script.

Tasks:
01. [x] DDQN model.
02. [x] OpenCV forward camera topic subscription
03. [x] Image pre-processing: resize
04. [x] Feed input to DDQN
05. [ ] Model load/save/restore during training
06. [x] Tensorboard logging
07. [x] Implement collision checking for ardone.
08. [ ] Implement distance to waypoint calculation using tf.transforms for race gates.

This script is based on PyTorch DQN example by `Adam Paszke <https://github.com/apaszke>`_.
ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

__copyright__ = "Copyright 2019, Elvis Dowson"
__license__ = "BSD-3-Clause"
__author__ = "Elvis Dowson <elvis.dowson@gmail.com>"

import cv2
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL
import time

from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import rospy
import rospkg

from std_msgs.msg import String
from sensor_msgs.msg import Image, RegionOfInterest, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

# import model
from model.dqn.dqn import DQN
from common.logger import TensorBoardLogger
from common.hyperparameters import HyperParameters
from common.replay_memory import ReplayMemory, Transition
from common.util import get_object_attribute_values, get_output_folder

# import our training environment
from gym import wrappers
import ardrone_v1_goto_task_env

# configuration
robot_ns = "drone"

# hyperparameters
seed = rospy.get_param("/{0}/seed".format(robot_ns))

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# observations: screen
frame = None  # camera viewport frame
screen_height = rospy.get_param("/{0}/screen_height".format(robot_ns))
screen_width = rospy.get_param("/{0}/screen_width".format(robot_ns))

# actions
n_actions = rospy.get_param("/{0}/n_actions".format(robot_ns))

# transform pipeline
transform = T.Compose([T.ToPILImage(),
                       T.Resize(size=(screen_height, screen_width), interpolation=PIL.Image.CUBIC),
                       T.ToTensor()])

# create the cv_bridge object
bridge = CvBridge()


######################################################################
# Utility functions and callbacks
# -------------------------------

# image processing functions
def convert_image(ros_image, desired_encoding="rgb8"):
    """Convert the ROS image to the required format.

    This function uses a cv_bridge() helper function to convert the ROS image to the required format.
    - For OpenCV use "bgr8" encoding
    - For Python and OpenAI Gym use "rgb8" encoding.

    :param ros_image: Input ROS image.
    :param desired_encoding: String representation for desired encoding format.
    :return: Converted image as an ndarray.
    """
    try:
        image = bridge.imgmsg_to_cv2(ros_image, desired_encoding=desired_encoding)
        return np.array(image, dtype=np.uint8)
    except CvBridgeError, e:
        print e


def process_image(frame):
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
    image = transform(image)

    # add a batch dimension (BCHW)
    return image.unsqueeze(0).to(device)


# callback functions
def front_camera_image_callback(data):
    """Callback function for processing front camera rgb images.

    :return: None.
    """
    # store the image header
    global frame

    # convert the ros image to rgb and bgr formats
    image = convert_image(data, "rgb8")

    # copy the current
    frame = image



######################################################################
# Replay Memory
# -------------
#
# We'll be using experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data
# later. By sampling from it randomly, the transitions that build up a
# batch are decorrelated. It has been shown that this greatly stabilizes
# and improves the DQN training procedure.
#
# For this, we're going to need two classses:
#
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment. It maps essentially maps (state, action) pairs
#    to their (next_state, reward) result, with the state being the
#    screen difference image as described later on.
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
#

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


######################################################################
# Now, let's define our model. But first, let quickly recap what a DQN is.
#
# DQN algorithm
# -------------
#
# Our environment is deterministic, so all equations presented here are
# also formulated deterministically for the sake of simplicity. In the
# reinforcement learning literature, they would also contain expectations
# over stochastic transitions in the environment.
#
# Our aim will be to train a policy that tries to maximize the discounted,
# cumulative reward
# :math:`R_{t_0} = \sum_{t=t_0}^{\infty} \gamma^{t - t_0} r_t`, where
# :math:`R_{t_0}` is also known as the *return*. The discount,
# :math:`\gamma`, should be a constant between :math:`0` and :math:`1`
# that ensures the sum converges. It makes rewards from the uncertain far
# future less important for our agent than the ones in the near future
# that it can be fairly confident about.
#
# The main idea behind Q-learning is that if we had a function
# :math:`Q^*: State \times Action \rightarrow \mathbb{R}`, that could tell
# us what our return would be, if we were to take an action in a given
# state, then we could easily construct a policy that maximizes our
# rewards:
#
# .. math:: \pi^*(s) = \arg\!\max_a \ Q^*(s, a)
#
# However, we don't know everything about the world, so we don't have
# access to :math:`Q^*`. But, since neural networks are universal function
# approximators, we can simply create one and train it to resemble
# :math:`Q^*`.
#
# For our training update rule, we'll use a fact that every :math:`Q`
# function for some policy obeys the Bellman equation:
#
# .. math:: Q^{\pi}(s, a) = r + \gamma Q^{\pi}(s', \pi(s'))
#
# The difference between the two sides of the equality is known as the
# temporal difference error, :math:`\delta`:
#
# .. math:: \delta = Q(s, a) - (r + \gamma \max_a Q(s', a))
#
# To minimise this error, we will use the `Huber
# loss <https://en.wikipedia.org/wiki/Huber_loss>`__. The Huber loss acts
# like the mean squared error when the error is small, but like the mean
# absolute error when the error is large - this makes it more robust to
# outliers when the estimates of :math:`Q` are very noisy. We calculate
# this over a batch of transitions, :math:`B`, sampled from the replay
# memory:
#
# .. math::
#
#    \mathcal{L} = \frac{1}{|B|}\sum_{(s, a, s', r) \ \in \ B} \mathcal{L}(\delta)
#
# .. math::
#
#    \text{where} \quad \mathcal{L}(\delta) = \begin{cases}
#      \frac{1}{2}{\delta^2}  & \text{for } |\delta| \le 1, \\
#      |\delta| - \frac{1}{2} & \text{otherwise.}
#    \end{cases}
#
# Q-network
# ^^^^^^^^^
#
# Our model will be a convolutional neural network that takes in the
# difference between the current and previous screen patches. It has two
# outputs, representing :math:`Q(s, \mathrm{left})` and
# :math:`Q(s, \mathrm{right})` (where :math:`s` is the input to the
# network). In effect, the network is trying to predict the *expected return* of
# taking each action given the current input.
#

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


######################################################################
# Input extraction
# ^^^^^^^^^^^^^^^^
#
# The code below are utilities for extracting and processing rendered
# images from the environment. It uses the ``torchvision`` package, which
# makes it easy to compose image transforms. Once you run the cell it will
# display an example patch that it extracted.
#

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=PIL.Image.CUBIC),
                    T.ToTensor()])


######################################################################
# Training
# --------
#
# Hyperparameters and utilities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This cell instantiates our model and its optimizer, and defines some
# utilities:
#
# -  ``select_action`` - will select an action accordingly to an epsilon
#    greedy policy. Simply put, we'll sometimes use our model for choosing
#    the action, and sometimes we'll just sample one uniformly. The
#    probability of choosing a random action will start at ``EPS_START``
#    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
#    controls the rate of the decay.
# -  ``plot_durations`` - a helper for plotting the durations of episodes,
#    along with an average over the last 100 episodes (the measure used in
#    the official evaluations). The plot will be underneath the cell
#    containing the main training loop, and will update after every
#    episode.
#


GAMMA = rospy.get_param("/{0}/gamma".format(robot_ns))
EPS_START = rospy.get_param("/{0}/epsilon_start".format(robot_ns))
EPS_END = rospy.get_param("/{0}/epsilon_end".format(robot_ns))
EPS_DECAY = rospy.get_param("/{0}/epsilon_decay".format(robot_ns))

BATCH_SIZE = rospy.get_param("/{0}/batch_size".format(robot_ns))
REPLAY_MEMORY_SIZE = rospy.get_param("/{0}/replay_memory_size".format(robot_ns))
TARGET_UPDATE = rospy.get_param("/{0}/target_network_update_interval".format(robot_ns))

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()

policy_net = DQN(screen_height, screen_width, n_actions)
target_net = DQN(screen_height, screen_width, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), 0.001)
policy_net.to(device)
target_net.to(device)
memory = ReplayMemory(REPLAY_MEMORY_SIZE)

# TODO: Drop this and use self.accumulated_steps
steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
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


######################################################################
# Training loop
# ^^^^^^^^^^^^^
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By defition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state. We also use a target network to compute :math:`V(s_{t+1})` for
# added stability. The target network has its weights kept frozen most of
# the time, but is updated with the policy network's weights every so often.
# This is usually a set number of steps but we shall use episodes for
# simplicity.
#

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net.
    rospy.logdebug("policy_net(state_batch).shape: {}".format(policy_net(state_batch).shape))
    rospy.logdebug("state_batch.shape: {}".format(state_batch.shape))
    rospy.logdebug("action_batch: shape= {}, max= {}, min= {}".format(action_batch.shape, action_batch.max(), action_batch.min()))
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.float()

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss.item()


if __name__ == '__main__':

    # configuration
    env_name = 'ARDroneGoto-v1'
    node_name = rospy.get_param("/{0}/node_name".format(robot_ns))

    # auto-generate output directory name for the current run
    output_dir = rospy.get_param("/{0}/output_dir".format(robot_ns))
    output_dir = get_output_folder('%s/%s' % (output_dir, node_name), env_name)

    # tensorboard logger
    log_inteval = rospy.get_param("/{0}/log_interval".format(robot_ns))
    logger = TensorBoardLogger(output_dir)

    # ros init node
    rospy.init_node(node_name, anonymous=True, log_level=rospy.ERROR)
    rospy.loginfo("Starting node: " + str(node_name))

    # subscribe to the image topic and set callback.
    # image topic names can be remapped in a launch file.
    image_sub = rospy.Subscriber("/drone/front_camera/image_raw",
                                 Image,
                                 front_camera_image_callback,
                                 queue_size=1)

    # create gym environment
    env = gym.make(env_name)
    env.seed(seed)
    torch.manual_seed(seed)
    rospy.loginfo("Gym environment done")

    # set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('ardrone_race_track')
    output_dir = pkg_path + '/' + output_dir
    env = wrappers.Monitor(env, output_dir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    while not rospy.is_shutdown():
        # acquire a single process frame and display it
        env.reset()

        plt.figure()
        plt.imshow(process_image(frame).cpu().squeeze(0).permute(1, 2, 0).numpy(),
                   interpolation='none')
        plt.title('Example extracted screen')
        plt.show()

        ######################################################################
        #
        # Below, you can find the main training loop. At the beginning we reset
        # the environment and initialize the ``state`` Tensor. Then, we sample
        # an action, execute it, observe the next screen and the reward (always
        # 1), and optimize our model once. When the episode ends (our model
        # fails), we restart the loop.
        #
        # Below, `num_episodes` is set small. You should download
        # the notebook and run lot more epsiodes, such as 300+ for meaningful
        # duration improvements.
        #

        num_episodes = rospy.get_param("/{0}/nepisodes".format(robot_ns))
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            env.reset()
            last_screen = process_image(frame)
            current_screen = process_image(frame)
            state = current_screen - last_screen
            for t in count():
                # Select and perform an action
                action = select_action(state)
                _, reward, done, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)

                # Observe new state
                last_screen = current_screen
                current_screen = process_image(frame)
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                loss = optimize_model()
                logger.scalar_summary('Loss per step', env.unwrapped.accumulated_steps, loss)
                logger.scalar_summary('Reward per episode', env.episode_id, env.unwrapped.accumulated_reward)

                if done:
                    episode_durations.append(t + 1)
                    plot_durations()
                    break

            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        print('Complete')
        env.close()
        plt.ioff()
        plt.show()

        ######################################################################
        # Here is the diagram that illustrates the overall resulting data flow.
        #
        # .. figure:: /_static/img/reinforcement_learning_diagram.jpg
        #
        # Actions are chosen either randomly or based on a policy, getting the next
        # step sample from the gym environment. We record the results in the
        # replay memory and also run optimization step on every iteration.
        # Optimization picks a random batch from the replay memory to do training of the
        # new policy. "Older" target_net is also used in optimization to compute the
        # expected Q values; it is updated occasionally to keep it current.
        #
