"""Agent trainer.

This module performs agent training.
TODO: We shouldn't use a replay memory for SNAIL.

Ref: https://github.com/blackredscarf/pytorch-DQN/blob/master/trainer.py
"""

__license__ = "Apache-2.0"
__author__ = "blackredscarf <blackredscarf@gmail.com>"

import math

import torch

import numpy as np
from common.logger import TensorBoardLogger
from common.util import get_output_folder


class Trainer(object):
    def __init__(self, agent, env, cfg):
        self.agent = agent
        self.env = env
        self.cfg = cfg

        # non-linear epsilon decay
        epsilon_start = self.cfg.epsilon_start
        epsilon_end   = self.cfg.epsilon_end
        epsilon_decay = self.cfg.epsilon_decay

        self.epsilon_by_frame = lambda frame_idx: epsilon_end + (epsilon_start - epsilon_end) * math.exp(
            -1. * frame_idx / epsilon_decay)

        self.output_dir = get_output_folder(self.cfg.output_dir, self.env)
        self.agent.save_config(self.output_dir)
        self.board_logger = TensorBoardLogger(self.output_dir)

    def train(self, pre_fr=0):
        losses = []
        all_rewards = []
        episode_reward = 0
        ep_num = 0
        is_win = False

        # Initialize the environment and state
        self.agent.env.reset()
        last_screen = self.agent.processed_image
        current_screen = self.agent.processed_image
        state = current_screen - last_screen

        # TODO: Ensure frames is int and is getting updated properly
        for fr in range(pre_fr + 1, self.cfg.frames + 1):
            epsilon = self.epsilon_by_frame(fr)

            # TODO: Change this to action, use epsilon
            action = self.agent.select_action2(state, epsilon)

            next_state, reward, done, _ = self.env.step(action)
            reward = torch.tensor([reward], device=self.agent.device)

            # observe new state
            last_screen = current_screen
            current_screen = self.agent.processed_image
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # store the transition in memory
            # TODO: Check memory
            self.agent.memory.add(state, action, reward, next_state, done)

            # move to the next state
            state = next_state
            episode_reward += reward

            # TODO: Debug this. This is the core of the optimization process.
            loss = 0
            if self.agent.memory.size() > self.cfg.batch_size:
                # TODO: Implement optimize_model.
                # TODO: Important to debug this.
                loss = self.agent.optimize_model(fr)
                losses.append(loss)
                self.board_logger.scalar_summary('Loss per frame', fr, loss)

            # TODO: Review the remaining sections carefully.

            if fr % self.cfg.print_interval == 0:
                print("frames: %5d, reward: %5f, loss: %4f episode: %4d" % (
                 fr, np.mean(all_rewards[-10:]), loss, ep_num))

            if fr % self.cfg.log_interval == 0:
                self.board_logger.scalar_summary('Reward per episode', ep_num, all_rewards[-1])

            # TODO: There is no agent.save_checkpoint implemented. Implement this method.
            # Ref: https://pytorch.org/tutorials/beginner/saving_loading_models.html to implement it.
            if self.cfg.checkpoint and fr % self.cfg.checkpoint_interval == 0:
                # TODO: Use the checkpoint code from previous project.
                # TODO: Implement the corresponding load from checkpoint code as-well. Test it.
                self.agent.save_checkpoint(fr, self.output_dir)

            if done:
                state = self.agent.env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0
                ep_num += 1
                avg_reward = float(np.mean(all_rewards[-100:]))
                self.board_logger.scalar_summary('Best 100-episodes average reward', ep_num, avg_reward)

                if len(all_rewards) >= 100 and avg_reward >= self.cfg.win_reward and all_rewards[
                    -1] > self.cfg.win_reward:
                    is_win = True
                    self.agent.save_model(self.output_dir, 'best')
                    print('Ran %d episodes best 100-episodes average reward is %3f. Solved after %d trials.' % (
                    ep_num, avg_reward, ep_num - 100))
                    if self.cfg.win_break:
                        break

        if not is_win:
            print('Did not solve after %d episodes' % ep_num)
            self.agent.save_model(self.output_dir, 'last')
