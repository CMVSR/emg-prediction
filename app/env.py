
# standard
import random

# packages
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pandas as pd
import tensorflow as tf

# internal
import const


class DataEnv(gym.Env):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, features, labels, random=True):
        """
            Initialization function for the DataEnv class. This is an
            environment class to be used as an OpenAI Gym environment for
            training reinforcement learning models.

            params:
                features: the input features to be made available for the agent


        """
        # if random is true, states will be evaluated randomly. otherwise,
        # they will be evaluated in order.
        self.random = random

        # load data and labels
        self.states, self.labels = features, labels

        # load input state shape
        self.shape = self.states[0].shape

        # states should be a 2d array, with (sample, feature)
        if np.ndim(self.states) != 2:
            raise ValueError('Input feature array must be 2d')

        # possible actions are
        # 1. label_action (classificiation action)
        # 2. feature_action (request data action)
        # as a list [label1, label2, ... , feature1, feature2, ...]
        self.feature_actions = (np.shape(self.states)[1])
        self.label_actions = len(np.unique(self.labels))

        # last feature is automatically unmasked
        self.total_available_actions = self.label_actions \
            + self.feature_actions

        # gym.Env attributes
        self.action_space = spaces.Discrete(self.total_available_actions)
        self.observation_space = spaces.Box(
            low=self.states.min(),
            high=self.states.max(),
            shape=self.shape,
            dtype=np.float32
        )
        self.reward_range = (min(const.REWARDS), max(const.REWARDS))

        # set the memory
        self.memory = -1*np.ones(const.ENV_MEMORY_SIZE, dtype=float)

        # keep track of true and false positives
        self.true_positives = 0
        self.false_positives = 0

    def step(self, action):
        self.current_step += 1

        if self.current_step > self.total_available_actions:
            # agent has attempted to make more actions than possible, so punish
            self.reward = const.ILLEGAL_ACTION_PUNISH
            self.reward = self.noisify(self.reward)
            # prevent endless loops where the same action is taken repeatedly
            self.done = True
            return [self.state, self.reward, self.done, self.current_step]

        if not self.get_actions()[action]:
            # action is illegal, so give punishment and don't update
            self.reward = const.ILLEGAL_ACTION_PUNISH
            self.reward = self.noisify(self.reward)
            return [self.state, self.reward, self.done, self.current_step]

        # if the action taken is a feature action, update the mask
        if action >= (self.label_actions):
            # make the feature action a number within the array of
            # self.feature_actions, rather than the entire action_space
            feature_action = action - self.label_actions

            # update the mask
            self.mask[feature_action] = 1
            
            # small penalty for feature action
            self.reward = const.FEATURE_ACTION_PUNISH

        # if the action taken is a classification action
        # or there are no more actions to take the episode is finished
        if action < (self.label_actions):
            # reward correct classification, punish for wrong classification
            if action == self.labels[self.state_index]:
                # scale with time for risk/reward payoff
                self.true_positives += 1
                self.reward = const.CLASSIFICATION_REWARD \
                    + self.precision_modifier()
                
            else:
                self.false_positives += 1
                self.reward = const.CLASSIFICATION_PUNISHMENT \
                    + self.precision_modifier()

            # episode is finished after a classification action
            self.done = True
            # store the memory of the state/action pair
            self.store_memory(action)
            self.reward = self.noisify(self.reward)

        # add the action to the actions taken in this episode
        self.actions_taken.append(action)

        # update the state by applying the mask
        # to the input state at the state index
        self.state = self.states[self.state_index]*self.mask

        return [self.state, self.reward, self.done, self.current_step]

    def render(self):
        print(f'Current Step: {self.current_step}, \
            Actions Taken: {self.actions_taken}')
        print(f'Correct Label: {self.labels[self.state_index]}')
        return self.actions_taken[-1], self.labels[self.state_index]

    def reset(self):
        """
            Reset the environment to a random state input (list of features)
        """
        # state_index is the current features being worked on
        if self.random:
            self.state_index = np.random.randint(0, (len(self.states)-1))
        else:
            self.done_testing = False
            try:
                if self.state_index < len(self.states)-1:
                    self.state_index += 1
                else:
                    self.done_testing = True
            except AttributeError:
                self.state_index = 0

        # reset the mask
        self.mask = np.zeros(self.feature_actions)

        # list of the actions taken in this episode
        self.actions_taken = []

        # Episode has just started, so reward is 0
        self.reward = 0
        self.done = False
        self.current_step = 0

        # apply the mask to the current state with the state index
        self.state = self.states[self.state_index]*self.mask
        return self.state

    def get_actions(self):
        """
            return a mask of available actions based on the actions taken
            by the agent. this will disallow classification actions on the
            first step, 
        """
        # all actions available by default
        actions = np.ones(self.total_available_actions)

        if not self.actions_taken:
            # if no actions have been taken, label actions are unavailable
            # because the agent would just be guessing
            for label_action in range(0, self.label_actions):
                actions[label_action] = 0
        else:
            # otherwise, only feature actions have been taken. for each feature
            # action already taken, that action is unavailable
            for feature_action in self.actions_taken:
                actions[feature_action] = 0

        return actions

    def store_memory(self, action: float):
        self.memory = self.memory[:-1]
        self.memory = np.insert(self.memory, 0, action)

    def noisify(self, reward: float):
        # calculate variance based on stored memory 
        if self.memory[-1] == -1:
            return reward
        else:
            var = np.var(self.memory)
            if var != 0:
                epsilon = const.REWARD_BETA*random.gauss(0.0, 1/var)
                return reward + epsilon
            else:
                return reward

    def precision_modifier(self):
        """
            Calculate the precision modifier of the reward for a classification action
            - True positives are the number of correct classifiations
            - False positives are the number of false classifiations
            - A high precision will have less punishment and more reward
            - A low precision will have more punishment and less reward
            - After 1000 true or false positives are reached (max 2000 classifications), they will be reset to have a
                better representation of the current precision during training
        """
        if (self.true_positives > 1000 or self.false_positives > 1000):
            self.true_positives = 0
            self.false_positives = 0

        if (self.true_positives > 0 and self.false_positives > 0):
            # calculate precision (scale [0,1])
            self.precision = self.true_positives/(self.true_positives + self.false_positives)

            # normalize precision from [0,1] to [-1, 0]
            return (self.precision - 1)
        else:
            return 0
