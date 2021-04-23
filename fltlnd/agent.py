import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.observations import TreeObsForRailEnv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

class Agent:

    def __init__(self, state_size, action_size, exp_params, trn_params, checkpoint=None):
        self.__state_size = state_size
        self.__action_size = action_size
        self.__exp_params = exp_params
        self.__trn_params = trn_params

        if checkpoint is None:
            self.__create()
            #self.__model = self.__create_q_model(state_size, action_size)
        else: 
            self.load(checkpoint)

    def __create(self):
        pass

    def act(self, obs):
        pass

    def step(self, obs, action, reward, next_obs, done):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass


class RandomAgent(Agent):

    def __init__(self, state_size, action_size, exp_params, trn_params):
        self.__state_size = state_size
        self.__action_size = action_size

    def act(self, obs):
        return np.random.choice(np.arange(self.__action_size))

    def step(self, obs, action, reward, next_obs, done):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass

class NaiveAgent(Agent):

    def act(self, obs):
        if self.__eps_start > np.random.rand(1)[0]:
            action = np.random.choice(self.__action_size)
        else:
            state_tensor = tf.convert_to_tensor(obs)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.__model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()
        return action

    def step(self, obs, action, reward, next_obs, done):
        #TODO: Implement
        pass

    def load(self, filename):
        self.__model = keras.models.load_model(filename)

    def save(self, filename):
        self.__model.save(filename)

    def __create_q_model(self):
        self.__eps_start = 0 #TODO

        inputs = layers.Input(shape=(self.__state_size))
        layer1 = layers.Dense(32, activation="relu")(inputs)
        action = layers.Dense(self.__action_size, activation="linear")(layer1)

        # Network defined by the Deepmind paper
        # inputs = layers.Input(shape=(84, 84, 4,))

        # Convolutions on the frames on the screen
        # layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        # layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        # layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

        # layer4 = layers.Flatten()(layer3)

        # layer5 = layers.Dense(512, activation="relu")(layer4)
        # action = layers.Dense(num_actions, activation="linear")(layer5)

        self.__model = Model(inputs=inputs, outputs=action)