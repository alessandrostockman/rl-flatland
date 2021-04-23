import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.observations import TreeObsForRailEnv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

class DDDQNAgent:
    def __init__(self, state_size, action_size, training_parameters, new=True):
        if new:
            self.__state_size = state_size
            self.__action_size = action_size
            self.__model = self.__create_q_model(state_size, action_size)
            #self.__policy = DDDQNPolicy(state_size, action_size, training_parameters)
        pass

    def step(self, agent, prev_obs, prev_action, all_rewards, obs, done):
        #TODO: Keras
        pass

    def act(self, obs, eps_start=0):
        if eps_start > np.random.rand(1)[0]:
            action = np.random.choice(self.__action_size)
        else:
            state_tensor = tf.convert_to_tensor(obs)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.policy(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()
        return action

    def load(self, filename):
        self.__policy = keras.models.load_model(filename)
        return

    def save(self, filename):
        self.__policy.save(filename)

    def __create_q_model(self, input_size, output_size):
        inputs = layers.Input(shape=(input_size))
        layer1 = layers.Dense(32, activation="relu")(inputs)
        action = layers.Dense(output_size, activation="linear")(layer1)

        # Network defined by the Deepmind paper
        # inputs = layers.Input(shape=(84, 84, 4,))

        # Convolutions on the frames on the screen
        # layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        # layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        # layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

        # layer4 = layers.Flatten()(layer3)

        # layer5 = layers.Dense(512, activation="relu")(layer4)
        # action = layers.Dense(num_actions, activation="linear")(layer5)


        return Model(inputs=inputs, outputs=action)

