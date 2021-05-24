from abc import ABC, abstractmethod
import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
import pickle

# added for DDDQN
from tensorflow.keras import layers
from glob import glob

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Lambda, Add
from tensorflow.keras import backend as K

from fltlnd.replay_buffer import ReplayBuffer
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.ops import disable_eager_execution

# disable_eager_execution()  # Disable Eager, IMPORTANT!

class Agent(ABC):

    def __init__(self, state_size, action_size, params, exploration=True, train_best=True, base_dir=""):
        self._state_size = state_size
        self._action_size = action_size
        self._params = params
        self._exploration = exploration
        self._base_dir = base_dir

        if train_best:
            self.load_best()
        else:
            self.create()

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def init_params(self):
        pass

    @abstractmethod
    def act(self, obs):
        pass

    @abstractmethod
    def step(self, obs, action, reward, next_obs, done):
        pass

    @abstractmethod
    def save(self, filename, overwrite=False):
        pass

    @abstractmethod
    def load(self, filename):
        pass

    def load_best(self):
        filename = self._base_dir + 'checkpoints/' + str(self)

        if os.path.exists(filename):
            self.load(filename)
        else:
            self.create()

    def save_best(self):
        self.save(self._base_dir + 'checkpoints/' + str(self))

    @abstractmethod
    def __str__(self):
        pass


class RandomAgent(Agent):

    def act(self, obs):
        return np.random.choice(np.arange(self._action_size))

    def step(self, obs, action, reward, next_obs, done):
        pass

    def save(self, filename, overwrite=False):
        pass

    def load(self, filename):
        pass

    def __str__(self):
        return "random-agent"


class DQNAgent(Agent):

    def act(self, obs):

        if self.stats['eps_val'] > np.random.rand(1)[0] and self._exploration:
            action = np.random.choice(self._action_size)
            self.stats['eps_counter'] += 1
        else:
            state_tensor = tf.convert_to_tensor(obs)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self._model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()
        return action

    def episode_end(self):
        # Decay probability of taking random action
        self.stats['eps_val'] = max(self._eps_end, self._eps_decay * self.stats['eps_val'])

    def step(self, obs, action, reward, next_obs, done):
        self._step_count += 1

        #Save experience in replay memory
        self._memory.add(obs, action, reward, next_obs, done)

        # If enough samples are available in memory, get random subset and learn
        if self._step_count % self._update_every == 0 and len(self._memory) > self._buffer_min_size and len(
                self._memory) > self._batch_size:
            self.train()

    def train(self):

        # Get samples from replay buffer
        state_sample, action_sample, rewards_sample, state_next_sample, done_sample = self._memory.sample()

        # Build the updated Q-values for the sampled future states
        # Use the target model for stability
        future_rewards = self._get_future_rewards(state_next_sample)
        # Q value = reward + discount factor * expected future reward
        updated_q_values = rewards_sample + self._gamma * tf.reduce_max(
            future_rewards, axis=1
        )

        # If final frame set the last value to -1
        updated_q_values = updated_q_values * (1 - done_sample) - done_sample

        # Create a mask so we only calculate loss on the updated Q-values
        masks = tf.one_hot(action_sample, self._action_size)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = self._model(state_sample)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = self._loss(updated_q_values, q_action)
            self.stats['loss'] = loss

        # Backpropagation
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

    def load(self, filename):
        self.init_params()

        self._step_count = 0

        self._loss = keras.losses.Huber()
        self._optimizer = keras.optimizers.Adam(learning_rate=self._learning_rate, clipnorm=1.0)

        self._model = keras.models.load_model(filename)

    def save(self, filename, overwrite=True):
        self._model.save(filename, overwrite=overwrite)

    def init_params(self):
        self.stats = {
            "eps_val": self._params['exp_start'],
            "eps_counter": 0,
            "loss": None
        }

        self._eps_end = self._params['exp_end']
        self._eps_decay = self._params['exp_decay']
        self._memory_size = self._params['memory_size']
        self._batch_size = self._params['batch_size']
        self._update_every = self._params['update_every']
        self._learning_rate = self._params['learning_rate']
        self._tau = self._params['tau']
        self._gamma = self._params['gamma']
        self._buffer_min_size = self._params['batch_size']
        self._hidden_sizes = self._params['hidden_sizes']
        self._buffer_size = 32000

        self._memory = ReplayBuffer(self._buffer_size, self._batch_size)

    def create(self):
        self.init_params()

        self._step_count = 0

        self._loss = keras.losses.Huber()
        self._optimizer = keras.optimizers.Adam(learning_rate=self._learning_rate, clipnorm=1.0)

        self._model = self.build_network()

    def build_network(self):
        inputs = layers.Input(shape=(self._state_size,))

        layer = inputs
        for hidden_size in self._hidden_sizes:
            layer = layers.Dense(hidden_size, activation="relu")(layer)
        action = layers.Dense(self._action_size, activation="linear")(layer)
        model = Model(inputs=inputs, outputs=action)
        model.compile(loss='mse', optimizer=self._optimizer, metrics=["mae"])
        return model

    def _get_future_rewards(self, state_next_sample):
        return self._model.predict(state_next_sample)

    def __str__(self):
        return "dqn-agent"


class DoubleDQNAgent(DQNAgent):

    def load(self, filename):
        super().load(filename)
        self.create_target()

    def create(self):
        super().create()
        self.create_target()

    def init_params(self):
        super().init_params()

        self._target_update = self._params['target_update']

    def create_target(self):
        self._model_target = keras.models.clone_model(self._model)
        self._model_target.build((self._state_size,))
        self._model_target.set_weights(self._model.get_weights())

    def step(self, obs, action, reward, next_obs, done):
        super().step(obs, action, reward, next_obs, done)
        if self._step_count % self._target_update == 0:
            # update the the target network with new weights
            self._model_target.set_weights(
                self._tau * np.array(self._model.get_weights()) + (1.0 - self._tau) * np.array(
                    self._model_target.get_weights()))

    def _get_future_rewards(self, state_next_sample):
        return self._model_target.predict(state_next_sample)

    def __str__(self):
        return "double-dqn-agent"


class DuelingDQNAgent(DQNAgent):

    def build_network(self):
        inputs = layers.Input(shape=(self._state_size,))

        X_layer = inputs
        for hidden_size in self._hidden_sizes:
            X_layer = layers.Dense(hidden_size, activation="relu")(X_layer)
        # action = layers.Dense(self._action_size, activation="linear")(X_layer)
        X_layer = Dense(1024, activation='relu', kernel_initializer='he_uniform')(X_layer)
        X_layer = Dense(512, activation='relu', kernel_initializer='he_uniform')(X_layer)

        # value layer
        V_layer = Dense(1, activation='linear', name='V')(X_layer)  # V(S)
        # advantage layer
        A_layer = Dense(self._action_size, activation='linear', name='Ai')(X_layer)  # A(s,a)
        A_layer = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), name='Ao')(A_layer)  # A(s,a)
        # Q layer (V + A)
        Q = Add(name='Q')([V_layer, A_layer])  # Q(s,a)
        Q_model = Model(inputs=[inputs], outputs=[Q], name='qvalue')
        Q_model.compile(loss='mse', optimizer=self._optimizer)
        return Q_model

    def __str__(self):
        return "dueling-dqn-agent"


class DDDQNAgent(DuelingDQNAgent, DoubleDQNAgent):

    def __str__(self):
        return "dddqn-agent"
