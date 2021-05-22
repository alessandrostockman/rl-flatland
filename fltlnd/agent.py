from abc import ABC, abstractmethod
import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model


# added for DDDQN
from tensorflow.keras import layers
from glob import glob
from random import sample  # used to get random minibacth
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Lambda, Add
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from collections import deque  # needed for replay memory
from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()  # Disable Eager, IMPORTANT!

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
        self.save(self._base_dir + 'checkpoints/' + str(self), overwrite=True)

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

        if self.eps > np.random.rand(1)[0] and self._exploration:
            action = np.random.choice(self._action_size)
        else:
            state_tensor = tf.convert_to_tensor(obs)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self._model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()
        return action

    def episode_end(self):
        # Decay probability of taking random action
        self.eps = max(self._eps_end, self._eps_decay * self.eps)

    def step(self, obs, action, reward, next_obs, done):
        self._step_count += 1

        # Save actions and states in replay buffer
        self._action_history.append(action)
        self._state_history.append(obs)
        self._state_next_history.append(next_obs)
        self._done_history.append(done)
        self._rewards_history.append(reward)

        if self._step_count % self._update_every == 0 and len(self._done_history) > self._buffer_min_size:
            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(self._done_history)), size=self._batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([self._state_history[i] for i in indices])
            state_next_sample = np.array([self._state_next_history[i] for i in indices])
            rewards_sample = [self._rewards_history[i] for i in indices]
            action_sample = [self._action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(self._done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = self._model_target.predict(state_next_sample)
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

            # Backpropagation
            grads = tape.gradient(loss, self._model.trainable_variables)
            self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

        if self._step_count % 10000 == 0:  # TODO update_target_network as parameter
            # update the the target network with new weights
            self._model_target.set_weights(self._tau * np.array(self._model.get_weights()) + (1.0 - self._tau) * np.array(self._model_target.get_weights()))

        # Limit the state and reward history
        if len(self._rewards_history) > self._memory_size:
            del self._rewards_history[:1]
            del self._state_history[:1]
            del self._state_next_history[:1]
            del self._action_history[:1]
            del self._done_history[:1]

    def load(self, filename):
        self.init_params()

        self._step_count = 0

        self._action_history = []
        self._state_history = []
        self._state_next_history = []
        self._done_history = []
        self._rewards_history = []

        self._loss = keras.losses.Huber()

        self._model = keras.models.load_model(filename)
        self.create_target()

    def save(self, filename, overwrite=False):
        self._model.save(filename, overwrite=overwrite)

    def init_params(self):
        self.eps = self._params['exp_start']

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
        
    def create(self):
        self.init_params()

        self._step_count = 0

        self._action_history = []
        self._state_history = []
        self._state_next_history = []
        self._done_history = []
        self._rewards_history = []

        self._loss = keras.losses.Huber()
        self._optimizer = keras.optimizers.Adam(learning_rate=self._learning_rate, clipnorm=1.0)

        inputs = layers.Input(shape=(self._state_size,))

        layer = inputs
        for hidden_size in self._hidden_sizes:
            layer = layers.Dense(hidden_size, activation="relu")(layer)
        action = layers.Dense(self._action_size, activation="linear")(layer)

        self._model = Model(inputs=inputs, outputs=action)
        self._model.compile(optimizer="Adam", loss="mse", metrics=["mae"])

        self.create_target()

    def create_target(self):

        self._model_target = keras.models.clone_model(self._model)
        self._model_target.build((self._state_size,))
        self._model_target.compile(optimizer='Adam', loss='mse', metrics=["mae"])
        self._model_target.set_weights(self._model.get_weights())

    def __str__(self):
        return "dqn-agent"


class DDDQNAgent(Agent):

    def init_params(self):
        # environment
        self.num_actions = 5  # hard coded, in handler ExcHandler
        # Reset environment -> done in the handler

        # training
        self._batch_size = self._trn_params['batch_size']
        self.steps = 0  # number of steps ran
        self.learn_every = 10  # interval of steps to fit model
        self._update_every = self._trn_params['update_every']  # interval of steps to update target model
        self._learning_rate = self._trn_params['learning_rate']  # alpha
        self._gamma = self._trn_params['gamma']
        self._eps = self._exp_params['start']
        self._eps_decay = self._exp_params['decay']
        self._eps_end = self._exp_params['end']

        self._tau = self._trn_params['tau']  # TODO Capire se serve
        self._buffer_min_size = self._trn_params['batch_size']
        self._hidden_sizes = self._trn_params['hidden_sizes']
        self._use_gpu = self._trn_params['use_gpu']  # TODO: always true
        self._training = True

        # memory
        self._memory_size = self._trn_params['memory_size']
        # self.memory = deque(maxlen=self._trn_params['memory_size'])  # replay memory
        self.log = []  # stores information from training

    def create(self): #TODO
        self.init_params()

        self._step_count = 0

        self._action_history = []
        self._state_history = []
        self._state_next_history = []
        self._done_history = []
        self._rewards_history = []

        self._loss = keras.losses.Huber()
        self._optimizer = keras.optimizers.Adam(learning_rate=self._learning_rate, clipnorm=1.0)

        # models
        self.q_eval = self.build_network()  # Q eval model
        self.q_target = self.build_network()  # Q target model

    def build_network(self):
        # Build the Dueling DQN Network
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
        A_layer = Dense(self.num_actions, activation='linear', name='Ai')(X_layer)  # A(s,a)
        A_layer = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), name='Ao')(A_layer)  # A(s,a)
        # Q layer (V + A)
        Q = Add(name='Q')([V_layer, A_layer])  # Q(s,a)
        Q_model = Model(inputs=[inputs], outputs=[Q], name='qvalue')
        Q_model.compile(loss='mse', optimizer=self._optimizer)
        return Q_model

    def update_target(self):
            self.q_target.set_weights(self.q_eval.get_weights())

    def act(self, obs):
        # Predict next action based on current state and decay epsilon.
        if self._eps > np.random.rand(1)[0] and self._exploration:  # when training allow random exploration
            if np.random.random() < self._eps:  # get random action
                action = np.random.choice(self._action_size)
            else:  # predict best action
                state_tensor = tf.convert_to_tensor(obs)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = self.q_eval(state_tensor, training=False)
                action = tf.argmax(action_probs[0]).numpy()
        else:  # if not training then always get best action
            state_tensor = tf.convert_to_tensor(obs)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.q_eval(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()
        return action

    def episode_end(self):
        # Decay probability of taking random action
        self._eps = max(self._eps_end, self._eps_decay * self._eps)

    def step(self, obs, action, reward, next_obs, done): #TODO
        self._step_count += 1

        # Save actions and states in replay buffer
        self._action_history.append(action)
        self._state_history.append(obs)
        self._state_next_history.append(next_obs)
        self._done_history.append(done)
        self._rewards_history.append(reward)

        # immagino di essere nel learn
        if self._step_count % self._update_every == 0 and len(self._done_history) > self._buffer_min_size:
            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(self._done_history)), size=self._batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([self._state_history[i] for i in indices])
            state_next_sample = np.array([self._state_next_history[i] for i in indices])
            rewards_sample = [self._rewards_history[i] for i in indices]
            action_sample = [self._action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(self._done_history[i]) for i in indices]
            )

            # TODO update q values
            # Q = self.q_eval.predict(state_sample)  # get Q values for starting states
            # Q_next = self.q_eval.predict(state_next_sample)  # get Q values for ending states
            # Q_target = self.q_target.predict(state_next_sample)  # get Q values from target model
            #
            # for i in range(self._batch_size):
            #     if done:
            #         Q[i][action[i]] = 0.0  # terminal state
            #     else:
            #         a = np.argmax(Q_next[i])  ## a'_max = argmax(Q(s',a'))
            #         Q[i][action[i]] = reward[i] + self._gamma * Q_target[i][a]  # Q_max = Q_target(s',a'_max)
            #     # fit network on batch_size = minibatch_size
            # self.q_eval.fit(state_sample, Q, batch_size=self._batch_size, verbose=0, shuffle=False)


            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = self.q_target.predict(state_next_sample)
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
                q_values = self.q_eval(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = self._loss(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, self.q_eval.trainable_variables)
            self._optimizer.apply_gradients(zip(grads, self.q_eval.trainable_variables))

        if self._step_count % 10000 == 0:  # TODO update_target_network as parameter
            # update the the target network with new weights
            self.update_target()

        # Limit the state and reward history
        if len(self._rewards_history) > self._memory_size:
            del self._rewards_history[:1]
            del self._state_history[:1]
            del self._state_next_history[:1]
            del self._action_history[:1]
            del self._done_history[:1]


    def save(self, filename):
        self.q_eval.save(filename)

    def load(self, filename):
        self.init_params()

        self._step_count = 0

        self._action_history = []
        self._state_history = []
        self._state_next_history = []
        self._done_history = []
        self._rewards_history = []

        self._loss = keras.losses.Huber()

        self.q_eval = keras.models.load_model(filename)
        self.q_target = self.build_network()  # Q target model

    def __str__(self):
        return "dddqn-agent-"


class PPOAgent(Agent):

    def act(self, obs):
        pass

    def step(self, obs, action, reward, next_obs, done):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def __str__(self):
        return "ppo-agent"


class LSTMAgent(Agent):

    #TODO: Organize input_shape
    def lstm_model(self,input_shape):
        self.model = keras.Sequential()

        #2 LSTM layers
        #5 perchè le mosse che posso eseguire sono 5: destra, sinistra, avanti, indietro, fermo
        self.model.add(keras.layers.LSTM(5, input_shape = input_shape, return_sequences= True))
        self.model.add(keras.layers.LSTM(5))

        #dense layer
        self.model.add(keras.layers.Dense(5, activation='relu'))

        #mitigate overfitting
        self.model.add(keras.layers.Dropout(0.3))

        #output layer
        self.model.add(keras.layers.Dense(5, activation='softmax'))


    def act(self, obs):
        if self.eps > np.random.rand(1)[0] and self._exploration:
            action = np.random.choice(self._action_size)
        else:
            state_tensor = tf.convert_to_tensor(obs)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self._model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()
        return action

    def step(self, obs, action, reward, next_obs, done):
        pass

    def save(self, filename):
        self._model.save(filename)

    def load(self, filename):
        pass

    def __str__(self):
        return "lstm-agent"
