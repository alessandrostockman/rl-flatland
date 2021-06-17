from abc import ABC, abstractmethod
import os
import numpy as np
from numpy.core.arrayprint import dtype_short_repr
from numpy.core.numeric import indices
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

import pickle
import time

# added for DDDQN
from tensorflow.keras import layers
from glob import glob

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Lambda, Add
from tensorflow.keras import backend as K

from fltlnd.actorCriticNetwork import ActorCriticNetwork
from fltlnd.replay_buffer import ReplayBuffer
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.ops import disable_eager_execution


# disable_eager_execution()  # Disable Eager, IMPORTANT!

# TODO: veririficare come inserire il blocco PER della gestione della memoria e come costruire i valori d'errore e passarli alla memoria stessa.
class Agent(ABC):

    def __init__(self, state_size, action_size, params, memory_class, exploration=True, train_best=True, base_dir=""):
        self._state_size = state_size
        self._action_size = action_size
        self._params = params
        self._exploration = exploration
        self._base_dir = base_dir

        if train_best:
            self.load_best()
        else:
            self.create()

        self._memory = memory_class(self._memory_size, self._batch_size)

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
        filename = os.path.join(self._base_dir, 'checkpoints', str(self))

        if os.path.exists(filename):
            self.load(filename)
        else:
            self.create()

    def save_best(self):
        self.save(os.path.join(self._base_dir, 'checkpoints', str(self)))

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def episode_start(self):
        pass

    @abstractmethod
    def episode_end(self):
        pass


class RandomAgent(Agent):

    def act(self, obs):
        self.stats['eps_counter'] += 1
        return np.random.choice(np.arange(self._action_size))

    def step(self, obs, action, reward, next_obs, done):
        pass

    def save(self, filename, overwrite=False):
        pass

    def load(self, filename):
        pass

    def load_best(self):
        self.create()

    def episode_start(self):
        pass

    def episode_end(self):
        pass

    def create(self):
        self.init_params()

    def init_params(self):
        self.stats = {
            "eps_val": 1,
            "eps_counter": 0,
            "loss": 0
        }

        self._memory_size = self._params['memory_size']
        self._batch_size = self._params['batch_size']

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

    def episode_start(self):
        self.stats['eps_counter'] = 0

    def episode_end(self):
        # Decay probability of taking random action
        self.stats['eps_val'] = max(self._eps_end, self._eps_decay * self.stats['eps_val'])

    def step(self, obs, action, reward, next_obs, done):
        self._step_count += 1

        # Save experience in replay memory
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
        self._memory.update(loss)

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

        self._tau = self._params['tau']
        self._target_update = self._params['target_update']
        self._soft_update = self._params['soft_update']

    def create_target(self):
        self._model_target = keras.models.clone_model(self._model)
        self._model_target.build((self._state_size,))
        self._model_target.set_weights(self._model.get_weights())

    def step(self, obs, action, reward, next_obs, done):
        super().step(obs, action, reward, next_obs, done)
        if self._step_count % self._target_update == 0 or self._soft_update:
            # update the the target network with new weights
            weights = self._model.get_weights()
            target_weights = self._model_target.get_weights()

            if self._soft_update:
                for i in range(len(weights)):
                    target_weights[i] = self._tau * weights[i] + (1 - self._tau) * target_weights[i]

            self._model_target.set_weights(target_weights)

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


class ActorCriticAgent(Agent):

    def create(self):
        self.init_params()

        self._step_count = 0

        self._loss = keras.losses.Huber()
        self._optimizer = keras.optimizers.Adam(learning_rate=self._learning_rate, clipnorm=1.0)
        self.actor_critic_model = ActorCriticNetwork(n_actions=self._action_size, state_size=self._state_size)

        self.actor_critic_model.compile(optimizer=self._optimizer)

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

    def episode_start(self):
        self.stats['eps_counter'] = 0

    def episode_end(self):
        # Decay probability of taking random action
        self.stats['eps_val'] = max(self._eps_end, self._eps_decay * self.stats['eps_val'])

    def act(self, obs):
        # if self.stats['eps_val'] > np.random.rand(1)[0] and self._exploration:
        #     action = np.random.choice(self._action_size)
        #     self.stats['eps_counter'] += 1
        # else:
        #     state_tensor = tf.convert_to_tensor([obs,], dtype=tf.float32)
        #     state_tensor = tf.expand_dims(state_tensor, 0)
        #     _, action_probs = self.actor_critic_model(state_tensor, training=False)
        #     # action = tf.argmax(action_probs[0]).numpy() #action scelta nelle act precedenti
        #
        #     action_probabilities = tfp.distributions.Categorical(probs=action_probs)
        #     action = action_probabilities.sample()
        #     log_prob = action_probabilities.log_prob(action)
        #     action = action.numpy()[0]
        #
        # self.action = action
        # return action
        state = tf.convert_to_tensor([obs])
        _, probs = self.actor_critic_model(state)

        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()
        log_prob = action_probabilities.log_prob(action)
        self.action = action

        return action.numpy()[0]

    def step(self, obs, action, reward, next_obs, done):
        self._step_count += 1

        # Save experience in replay memory
        self._memory.add(obs, action, reward, next_obs, done)

        # If enough samples are available in memory, get random subset and learn
        if self._step_count % self._update_every == 0 and len(self._memory) > self._buffer_min_size and len(
                self._memory) > self._batch_size:
            self.train()

    def train(self):
        # Get samples from replay buffer
        state_sample, action_sample, rewards_sample, state_next_sample, done_sample = self._memory.sample()
        state = tf.convert_to_tensor([state_sample], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_next_sample], dtype=tf.float32)
        reward = tf.convert_to_tensor(rewards_sample, dtype=tf.float32)  # not fed to NN
        with tf.GradientTape(persistent=True) as tape:
            state_value, probs = self.actor_critic_model(state)
            state_value_, _ = self.actor_critic_model(state_)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)
            # done = tf.squeeze(done_sample)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(self.action)

            # Q value = reward + discount factor * expected future reward
            # updated_q_values = rewards_sample + self._gamma * tf.reduce_max(
            #     future_rewards, axis=1)
            # # If final frame set the last value to -1
            # updated_q_values = updated_q_values * (1 - done_sample) - done_sample
            # delta = reward + self.gamma*state_value_*(1-int(done)) - state_value

            delta = reward + self._gamma * state_value_ * (1 - done_sample) - state_value
            actor_loss = -log_prob * delta
            critic_loss = delta ** 2
            total_loss = actor_loss + critic_loss

        gradient = tape.gradient(total_loss, self.actor_critic_model.trainable_variables)
        self.actor_critic_model.optimizer.apply_gradients(zip(
            gradient, self.actor_critic_model.trainable_variables))

    def save(self, filename, overwrite=False):
        self.actor_critic_model.save(filename, overwrite=overwrite)

    def load(self, filename):
        self.init_params()

        self._step_count = 0

        self._loss = keras.losses.Huber()
        self._optimizer = keras.optimizers.Adam(learning_rate=self._learning_rate, clipnorm=1.0)

        self.actor_critic_model = keras.models.load_model(filename)

        self.critic_grads = tf.gradients(self.actor_critic_model.output, self.critic_action_input)

        # # Initialize for later gradient calculations

    # self.sess.run(tf.initialize_all_variables())

    def __str__(self):
        return "actorcritic-agent"


class AltDDDQNAgent(Agent):
    def episode_start(self):
        self.stats['eps_counter'] = 0

class ACAgent(Agent):
    def create(self):
        self.init_params()

        self._step_count = 0

        self._loss = keras.losses.Huber()
        self._optimizer = keras.optimizers.Adam(learning_rate=self._learning_rate, clipnorm=1.0)

        self._actor, self._critic, self._policy = self.build_network()

    def init_params(self):
        self.stats = {
            "eps_val": self._params['exp_start'],
            "eps_counter": 0,
            "loss": None,
            "time_train": None,
            "time_fit_actor": None,
            "time_fit_critic": None,
            "time_act": None
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

        # self._fc1_dims = 1024
        # self._fc2_dims = 512
        self._fc1_dims = 128

    def build_network(self):
        input = Input(shape=(self._state_size,))
        delta = Input(shape=[1])
        dense1 = Dense(self._fc1_dims, activation='relu')(input)
        # dense2 = Dense(self._fc2_dims, activation='relu')(dense1)
        # probs = Dense(self._action_size, activation='softmax')(dense2)
        
        probs = Dense(self._action_size, activation='softmax')(dense1)
        values = Dense(1, activation='linear')(dense1)

        actor = Model(inputs=[input, delta], outputs=[probs])

        actor.compile(optimizer=Adam(learning_rate=self._learning_rate), loss=self._loss)

        critic = Model(inputs=[input], outputs=[values])

        critic.compile(optimizer=Adam(learning_rate=self._learning_rate), loss='mean_squared_error')

        policy = Model(inputs=[input], outputs=[probs])

        return actor, critic, policy

    def act(self, obs):
        start_act = time.time()
        state = obs[np.newaxis, :]
        probabilities = self._policy.predict(state)[0]
        action = np.random.choice(self._action_size, p=probabilities)

        self.stats['time_act'] += time.time() - start_act
        return action

    def step(self, obs, action, reward, next_obs, done):
        self._step_count += 1

        # Save experience in replay memory
        self._memory.add(obs, action, reward, next_obs, done)

        self.train()

    def train(self):
        train_start = time.time()
        state, action, reward, state_, done = self._memory.get_last()
        state = state[np.newaxis, :]
        state_ = state_[np.newaxis, :]
        critic_value_ = self._critic.predict(state_)
        critic_value = self._critic.predict(state)

        target = reward + self._gamma * critic_value_ * (1 - int(done))
        delta = target - critic_value

        actions = np.zeros([1, self._action_size])
        actions[np.arange(1), action] = 1

        fit_start = time.time()
        self._actor.fit([state, delta], actions, verbose=0)
        self.stats['time_fit_actor'] += time.time() - fit_start

        fit_start = time.time()
        self._critic.fit(state, target, verbose=0)
        self.stats['time_fit_critic'] += time.time() - fit_start

        self.stats['time_train'] += time.time() - train_start

    def save(self, filename, overwrite=False):
        self._actor.save(os.path.join(filename, "actor"), overwrite=overwrite)
        self._critic.save(os.path.join(filename, "ctitic"), overwrite=overwrite)
        self._policy.save(os.path.join(filename, "policy"), overwrite=overwrite)

    def load(self, filename):
        self.init_params()

        self._step_count = 0

        self._loss = keras.losses.Huber()
        self._optimizer = keras.optimizers.Adam(learning_rate=self._learning_rate, clipnorm=1.0)

        self._actor = keras.models.load_model(os.path.join(filename, "actor"))
        self._critic = keras.models.load_model(os.path.join(filename, "critic"))
        self._policy = keras.models.load_model(os.path.join(filename, "policy"))

    def __str__(self):
        return "ac-agent"

    def episode_start(self):
        self.stats['eps_counter'] = 0
        self.stats['time_act'] = 0
        self.stats['time_train'] = 0
        self.stats['time_fit_actor'] = 0
        self.stats['time_fit_critic'] = 0

    def episode_end(self):
        # Decay probability of taking random action
        self.stats['eps_val'] = max(self._eps_end, self._eps_decay * self.stats['eps_val'])
