from abc import ABC, abstractmethod
import os
import numpy as np
from numpy.core.arrayprint import dtype_short_repr
from numpy.core.numeric import indices
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

import tensorflow_probability as tfp

# added for DDDQN
from tensorflow.keras import layers

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Lambda, Add
from tensorflow.keras import backend as K

from fltlnd.actorCriticNetwork import ActorCriticNetwork
from fltlnd.replay_buffer import ReplayBuffer
from tensorflow.keras.optimizers import Adam

class Agent(ABC):

    def __init__(self, state_size, action_size, params, memory_class, exploration=True, train_best=True, base_dir="",
                 checkpoint=None):
        self._state_size = state_size
        self._action_size = action_size
        self._params = params
        self._exploration = exploration
        self._base_dir = base_dir

        self._step_count = 0

        if checkpoint is not None:
            self.load(checkpoint)
        else:
            self.init_params()

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
    def step(self, obs, action, reward, next_obs, done, agent):
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
    def step_start(self):
        pass

    @abstractmethod
    def episode_start(self):
        pass

    @abstractmethod
    def episode_end(self, agents):
        pass

    @abstractmethod
    def __str__(self):
        pass


class RandomAgent(Agent):

    def act(self, obs):
        self.stats['eps_counter'] += 1
        return np.random.choice(np.arange(self._action_size))

    def step(self, obs, action, reward, next_obs, done, agent):
        pass

    def save(self, filename, overwrite=False):
        pass

    def load(self, filename):
        pass

    def load_best(self):
        self.create()

    def step_start(self):
        pass

    def episode_start(self):
        pass

    def episode_end(self, agents):
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


class NNAgent(Agent):

    def init_params(self):
        self._memory_size = self._params['memory_size']
        self._batch_size = self._params['batch_size']
        self._update_every = self._params['update_every'] #TODO Common?
        self._learning_rate = self._params['learning_rate']
        self._gamma = self._params['gamma']
        self._buffer_min_size = self._params['batch_size']
        self._hidden_sizes = self._params['hidden_sizes']


class DQNAgent(NNAgent):

    def act(self, obs):

        if self.stats['eps_val'] > np.random.rand(1)[0] and self._exploration and not (self.noisy_net):
            action = np.random.choice(self._action_size)
            self.stats['eps_counter'] += 1
        else:
            state_tensor = tf.convert_to_tensor(obs)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self._model.predict_on_batch(state_tensor)
            action = np.argmax(action_probs[0])
        return action

    def step(self, obs, action, reward, next_obs, done, agent):
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

    def save(self, filename, overwrite=True):
        self._model.save(filename, overwrite=overwrite)

    def load(self, filename):
        self._model = keras.models.load_model(filename)

    def create(self):        
        self._model = self.build_network()

    def init_params(self):
        super().init_params()

        self.stats = {
            "eps_val": self._params['exp_start'],
            "eps_counter": 0,
            "loss": None
        }

        self.noisy_net = self._params["noisy_net"]
        self._eps_end = self._params['exp_end']
        self._eps_decay = self._params['exp_decay']
        self._tau = self._params['tau']

        self._loss = keras.losses.Huber()
        self._optimizer = keras.optimizers.Adam(learning_rate=self._learning_rate, clipnorm=1.0)

    def build_network(self):
        inputs = layers.Input(shape=(self._state_size,))

        layer = inputs

        for hidden_size in self._hidden_sizes:
            if (self.noisy_net):
                layer = tfa.layers.NoisyDense(hidden_size, activation='relu')(layer)
            else:
                layer = layers.Dense(hidden_size, activation="relu")(layer)
        action = layers.Dense(self._action_size, activation="linear")(layer)
        model = Model(inputs=inputs, outputs=action)
        model.compile(loss='mse', optimizer=self._optimizer, metrics=["mae"])
        return model

    def step_start(self):
        pass

    def episode_start(self):
        self.stats['eps_counter'] = 0

    def episode_end(self, agents):
        # Decay probability of taking random action
        self.stats['eps_val'] = max(self._eps_end, self._eps_decay * self.stats['eps_val'])

    def _get_future_rewards(self, state_next_sample):
        return self._model.predict_on_batch(state_next_sample)

    def __str__(self):
        return "dqn-agent"


class DoubleDQNAgent(DQNAgent):

    def step(self, obs, action, reward, next_obs, done, agent):
        super().step(obs, action, reward, next_obs, done)
        if self._step_count % self._target_update == 0 or self._soft_update:
            # update the the target network with new weights
            weights = self._model.get_weights()
            target_weights = self._model_target.get_weights()

            if self._soft_update:
                for i in range(len(weights)):
                    target_weights[i] = self._tau * weights[i] + (1 - self._tau) * target_weights[i]

            self._model_target.set_weights(target_weights)

    def load(self, filename):
        super().load(filename)
        self._create_target()

    def create(self):
        super().create()
        self._create_target()

    def init_params(self):
        super().init_params()

        self._tau = self._params['tau']
        self._target_update = self._params['target_update']
        self._soft_update = self._params['soft_update']

    def _create_target(self):
        self._model_target = keras.models.clone_model(self._model)
        self._model_target.build((self._state_size,))
        self._model_target.set_weights(self._model.get_weights())

    def _get_future_rewards(self, state_next_sample):
        return self._model_target.predict_on_batch(state_next_sample)

    def __str__(self):
        return "double-dqn-agent"


class DuelingDQNAgent(DQNAgent):

    def build_network(self):
        inputs = layers.Input(shape=(self._state_size,))

        X_layer = inputs
        for hidden_size in self._hidden_sizes:
            X_layer = layers.Dense(hidden_size, activation="relu")(X_layer)
        # action = layers.Dense(self._action_size, activation="linear")(X_layer) #TODO Parametrize layer sizes
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


class PPOModel(keras.Model):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.num_actions = action_size
        self.inputx = keras.layers.Dense(state_size)
        self.dense1 = keras.layers.Dense(64, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.dense2 = keras.layers.Dense(64, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.value = keras.layers.Dense(1)
        self.policy_logits = keras.layers.Dense(action_size)

    def call(self, inputs):
        x = self.inputx(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.value(x), self.policy_logits(x)

    def action_value(self, state):
        value, logits = self.predict_on_batch(state)
        dist = tfp.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, value

class PPOAgent(NNAgent):

    def __init__(self, state_size, action_size, params, memory_class, exploration, train_best, base_dir, checkpoint):
        super().__init__(state_size, action_size, params, memory_class, exploration=exploration, train_best=train_best, base_dir=base_dir, checkpoint=checkpoint)
        self._last_value = None

    def act(self, obs):
        action, value = self._model.action_value(obs.reshape(1, -1))
        self._last_value = value
        return action.numpy()[0]

    def step(self, obs, action, reward, next_obs, done, agent):
        self._step_count += 1

        _, policy_logits = self._model(obs.reshape(1, -1))

        if self._last_value is None:
            _, self._last_value = self._model.action_value(obs.reshape(1, -1))
        self._memory.add_agent_episode(agent, action, self._last_value[0], obs, reward, done, policy_logits)
        self._last_value = None

    def train(self, agents):
        for agent in agents:
            actions, values, states, rewards, dones, probs = self._memory.retrieve_agent_episodes(agent)

            _, next_value = self._model.action_value(states[-1].reshape(1, -1))
            discounted_rewards, advantages = self._get_advantages(rewards, dones, values, next_value[0])

            actions = tf.squeeze(tf.stack(actions))
            probs = tf.nn.softmax(tf.squeeze(tf.stack(probs)))
            action_inds = tf.stack([tf.range(0, actions.shape[0]), tf.cast(actions, tf.int32)], axis=1)
            
            old_probs = tf.gather_nd(probs, action_inds),
            ent_discount_val = 0.01

            with tf.GradientTape() as tape:
                values, policy_logits = self._model(tf.stack(states))
                act_loss = self._actor_loss(advantages, old_probs, action_inds, policy_logits)
                ent_loss = self._entropy_loss(policy_logits, ent_discount_val)
                c_loss = self._critic_loss(discounted_rewards, values)
                tot_loss = act_loss + ent_loss + c_loss
                self.stats['loss'] = tot_loss

            # Backpropagation
            grads = tape.gradient(tot_loss, self._model.trainable_variables)
            self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

        self._memory.reset()

    def save(self, filename, overwrite=False):
        self._model.save(filename, overwrite=overwrite)

    def load(self, filename):
        self._model = keras.models.load_model(filename)

    def create(self):
        self._model = self._build_network()

    def init_params(self):
        super().init_params()
        
        self.stats = {
            "loss": None,
            "eps_val": 0,
            "eps_counter": 0
        }

        self._learning_rate = self._params['learning_rate']
        self._clipping_loss_ratio = 0.1
        self._entropy_loss_ratio = 0.2
        self._positive_reward = False
        self._target_update_alpha = 0.9

        self.surrogate_eps_clip = 0.1
        self.K_epoch = 10
        self.weight_loss = 0.5
        self.weight_entropy = 0.01

        self._optimizer = Adam(learning_rate=self._learning_rate)

    def step_start(self):
        pass

    def episode_start(self):
        pass

    def episode_end(self, agents):
        self.train(agents)

    def _get_advantages(self, rewards, dones, values, next_value):
        discounted_rewards = np.array(rewards.tolist() + [next_value[0]])

        for t in reversed(range(len(rewards))):
            discounted_rewards[t] = rewards[t] + 0.99 * discounted_rewards[t+1] * (1-dones[t])
        discounted_rewards = discounted_rewards[:-1]
        # advantages are bootstrapped discounted rewards - values, using Bellman's equation
        advantages = discounted_rewards - values
        # standardise advantages
        advantages -= np.mean(advantages)
        advantages /= (np.std(advantages) + 1e-10)
        # standardise rewards too
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= (np.std(discounted_rewards) + 1e-8)
        return discounted_rewards, advantages

    def _build_network(self):
        return PPOModel(self._state_size, self._action_size)

    def _actor_loss(self, advantages, old_probs, action_inds, policy_logits):
        probs = tf.nn.softmax(policy_logits)
        new_probs = tf.gather_nd(probs, action_inds)

        ratio = new_probs / old_probs

        policy_loss = -tf.reduce_mean(tf.math.minimum(
            ratio * advantages,
            tf.clip_by_value(ratio, 1.0 - self.surrogate_eps_clip, 1.0 + self.surrogate_eps_clip) * advantages
        ))
        return policy_loss

    def _critic_loss(self, discounted_rewards, value_est):
        return tf.cast(tf.reduce_mean(keras.losses.mean_squared_error(discounted_rewards, value_est)) * self.weight_loss,
                    tf.float32)


    def _entropy_loss(self, policy_logits, ent_discount_val):
        probs = tf.nn.softmax(policy_logits)
        entropy_loss = -tf.reduce_mean(keras.losses.categorical_crossentropy(probs, probs))
        return entropy_loss * ent_discount_val

    def __str__(self):
        return "ppo"

