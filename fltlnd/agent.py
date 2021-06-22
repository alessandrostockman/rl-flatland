from abc import ABC, abstractmethod
import os
import numpy as np
from numpy.core.arrayprint import dtype_short_repr
from numpy.core.numeric import indices
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

# added for DDDQN
from tensorflow.keras import layers

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Lambda, Add
from tensorflow.keras import backend as K

from fltlnd.actorCriticNetwork import ActorCriticNetwork
from fltlnd.replay_buffer import ReplayBuffer
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.ops import disable_eager_execution


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
    def step_start(self):
        pass

    @abstractmethod
    def episode_start(self):
        pass

    @abstractmethod
    def episode_end(self):
        pass

    @abstractmethod
    def __str__(self):
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

    def step_start(self):
        pass

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

    def episode_end(self):
        # Decay probability of taking random action
        self.stats['eps_val'] = max(self._eps_end, self._eps_decay * self.stats['eps_val'])

    def _get_future_rewards(self, state_next_sample):
        return self._model.predict_on_batch(state_next_sample)

    def __str__(self):
        return "dqn-agent"


class DoubleDQNAgent(DQNAgent):

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


class ActorCriticAgent(NNAgent):

    def act(self, obs):
        state = obs[np.newaxis, :]
        probabilities = self._policy.predict(state)[0]
        action = np.random.choice(self._action_size, p=probabilities)
        return action

    def step(self, obs, action, reward, next_obs, done):
        pass

    def train(self):
        pass

    def load(self, filename):
        self.init_params()

        self._step_count = 0

        self._loss = None #TODO custom
        self._optimizer = keras.optimizers.Adam(learning_rate=self._learning_rate, clipnorm=1.0)

        self._model = keras.models.load_model(filename)

    def save(self, filename, overwrite=True):
        self._actor_model.save(os.path.join(filename, 'actor'), overwrite=overwrite)
        self._critic_model.save(os.path.join(filename, 'critic'), overwrite=overwrite)

    def init_params(self):
        super().init_params()

        self.stats = {
            "loss": None
        }

    def create(self):
        self.init_params()

        self._step_count = 0

        self._actor_optimizer = keras.optimizers.Adam(learning_rate=self._learning_rate, clipnorm=1.0)
        self._critic_optimizer = keras.optimizers.Adam(learning_rate=self._learning_rate, clipnorm=1.0)

        self._actor_model, self._critic_model, self._policy_model = self.build_network()

    def build_network(self):
        input = Input(shape=(self._state_size,))
        delta = Input(shape=[1])
        dense1 = Dense(self._fc1_dims, activation='relu')(input)
        # dense2 = Dense(self._fc2_dims, activation='relu')(dense1)
        # probs = Dense(self._action_size, activation='softmax')(dense2)

        probs = Dense(self._action_size, activation='softmax')(dense1)
        values = Dense(1, activation='linear')(dense1)

        actor = Model(inputs=[input, delta], outputs=[probs])
        actor.compile(optimizer=self._actor_optimizer, loss=self._actor_loss) #TODO

        critic = Model(inputs=[input], outputs=[values])
        critic.compile(optimizer=self._critic_optimizer, loss='mse')

        policy = Model(inputs=[input], outputs=[probs])

        return actor, critic, policy

    def step_start(self):
        pass

    def episode_start(self):
        self.stats['eps_counter'] = 0

    def episode_end(self):
        # Decay probability of taking random action
        self.stats['eps_val'] = max(self._eps_end, self._eps_decay * self.stats['eps_val'])

    def _get_future_rewards(self, state_next_sample): #TODO Serve?
        return self._model.predict_on_batch(state_next_sample)

    def __str__(self):
        return "ac-agent"


class ACCustomAgent(Agent):

    def create(self):
        self.init_params()

        self._step_count = 0

        self._loss = keras.losses.Huber()
        self._optimizer = keras.optimizers.Adam(learning_rate=self._learning_rate, clipnorm=1.0)
        self.actor_critic_model = ActorCriticNetwork(n_actions=self._action_size, state_size=self._state_size)

        self.actor_critic_model.compile(optimizer=self._optimizer)

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
        state = tf.convert_to_tensor([obs])
        _, probs = self.actor_critic_model(state)

        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()
        if action == 5:
            a = 1
        log_prob = action_probabilities.log_prob(action)
        self.action = action

        return action.numpy()[0]

    def step(self, obs, action, reward, next_obs, done):
        self._step_count += 1

        # Save experience in replay memory
        self._memory.add(obs, action, reward, next_obs, done)

        self.train()

    def train(self):
        # Get samples from replay buffer
        state_sample, action_sample, rewards_sample, state_next_sample, done_sample = self._memory.get_last()
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

            delta = reward + self._gamma * state_value_ * (1 - int(done_sample)) - state_value
            actor_loss = -log_prob * delta
            critic_loss = delta ** 2
            total_loss = actor_loss + critic_loss

        gradient = tape.gradient(total_loss, self.actor_critic_model.trainable_variables)
        self.actor_critic_model.optimizer.apply_gradients(zip(
            gradient, self.actor_critic_model.trainable_variables))

    def save(self, filename, overwrite=True):
        self.actor_critic_model.save(filename, overwrite=overwrite)

    def load(self, filename):
        self.init_params()

        self._step_count = 0

        self._loss = keras.losses.Huber()
        self._optimizer = keras.optimizers.Adam(learning_rate=self._learning_rate, clipnorm=1.0)

        self.actor_critic_model = keras.models.load_model(filename)

    def __str__(self):
        return "ACCustom-agent"

    def step_start(self):
        pass


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
        # AC choose action
        state = obs[np.newaxis, :]
        probabilities = self._policy.predict(state)[0]
        action = np.random.choice(self._action_size, p=probabilities)

        # keras choose action
        # Sample action from action probability distribution
        # action = np.random.choice(num_actions, p=np.squeeze(action_probs))

        # DQN choose action
        # state_tensor = tf.convert_to_tensor(obs)
        # state_tensor = tf.expand_dims(state_tensor, 0)
        # action_probs = self._policy(state_tensor, training=False)
        # # action = tf.argmax(action_probs[0]).numpy()
        # action = np.random.choice(self._action_size, p=np.squeeze(action_probs))

        return action

    def step(self, obs, action, reward, next_obs, done):
        self._step_count += 1

        # Save experience in replay memory
        self._memory.add(obs, action, reward, next_obs, done)

        self.train()

    def train(self):
        state, action, reward, state_, done = self._memory.get_last()
        state = state[np.newaxis, :]
        state_ = state_[np.newaxis, :]
        critic_value_ = self._critic.predict(state_)
        critic_value = self._critic.predict(state)

        target = reward + self._gamma * critic_value_ * (1 - int(done))
        delta = target - critic_value

        actions = np.zeros([1, self._action_size])
        actions[np.arange(1), action] = 1

        self._actor.fit([state, delta], actions, verbose=0)

        self._critic.fit(state, target, verbose=0)

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

    def step_start(self):
        pass

    def episode_start(self):
        self.stats['eps_counter'] = 0

    def episode_end(self):
        # Decay probability of taking random action
        self.stats['eps_val'] = max(self._eps_end, self._eps_decay * self.stats['eps_val'])


class ACKerasAgent(Agent):
    def step_start(self):
        pass

    def create(self):
        self.init_params()

        self._step_count = 0

        self._loss = keras.losses.Huber()
        self._optimizer = keras.optimizers.Adam(learning_rate=self._learning_rate, clipnorm=1.0)

        self._action_probs_history = []
        self._critic_value_history = []
        self._rewards_history = []
        self._running_reward = 0

        self._eps_keras = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

        self._model = self.build_network()

    def build_network(self):
        inputs = layers.Input(shape=(self._state_size,))
        common = layers.Dense(128, activation="relu")(inputs)
        action = layers.Dense(self._action_size, activation="softmax")(common)
        critic = layers.Dense(1)(common)

        model = keras.Model(inputs=inputs, outputs=[action, critic])
        return model

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

    def act(self, obs):
        state = tf.convert_to_tensor(obs)
        state = tf.expand_dims(state, 0)

        # Predict action probabilities and estimated future rewards
        # from environment state
        action_probs, critic_value = self._model(state)
        self._critic_value_history.append(critic_value[0, 0])

        # Sample action from action probability distribution
        if np.any(tf.math.is_nan(action_probs)):
            a = 1
        action = np.random.choice(self._action_size, p=np.squeeze(action_probs))
        self._action_probs_history.append(tf.math.log(action_probs[0, action]))

        return action

    def step(self, obs, action, reward, next_obs, done):
        self._step_count += 1

        # Save experience in replay memory
        self._memory.add(obs, action, reward, next_obs, done)

        self.train()

    def train(self):
        state, action, reward, state_, done = self._memory.get_last()
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)
        state_ = tf.convert_to_tensor(state_)
        state_ = tf.expand_dims(state_, 0)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)  # not fed to NN
        with tf.GradientTape(persistent=True) as tape:
            probs, state_value = self._model(state)
            _, state_value_ = self._model(state_)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(action)

            delta = reward + self._gamma * state_value_ * (1 - int(done)) - state_value
            actor_loss = -log_prob * delta
            critic_loss = delta ** 2
            total_loss = actor_loss + critic_loss

        gradient = tape.gradient(total_loss, self._model.trainable_variables)
        if np.any(tf.math.is_nan(total_loss)):
            a = 1
        self.g = gradient
        self._optimizer.apply_gradients(zip(gradient, self._model.trainable_variables))

    def save(self, filename, overwrite=False):
        self._model.save(filename, overwrite=overwrite)

    def load(self, filename):
        self.init_params()

        self._step_count = 0

        self._loss = keras.losses.Huber()
        self._optimizer = keras.optimizers.Adam(learning_rate=self._learning_rate, clipnorm=1.0)

        self._model = keras.models.load_model(filename)

    def __str__(self):
        return "acKeras-agent"

    def episode_start(self):
        self.stats['eps_counter'] = 0

    def episode_end(self):
        # Decay probability of taking random action
        self.stats['eps_val'] = max(self._eps_end, self._eps_decay * self.stats['eps_val'])


class PPOAgent(NNAgent):
    def init_params(self):
        super().init_params()
        
        tf.compat.v1.disable_eager_execution()
        self.stats = {
            "eps_val": self._params['exp_start'],
            "eps_counter": 0,
            "loss": None
        }

        # specific actor critic PPO
        self._actor_learnig_rate = self._params['learning_rate']
        self._critic_learning_rate = self._params['learning_rate']
        self._clipping_loss_ratio = 0.1
        self._entropy_loss_ratio = 0.2
        self._positive_reward = False
        self._target_update_alpha = 0.9

    def act(self, obs):
        prob = self.actor_network.predict_on_batch([obs, self.dummy_advantage, self.dummy_old_prediction]).flatten()
        action = np.random.choice(self._action_size, p=prob)
        return action

    def step(self, obs, action, reward, next_obs, done):
        #TODO actor evaluation in batch ???
        self._step_count += 1

        # Save experience in replay memory
        self._memory.add(obs, action, reward, next_obs, done)

        # If enough samples are available in memory, get random subset and learn
        #TODO Remove if ???
        if self._step_count % self._update_every == 0 and len(self._memory) > self._buffer_min_size and len(
                self._memory) > self._batch_size:
            self.train()

    def train(self):
        n = self._batch_size
        discounted_r = []

        last_state, last_action, last_reward, last_state_, last_done = self._memory.get_last()
        batch_state, batch_action, batch_reward, batch_state_, batch_done = self._memory.sample()
        if last_done:
            v = 0
        else:
            v = self.get_v(last_state_)
        for r in batch_reward[::-1]:
            v = r + self._gamma * v
            discounted_r.append(v)
        discounted_r.reverse()

        batch_s, batch_a, batch_discounted_r = np.vstack(batch_state), \
                                               np.vstack(batch_action), \
                                               np.vstack(discounted_r)

        batch_v = self.get_v(batch_s)
        batch_advantage = batch_discounted_r - batch_v
        batch_old_prediction = self.get_old_prediction(batch_s, batch_advantage)

        batch_a_final = np.zeros(shape=(len(batch_a), self._action_size))
        batch_a_final[:, batch_a.flatten()] = 1
        # print(batch_s.shape, batch_advantage.shape, batch_old_prediction.shape, batch_a_final.shape)
        self.actor_network.fit(x=[batch_s, batch_advantage, batch_old_prediction], y=batch_a_final, verbose=0)
        self.critic_network.fit(x=batch_s, y=batch_discounted_r, epochs=2, verbose=0)
        #self._memory.clear()
        self.update_target_network()

    def create(self):
        self.init_params()

        self._step_count = 0

        self.actor_network = self._build_actor_network()
        self.actor_old_network = self.build_network_from_copy(self.actor_network)

        self.critic_network = self._build_critic_network()

        self.dummy_advantage = np.zeros((1, 1))
        self.dummy_old_prediction = np.zeros((1, self._action_size))

        #TODO Custom loss and optimizer initialization ???

    def _build_actor_network(self):
        state = Input(shape=(self._state_size,), name="state")

        advantage = Input(shape=(1,), name="Advantage")
        old_prediction = Input(shape=(self._action_size,), name="Old_Prediction")

        shared_hidden = self._shared_network_structure(state)
        action_dim = self._action_size

        policy = Dense(action_dim, activation="softmax", name="actor_output_layer")(shared_hidden)

        actor_network = Model(inputs=[state, advantage, old_prediction], outputs=policy)

        actor_network.compile(optimizer=Adam(learning_rate=self._actor_learnig_rate),
                              loss=self.proximal_policy_optimization_loss(
                                  advantage=advantage, old_prediction=old_prediction
                              ))

        return actor_network

    def _build_critic_network(self):
        state = Input(shape=(self._state_size,), name="state")
        shared_hidden = self._shared_network_structure(state)

        if self._positive_reward:
            q = Dense(1, activation="relu", name="critic_output_layer")(shared_hidden)
        else:
            q = Dense(1, name="critic_output_layer")(shared_hidden)

        critic_network = Model(inputs=state, outputs=q)

        critic_network.compile(optimizer=Adam(learning_rate=self._critic_learning_rate),
                               loss="mean_squared_error")
        return critic_network

    def build_network_from_copy(self, actor_network):
        network = keras.models.clone_model(actor_network)
        network.build((self._state_size,))
        network.set_weights(actor_network.get_weights())
        network.compile(optimizer=Adam(learning_rate=self._actor_learnig_rate), loss="mse")
        return network

    def _shared_network_structure(self, state_features):
        dense_d = 32
        hidden1 = Dense(dense_d, activation="relu", name="hidden_shared_1")(state_features)
        hidden2 = Dense(dense_d, activation="relu", name="hidden_shared_2")(hidden1)
        return hidden2

    def proximal_policy_optimization_loss(self, advantage, old_prediction):
        loss_clipping = self._clipping_loss_ratio
        entropy_loss = self._entropy_loss_ratio

        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            r = prob / (old_prob + 1e-10)
            return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - loss_clipping,
                                                           max_value=1 + loss_clipping) * advantage) + entropy_loss * (
                                   prob * K.log(prob + 1e-10)))

        return loss

    def get_v(self, s):
        s = np.reshape(s, (-1, self._state_size))
        v = self.critic_network.predict_on_batch(s)
        return v

    def get_old_prediction(self, batch_s, batch_advantage):
        batch_s = np.reshape(batch_s, (-1, self._state_size))
        #batch_advantage = np.reshape(batch_advantage, (-1, ))
        dummy_batch_prediction = np.zeros((32, self._action_size))

        return self.actor_old_network.predict_on_batch([batch_s, batch_advantage, dummy_batch_prediction])

    def update_target_network(self):
        alpha = self._target_update_alpha
        self.actor_old_network.set_weights(alpha * np.array(self.actor_network.get_weights())
                                           + (1 - alpha) * np.array(self.actor_old_network.get_weights()))

    def save(self, filename, overwrite=False):
        self._actor_model.save(os.path.join(filename, 'actor'), overwrite=overwrite)
        self._critic_model.save(os.path.join(filename, 'critic'), overwrite=overwrite)

    def load(self, filename):
        self.init_params()

        self._step_count = 0

        self._loss = keras.losses.Huber()
        self._optimizer = keras.optimizers.Adam(learning_rate=self._learning_rate, clipnorm=1.0)

        self.actor_network = keras.models.load_model("%s_actor_network.h5" % filename)
        self.critic_network = keras.models.load_model("%s_critic_network.h5" % filename)

    def step_start(self):
        pass

    def episode_start(self):
        pass

    def episode_end(self):
        pass

    def __str__(self):
        return "ppo-agent"









