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

    def __str__(self):
        return "random-agent"

class NaiveAgent(Agent):

    def __init__(self, state_size, action_size, exp_params, trn_params, checkpoint=None):
        self.__state_size = state_size
        self.__action_size = action_size
        self.__trn_params = trn_params

        self.__eps_end = exp_params['end']
        self.__eps_decay = exp_params['decay']
        self.__eps = exp_params['start']
        
        self.__memory_size = trn_params['memory_size']
        self.__batch_size = trn_params['batch_size']
        self.__update_every = trn_params['update_every']
        self.__learning_rate = trn_params['learning_rate']
        self.__tau = trn_params['tau'] # TODO Capire se serve
        self.__gamma = trn_params['gamma']
        self.__buffer_min_size = trn_params['batch_size']
        self.__hidden_size = trn_params['hidden_size'] # TODO Hidden size of the DDDQN???
        self.__use_gpu = trn_params['use_gpu'] # TODO: always true

        self.__frame_count = 0

        self.__action_history = []
        self.__state_history = []
        self.__state_next_history = []
        self.__done_history = []
        self.__rewards_history = []

        self.__loss = keras.losses.Huber()
        self.__optimizer = keras.optimizers.Adam(learning_rate=self.__learning_rate, clipnorm=1.0)

        self.__create()

    def act(self, obs):
        # Decay probability of taking random action
        self.__eps = max(self.__eps_end, self.__eps_decay * self.__eps)

        if self.__eps > np.random.rand(1)[0]:
            action = np.random.choice(self.__action_size)
        else:
            state_tensor = tf.convert_to_tensor(obs)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.__model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()
        return action

    def step(self, obs, action, reward, next_obs, done):
        self.__frame_count += 1
        if self.__frame_count % self.__update_every == 0 and len(self.__done_history) > self.__buffer_min_size:
            # Save actions and states in replay buffer
            self.__action_history.append(action)
            self.__state_history.append(obs)
            self.__state_next_history.append(next_obs)
            self.__done_history.append(done)
            self.__rewards_history.append(reward)

            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(self.__done_history)), size=self.__batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([self.__state_history[i] for i in indices])
            state_next_sample = np.array([self.__state_next_history[i] for i in indices])
            rewards_sample = [self.__rewards_history[i] for i in indices]
            action_sample = [self.__action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(self.__done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = self.__model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + self.__gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, self.__action_size)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = self.__model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = self.__loss(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, self.__model.trainable_variables)
            self.__optimizer.apply_gradients(zip(grads, self.__model.trainable_variables))

        if self.__frame_count % 10000 == 0: #TODO update_target_network as parameter
            # update the the target network with new weights
            self.__model_target.set_weights(self.__model.get_weights())
            # Log details
            #template = "running reward: {:.2f} at episode {}, frame count {}"
            #print(template.format(running_reward, episode_count, frame_count))

        # Limit the state and reward history
        if len(self.__rewards_history) > self.__memory_size:
            del self.__rewards_history[:1]
            del self.__state_history[:1]
            del self.__state_next_history[:1]
            del self.__action_history[:1]
            del self.__done_history[:1]

    def load(self, filename):
        self.__model = keras.models.load_model(filename)

    def save(self, filename):
        self.__model.save(filename)

    def __create(self):
        print(self.__state_size)
        inputs = layers.Input(shape=(self.__state_size,))
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
        self.__model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
        self.__model_target = Model(inputs=inputs, outputs=action)
        self.__model_target.compile(optimizer="Adam", loss="mse", metrics=["mae"])

    def __str__(self):
        return "naive-agent"
