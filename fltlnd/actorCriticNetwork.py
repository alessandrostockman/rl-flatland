import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input


class ActorCriticNetwork(keras.Model):

    def __init__(self, n_actions, state_size, fc1_dims=1024, fc2_dims=512,
                 name='actor_critic', chkpt_dir='tmp/actor_critic'):
        super(ActorCriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self._state_size = state_size
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ac')

        #inputs = Input(shape=(self._state_size,))
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.v = Dense(1, activation=None)
        self.pi = Dense(n_actions, activation='softmax')

    #feed forward
    def call(self, state, training=None, mask=None):
        value = self.fc1(state)
        value = self.fc2(value)

        #value function
        v = self.v(value)
        #policy pi
        pi = self.pi(value)

        return v, pi