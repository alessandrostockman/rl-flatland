from random import sample
from collections import deque

import numpy as np

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object."""
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.memory.append([state, action, reward, next_state, done])

    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        # sample memory for a minibatch
        mini_batch = sample(self.memory, self.batch_size)
        # separate minibatch into elements
        state, action, reward, next_state, done = [np.squeeze(i) for i in zip(*mini_batch)]

        return state, action, reward, next_state, done

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

