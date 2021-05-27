from abc import abstractclassmethod, abstractmethod
from fltlnd.utils import SumTree
import random
from collections import deque
import numpy as np

class Buffer:
    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object."""
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    @abstractmethod
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        pass

    @abstractmethod
    def sample(self):
        """Sample a batch of experiences from memory."""
        pass

    
    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)


class ReplayBuffer(Buffer):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object."""
        super().__init__(buffer_size, batch_size)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.memory.append([state, action, reward, next_state, done])

    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        # sample memory for a minibatch
        mini_batch = random.sample(self.memory, self.batch_size)
        # separate minibatch into elements
        state, action, reward, next_state, done = [np.squeeze(i) for i in zip(*mini_batch)]

        return state, action, reward, next_state, done

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


#TODO: veririficare come ottiene il sample (state, next_state, action, reward, done)
class PrioritizedExperienceReplay(Buffer):
    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object."""
        super().__init__(buffer_size, batch_size)
        self.eta = 0.01
        self.alpha = 0.6
        self.tree = SumTree(batch_size)

    def _getPriority(self, error):
        return (error + self.eta) ** self.alpha

    def add(self, error, sample):
        """Add a new experience to memory."""

        p = self._getPriority(error)
        self.tree.add(p, sample) 

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

    
    
