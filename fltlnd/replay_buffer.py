from abc import abstractclassmethod, abstractmethod
import random
from collections import deque

import numpy as np

from fltlnd.utils import SumTree

class Buffer:

    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    @abstractmethod
    def add(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def update(self, idx, error):
        pass

    @abstractmethod
    def add_agent_episode(self, agent, action, value, obs, reward, done, policy_logits):
        pass

    @abstractmethod
    def retrieve_agent_episodes(self, agent):
        pass

    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def __len__(self):
        pass


class ReplayBuffer(Buffer):
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, buffer_size, batch_size):
        super().__init__(buffer_size, batch_size)
        self._batch_size = buffer_size
        self.memory = deque(maxlen=self._batch_size)
        self.has_probability = False

    def add(self, state, action, reward, next_state, done, probability=None):
        """Add a new experience to memory."""
        self.memory.append([state, action, reward, next_state, done, probability])
        if probability is not None:
            self.has_probability = True

    def get_last(self):
        return self.memory.__getitem__(self.memory.__len__()-1)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        # sample memory for a minibatch
        batch = random.sample(self.memory, self.batch_size)
        # separate minibatch into elements

        state, action, reward, next_state, done, probability = [np.squeeze(i) for i in zip(*batch)]

        if self.has_probability:
            return state, action, reward, next_state, done, probability
        else:
            return state, action, reward, next_state, done
        
    def update(self, error):
        pass
    

    def add_agent_episode(self, agent, action, value, obs, reward, done, policy_logits):
        raise NotImplementedError()
    
    def retrieve_agent_episodes(self, agent):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class PrioritizedBuffer(Buffer):
    def __init__(self, buffer_size, batch_size):
        super().__init__(buffer_size, batch_size)

        self._internal_len = 0
        self.eta = 0.01
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_growth = 0.001

        self._batch_size = batch_size
        self.tree = SumTree(batch_size)

    def _get_priority(self, error):
        return (error + self.eta) ** self.alpha

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        sample = [state, action, reward, next_state, done]
          # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this experience will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = 1

        self._internal_len += 1
        self.tree.add(max_priority, sample) 

    def sample(self):
        batch = []
        idxs = []
        segment = self.tree.total() / self.batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_growth])

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        self.sample_ids = idxs
        state, action, reward, next_state, done = [np.squeeze(i) for i in zip(*batch)]
        return state, action, reward, next_state, done

    def update(self, error):
        p = self._get_priority(error)

        for idx in self.sample_ids:
            self.tree.update(idx, p)

    def add_agent_episode(self, agent, action, value, obs, reward, done, policy_logits):
        raise NotImplementedError()
    
    def retrieve_agent_episodes(self, agent):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def __len__(self):
        return self._internal_len
        

class AgentEpisodeBuffer(Buffer):

    def __init__(self, buffer_size, batch_size):
        self._memory = {}

    def add(self, state, action, reward, next_state, done):
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()

    def update(self, idx, error):
        raise NotImplementedError()

    def add_agent_episode(self, agent, action, value, obs, reward, done, policy_logits):
        agent_mem = self._memory.get(agent, [])
        agent_mem.append([action, value, obs, reward, done, policy_logits])
        self._memory[agent] = agent_mem
    
    def retrieve_agent_episodes(self, agent):
        action, value, obs, reward, done, policy_logits = [np.squeeze(i) for i in zip(*self._memory[agent])]
        return action, value, obs, reward, done, policy_logits

    def reset(self):
        self._memory = {}
    
    def __len__(self):
        pass