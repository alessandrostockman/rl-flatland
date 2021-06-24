from abc import ABC, abstractmethod

import numpy as np

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from fltlnd.predict import StochasticPathPredictor
from fltlnd.utils import split_tree_into_feature_groups, norm_obs_clip


class Observation(ABC):
    def __init__(self, parameters, predictor=None):
        self.parameters = parameters
        self.create(predictor)

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def get_state_size(self):
        pass

    @abstractmethod
    def normalize(self, observation):
        pass


class TreeObs(Observation):
    def create(self, predictor):
        self.builder = TreeObsForRailEnv(max_depth=self.parameters['tree_depth'], predictor=predictor)

    def get_state_size(self):
        # Calculate the state size given the depth of the tree observation and the number of features
        n_features_per_node = self.builder.observation_dim
        n_nodes = 0
        for i in range(self.parameters['tree_depth'] + 1):
            n_nodes += np.power(4, i)
        return n_features_per_node * n_nodes

    def normalize(self, observation):
        data, distance, agent_data = split_tree_into_feature_groups(observation, self.parameters['tree_depth'])

        data = norm_obs_clip(data, fixed_radius=self.parameters['radius'])
        distance = norm_obs_clip(distance, normalize_to_range=True)
        agent_data = np.clip(agent_data, -1, 1)
        normalized_obs = np.concatenate((np.concatenate((data, distance)), agent_data))
        return normalized_obs
