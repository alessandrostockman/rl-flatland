import numpy as np

from flatland.envs.observations import TreeObsForRailEnv
from observation.utils import split_tree_into_feature_groups, norm_obs_clip 

class ObservationHandler:
    def __init__(self, parameters):
        self.parameters = parameters
        self.builder = TreeObsForRailEnv(max_depth=parameters['tree_depth'])

    def get_state_size(self, env):
        # Calculate the state size given the depth of the tree observation and the number of features
        n_features_per_node = env.obs_builder.observation_dim #TODO: Check if env.obs_builder == self.builder
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