from abc import ABC, abstractmethod

import random
import math
from flatland.envs.rail_trainrun_data_structures import Waypoint

import numpy as np

from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_env_shortest_paths import get_shortest_paths, get_valid_move_actions_
from flatland.utils.ordered_set import OrderedSet

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from fltlnd.utils import split_tree_into_feature_groups, norm_obs_clip


class StochasticPathPredictor(PredictionBuilder):

    def __init__(self, params):
        super().__init__(params['tree_depth'])

    def get(self, handle: int = None):
        """
        Called whenever get_many in the observation build is called.
        Requires distance_map to extract the shortest path.
        Does not take into account future positions of other agents!

        If there is no shortest path, the agent just stands still and stops moving.

        Parameters
        ----------
        handle : int, optional
            Handle of the agent for which to compute the observation vector.

        Returns
        -------
        np.array
            Returns a dictionary indexed by the agent handle and for each agent a vector of (max_depth + 1)x5 elements:
            - time_offset
            - position axis 0
            - position axis 1
            - direction
            - action taken to come here (not implemented yet)
            The prediction at 0 is the current position, direction etc.
        """

        agents = self.env.agents
        if handle:
            agents = [self.env.agents[handle]]
        distance_map: DistanceMap = self.env.distance_map

        paths, distances = self.get_weighted_paths(distance_map, max_depth=self.max_depth)

        prediction_dict = {}
        for agent in agents:
            if agent.status == RailAgentStatus.READY_TO_DEPART:
                agent_virtual_position = agent.initial_position
            elif agent.status == RailAgentStatus.ACTIVE:
                agent_virtual_position = agent.position
            elif agent.status == RailAgentStatus.DONE:
                agent_virtual_position = agent.target
            else:

                prediction = np.zeros(shape=(self.max_depth + 1, 5))
                for i in range(self.max_depth):
                    prediction[i] = [i, None, None, None, None]
                prediction_dict[agent.handle] = prediction
                continue

            agent_virtual_direction = agent.direction
            agent_speed = agent.speed_data["speed"]
            times_per_cell = int(np.reciprocal(agent_speed))
            prediction = np.zeros(shape=(self.max_depth + 1, 5))
            prediction[0] = [0, *agent_virtual_position, agent_virtual_direction, 0]

            agent_paths = paths[agent.handle]
            agent_distances = distances[agent.handle]

            # if there is a shortest path, remove the initial position
            if agent_paths:
                agent_paths = agent_paths[1:]
                agent_distances = agent_distances[1:]

            new_direction = agent_virtual_direction
            new_position = agent_virtual_position
            visited = OrderedSet()
            for index in range(1, self.max_depth + 1):
                # if we're at the target, stop moving until max_depth is reached
                if new_position == agent.target or not agent_paths:
                    prediction[index] = [index, *new_position, new_direction, RailEnvActions.STOP_MOVING]
                    visited.add((*new_position, agent.direction))
                    continue

                d = np.array(agent_distances)
                if d.size > 1:
                    s = d.sum()
                    probs = (s - d) / ((d.size - 1) * s)
                    a = np.random.choice(np.arange(0, probs.size), p=probs)
                else:
                    a = 0

                if index % times_per_cell == 0:
                    try:
                        new_position = agent_paths[a].position
                        new_direction = agent_paths[a].direction
                    except:
                        xxx = 1

                    agent_paths = agent_paths[1:]

                # prediction is ready
                prediction[index] = [index, *new_position, new_direction, 0]
                visited.add((*new_position, new_direction))

            # TODO: very bady side effects for visualization only: hand the dev_pred_dict back instead of setting on env!
            self.env.dev_pred_dict[agent.handle] = visited
            prediction_dict[agent.handle] = prediction

        return prediction_dict

    def get_weighted_paths(self, distance_map, max_depth=None, agent_handle=None):
        """
        Copy of rail_env_shortst_paths.get_shortest_paths
        """
        paths = dict()
        paths_distance = dict()

        def _shortest_path_for_agent(agent):
            if agent.status == RailAgentStatus.READY_TO_DEPART:
                position = agent.initial_position
            elif agent.status == RailAgentStatus.ACTIVE:
                position = agent.position
            elif agent.status == RailAgentStatus.DONE:
                position = agent.target
            else:
                paths[agent.handle] = None
                paths_distance[agent.handle] = None
                return
            direction = agent.direction

            paths[agent.handle] = []
            paths_distance[agent.handle] = []

            distance = math.inf
            depth = 0
            while (position != agent.target and (max_depth is None or depth < max_depth)):
                next_actions = get_valid_move_actions_(direction, position, distance_map.rail)
                best_next_action = None
                for next_action in next_actions:
                    next_action_distance = distance_map.get()[
                        agent.handle, next_action.next_position[0], next_action.next_position[
                            1], next_action.next_direction]
                    if next_action_distance < distance:
                        best_next_action = next_action
                        distance = next_action_distance

                paths[agent.handle].append(Waypoint(position, direction))
                paths_distance[agent.handle].append(next_action_distance)
                depth += 1

                # if there is no way to continue, the rail must be disconnected!
                # (or distance map is incorrect)
                if best_next_action is None:
                    paths[agent.handle] = None
                    paths_distance[agent.handle] = None
                    return

                position = best_next_action.next_position
                direction = best_next_action.next_direction
            if max_depth is None or depth < max_depth:
                paths[agent.handle].append(Waypoint(position, direction))
                paths_distance[agent.handle].append(next_action_distance)

        if agent_handle is not None:
            _shortest_path_for_agent(distance_map.agents[agent_handle])
        else:
            for agent in distance_map.agents:
                _shortest_path_for_agent(agent)

        return paths, paths_distance