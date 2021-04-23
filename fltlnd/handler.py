import random
import matplotlib.pyplot as plt
import numpy as np

from collections import deque
from pathlib import Path

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from flatland.envs.observations import TreeObsForRailEnv

from fltlnd.agent import RandomAgent, NaiveAgent
from fltlnd.obs.utils import split_tree_into_feature_groups, norm_obs_clip 

class ExcHandler:
    def __init__(self, params, training=True, rendering=False, interactive=False):
        self.__env_params = params['env']
        self.__obs_params = params['obs']
        self.__exp_params = params['exp']
        self.__trn_params = params['trn']

        self.__training = training
        self.__interactive = interactive
        self.__rendering = rendering

        self.__obs_handler = ObsHandler(self.__obs_params)
        self.__env_handler = EnvHandler(self.__env_params, self.__obs_handler.builder, self.__rendering)

        # The action space of flatland is 5 discrete actions
        self.__action_size = 5 
        self.__state_size = self.__obs_handler.get_state_size()
        self.__policy = NaiveAgent(self.__state_size, self.__action_size, self.__exp_params, self.__trn_params)

    def start(self, n_episodes):
        # And some variables to keep track of the progress
        # scores_window = deque(maxlen=100)  # todo smooth when rendering instead
        # completion_window = deque(maxlen=100)
        # scores = []
        # completion = []
        # update_values = False

        random.seed(self.__env_params['seed'])
        np.random.seed(self.__env_params['seed'])

        # Max number of steps per episode
        self.max_steps = int(4 * 2 * (self.__env_params['x_dim'] + self.__env_params['y_dim'] + (self.__env_params['n_agents'] / self.__env_params['n_cities']))) 

        self.run_episodes(n_episodes)

    def run_episodes(self, n_episodes):
        for episode_idx in range(n_episodes):
            score = 0
            action_dict = dict()
            action_count = [0] * self.__action_size
            agent_obs = [None] * self.__env_params['n_agents']
            agent_prev_obs = [None] * self.__env_params['n_agents']
            agent_prev_action = [2] * self.__env_params['n_agents']

            # Reset environment
            obs, info = self.__env_handler.reset()

            # Build agent specific observations
            for agent in self.__env_handler.env.get_agent_handles():
                if obs[agent]:
                    agent_obs[agent] = self.__obs_handler.normalize(obs[agent])
                    agent_prev_obs[agent] = agent_obs[agent].copy()


            # Epsilon decay
            eps_start = max(self.__exp_params['end'], self.__exp_params['decay'] * self.__exp_params['start'])

            # Collection information about training TODO: ???
            # tasks_finished = np.sum([int(done[idx]) for idx in env.get_agent_handles()])
            # completion_window.append(tasks_finished / max(1, env.get_num_agents()))
            # scores_window.append(score / (max_steps * env.get_num_agents()))
            # completion.append((np.mean(completion_window)))
            # scores.append(np.mean(scores_window))
            # action_probs = action_count / np.sum(action_count)

            # Run episode
            for step in range(self.max_steps - 1):
                for agent in self.__env_handler.env.get_agent_handles():
                    if info['action_required'][agent]:
                        # If an action is required, we want to store the obs at that step as well as the action
                        update_values = True
                        action = self.__policy.act(agent_obs[agent])
                        action_count[action] += 1
                    else:
                        update_values = False
                        action = 0
                    action_dict.update({agent: action})

                # Environment step
                next_obs, all_rewards, done, info = self.__env_handler.step(action_dict)

                # Update replay buffer and train agent
                for agent in range(self.__env_params['n_agents']):
                    # Only update the values when we are done or when an action was taken and thus relevant information is present
                    if update_values or done[agent]:
                        self.__policy.step(
                            agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent],
                            agent_obs[agent], done[agent]
                        )

                        agent_prev_obs[agent] = agent_obs[agent].copy()
                        agent_prev_action[agent] = action_dict[agent]

                    if next_obs[agent]:
                        agent_obs[agent] = self.__obs_handler.normalize(next_obs[agent])

                    score += all_rewards[agent]

                if done['__all__']:
                    break

                if episode_idx % 100 == 0:
                    end = "\n"
                    self.__policy.save('./checkpoints/' + str(self.__policy) + '-' + str(episode_idx) + '.pth')
                    action_count = [1] * self.__action_size
                else:
                    end = " "

                #print(
                #    '\rTraining {} agents on {}x{}\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
                #        env.get_num_agents(),
                #        env.parameters['x_dim'], env.parameters['y_dim'],
                #        episode_idx,
                #        np.mean(scores_window),
                #        100 * np.mean(completion_window),
                #        parameters['expl']['start'],
                #        action_probs
                #    ), end=end)
            
        # Plot overall training progress at the end
        plt.plot(scores)
        plt.show()

        plt.plot(completion)
        plt.show()

        import keyboard as kb

class EnvHandler:
    def __init__(self, params, obs_builder, rendering=False):
        self.__params = params
        self.__rendering = rendering

        self.env = RailEnv(
            width=self.__params['x_dim'],
            height=self.__params['y_dim'],
            rail_generator=sparse_rail_generator(
                max_num_cities=self.__params['n_cities'],
                seed=self.__params['seed'],
                grid_mode=True,
                max_rails_between_cities=self.__params['max_rails_between_cities'],
                max_rails_in_city=self.__params['max_rails_in_city']
            ),
            schedule_generator=sparse_schedule_generator(),
            number_of_agents=self.__params['n_agents'],
            obs_builder_object=obs_builder
        )

        self.__renderer = RenderTool(self.env)

    def step(self, action_dict):
        #TODO: If interactive mode wait for input 
        next_obs, all_rewards, done, info = self.env.step(action_dict)

        if self.__rendering:
            self.__renderer.render_env(show=True, show_observations=True, show_predictions=False)

        return next_obs, all_rewards, done, info

    def reset(self):
        obs, info = self.env.reset(True, True)
        #TODO: Oppure env.reset(regenerate_rail=True, regenerate_schedule=True)
        
        if self.__rendering:
            self.__renderer.reset()
        return obs, info

        import numpy as np

class ObsHandler:
    def __init__(self, parameters):
        self.parameters = parameters
        self.builder = TreeObsForRailEnv(max_depth=parameters['tree_depth'])

    def get_state_size(self,):
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

