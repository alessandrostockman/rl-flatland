import random
import matplotlib.pyplot as plt
import numpy as np

from collections import deque
from pathlib import Path

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool

import fltlnd.agent as agent_classes
import fltlnd.obs as obs_classes


class ExcHandler:
    def __init__(self, params, training=True, rendering=False, interactive=False, checkpoint=None, logging=None):
        self._env_params = params['env']  # Environment
        self._obs_params = params['obs']  # Observation
        self._exp_params = params['exp']  # Exploration
        self._trn_params = params['trn']  # Training
        self._pol_params = params['pol']  # Policy

        self._training = training
        self._interactive = interactive
        self._rendering = rendering

        obs_class = getattr(obs_classes, self._obs_params['class'])
        self._obs_wrapper = obs_class(self._obs_params)
        self._env_handler = EnvHandler(self._env_params, self._obs_wrapper.builder, self._rendering)

        # The action space of flatland is 5 discrete actions
        self._action_size = 5
        self._state_size = self._obs_wrapper.get_state_size()

        # create agent Agent per policy
        policy_class = getattr(agent_classes, self._pol_params['class'])
        self._policy = policy_class(self._state_size, self._action_size, self._exp_params, self._trn_params, 
            self._pol_params, checkpoint=checkpoint, exploration=training, logging=logging)

        # variables to keep track of the progress
        self._scores_window = deque(maxlen=100)  # todo smooth when rendering instead
        self._completion_window = deque(maxlen=100)
        self._scores = []
        self._completion = []
        self._update_values = False

        # Max number of steps per episode
        self._max_steps = int(4 * 2 * (self._env_params['x_dim'] + self._env_params['y_dim'] + (
                self._env_params['n_agents'] / self._env_params['n_cities'])))

    def start(self, n_episodes):

        random.seed(self._env_params['seed'])
        np.random.seed(self._env_params['seed'])

        self.run_episodes(n_episodes)

    def run_episodes(self, n_episodes):
        for episode_idx in range(n_episodes):
            score = 0
            action_dict = dict()
            action_count = [0] * self._action_size
            agent_obs = [None] * self._env_params['n_agents']
            agent_prev_obs = [None] * self._env_params['n_agents']
            agent_prev_action = [2] * self._env_params['n_agents']

            # Reset environment
            obs, info = self._env_handler.reset()

            # Build agent specific observations
            for agent in self._env_handler.env.get_agent_handles():
                if obs[agent]:
                    agent_obs[agent] = self._obs_wrapper.normalize(obs[agent])
                    agent_prev_obs[agent] = agent_obs[agent].copy()
            count_steps = 0
            # Run episode
            for step in range(self._max_steps - 1):
                count_steps += 1
                for agent in self._env_handler.env.get_agent_handles():
                    if info['action_required'][agent]:
                        # If an action is required, we want to store the obs at that step as well as the action
                        update_values = True
                        action = self._policy.act(agent_obs[agent])
                        action_count[action] += 1
                    else:
                        update_values = False
                        action = 0
                    action_dict.update({agent: action})

                # Environment step
                next_obs, all_rewards, done, info = self._env_handler.step(action_dict)

                # Update replay buffer and train agent
                for agent in range(self._env_params['n_agents']):
                    # Only update the values when we are done or when an action was taken and thus relevant information is present
                    if self._training and (update_values or done[agent]):
                        #TODO: agent_obs instead of agent_next_obs??
                        self._policy.step(
                            agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent],
                            agent_obs[agent], done[agent]
                        )

                        agent_prev_obs[agent] = agent_obs[agent].copy()
                        agent_prev_action[agent] = action_dict[agent]

                    if next_obs[agent]:
                        agent_obs[agent] = self._obs_wrapper.normalize(next_obs[agent])

                    score += all_rewards[agent]

                if done['__all__']:
                    break

            self._policy.episode_end()

            # Collection information about training TODO: ???
            tasks_finished = np.sum([int(done[idx]) for idx in self._env_handler.env.get_agent_handles()])
            self._completion_window.append(tasks_finished / max(1, self._env_handler.env.get_num_agents()))
            self._scores_window.append(score / (self._max_steps * self._env_handler.env.get_num_agents()))
            self._completion.append((np.mean(self._completion_window)))
            self._scores.append(np.mean(self._scores_window))
            action_probs = action_count / np.sum(action_count)
            if episode_idx % self._pol_params['checkpoint_freq'] == 0:
                end = "\n"
                action_count = [1] * self._action_size

                if self._training:
                    self._policy.save(self._pol_params['base_dir'] + 'checkpoints/' + str(self._policy) + '-' + str(
                        episode_idx) + '.pth/')
            else:
                end = " "

            #print("\n", done, "\n", self._policy.eps)
            #print(count_steps)
            # print training agent status (new function)
            self._env_handler.print_results(episode_idx, self._scores_window, self._completion_window,
                                             action_probs, end)

        # Plot overall training progress at the end
        plt.plot(self._scores)
        plt.title("Scores")
        plt.show()

        plt.plot(self._completion)
        plt.title("Completions")
        plt.show()


class EnvHandler:
    def __init__(self, params, obs_builder, rendering=False):
        self._params = params
        self._rendering = rendering

        self.env = RailEnv(
            width=self._params['x_dim'],
            height=self._params['y_dim'],
            rail_generator=sparse_rail_generator(
                max_num_cities=self._params['n_cities'],
                seed=self._params['seed'],
                grid_mode=True,
                max_rails_between_cities=self._params['max_rails_between_cities'],
                max_rails_in_city=self._params['max_rails_in_city']
            ),
            schedule_generator=sparse_schedule_generator(),
            number_of_agents=self._params['n_agents'],
            obs_builder_object=obs_builder
        )

        self._renderer = RenderTool(self.env)

    def print_results(self, episode_idx, scores_window, completion_window, action_probs, end):
        print(
            '\rTraining {} agents on {}x{}\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\t '
            'Action Probabilities: \t {}'.format(
                self.env.get_num_agents(),
                self._params['x_dim'], self._params['y_dim'],
                episode_idx,
                np.mean(scores_window),
                100 * np.mean(completion_window),
                # self._parameters['expl']['start'] to print epsilon,
                action_probs,
            ), end=end)

    def step(self, action_dict):
        # TODO: If interactive mode wait for input
        next_obs, all_rewards, done, info = self.env.step(action_dict)

        if self._rendering:
            self._renderer.render_env(show=True, show_observations=True, show_predictions=False)

        return next_obs, all_rewards, done, info

    def reset(self):
        obs, info = self.env.reset(True, True)
        # TODO: Oppure env.reset(regenerate_rail=True, regenerate_schedule=True)

        if self._rendering:
            self._renderer.reset()
        return obs, info

