from typing import Optional
from fltlnd.utils import TrainingMode
import json
import time
import random
from flatland.envs import malfunction_generators as mal_gen
import matplotlib.pyplot as plt
import numpy as np

from collections import deque

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool

from fltlnd.deadlocks import DeadlocksDetector
import fltlnd.agent as agent_classes
import fltlnd.obs as obs_classes
import fltlnd.logger as logger_classes
import fltlnd.replay_buffer as memory_classes

class ExcHandler:
    def __init__(self, params: dict, training_mode: TrainingMode, rendering: bool, checkpoint: Optional[str]):
        self._sys_params = params['sys'] # System
        self._obs_params = params['obs'] # Observation
        self._trn_params = params['trn'] # Training
        self._log_params = params['log'] # Policy

        self._rendering = rendering
        self._training = training_mode is not TrainingMode.EVAL
        self._tuning = training_mode is TrainingMode.TUNING
        self._train_best = training_mode in [TrainingMode.BEST, TrainingMode.EVAL]

        self._obs_class = getattr(obs_classes, self._sys_params['obs_class'])
        self._agent_class = getattr(agent_classes, self._sys_params['agent_class'])
        self._logger_class = getattr(logger_classes, self._sys_params['log_class'])
        self._memory_class = getattr(memory_classes, self._sys_params['memory_class'])

        self._obs_wrapper = self._obs_class(self._obs_params)
        self._env_handler = EnvHandler(self._sys_params['base_dir'] + "parameters/environments.json", self._obs_wrapper.builder, 
            self._rendering)

        # The action space of flatland is 5 discrete actions
        self._action_size = 5
        self._state_size = self._obs_wrapper.get_state_size()

        self._logger = self._logger_class(self._sys_params['base_dir'], self._log_params, self._tuning)

    def start(self, n_episodes):
        start_time = time.time()
        random.seed(self._sys_params['seed'])
        np.random.seed(self._sys_params['seed'])

        for run_id, params in enumerate(self._logger.get_run_params()):
            self._trn_params.update(params)
            self._policy = self._agent_class(self._state_size, self._action_size, self._trn_params,
            self._memory_class, self._training, self._train_best, self._sys_params['base_dir'])
            self._logger.run_start(self._trn_params, str(self._policy))
            self._env_handler.update(self._trn_params['env'], self._sys_params['seed'])

            # Max number of steps per episode
            self._max_steps = int(4 * 2 * (self._env_handler._params['x_dim'] + self._env_handler._params['y_dim'] + (
                    self._env_handler.get_num_agents() / self._env_handler._params['n_cities'])))

            eval_score = None
            for episode_idx in range(n_episodes):
                self._policy.episode_start()

                score = 0
                deadlocks = 0
                action_dict = dict()
                action_count = [0] * self._action_size
                agent_obs = [None] * self._env_handler.get_num_agents()
                agent_prev_obs = [None] * self._env_handler.get_num_agents()
                agent_prev_action = [2] * self._env_handler.get_num_agents()
                agent_prev_rewards = [0] * self._env_handler.get_num_agents()
                agent_prev_done = [0] * self._env_handler.get_num_agents()
                update_values = False

                # Reset environment
                obs, info = self._env_handler.reset()

                # Build agent specific observations
                for agent in self._env_handler.get_agents_handle():
                    if obs[agent]:
                        agent_obs[agent] = self._obs_wrapper.normalize(obs[agent])
                        agent_prev_obs[agent] = agent_obs[agent].copy()

                count_steps = 0
                # Run episode 
                for step in range(self._max_steps - 1):
                    count_steps += 1
                    for agent in self._env_handler.get_agents_handle():
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
                    for agent in self._env_handler.get_agents_handle():
                        # Only update the values when we are done or when an action was taken and thus relevant information is present
                        if self._training and (update_values or done[agent]):
                            self._policy.step(
                                agent_prev_obs[agent], agent_prev_action[agent], agent_prev_rewards[agent],
                                agent_obs[agent], agent_prev_done[agent]
                            )

                            agent_prev_obs[agent] = agent_obs[agent].copy()
                            agent_prev_action[agent] = action_dict[agent]
                            agent_prev_rewards[agent] = all_rewards[agent]
                            agent_prev_done[agent] = done[agent]

                        if next_obs[agent]:
                            agent_obs[agent] = self._obs_wrapper.normalize(next_obs[agent])

                        score += all_rewards[agent]
                    
                    deadlocks += sum(info['deadlocks'].values())

                    if done['__all__']:
                        break

                # Collection information about training
                tasks_finished = np.sum([int(done[idx]) for idx in self._env_handler.get_agents_handle()])
                action_probs = action_count / np.sum(action_count)

                self._logger.log_episode({**{
                    "completions": tasks_finished / max(1, self._env_handler.env.get_num_agents()),
                    "scores": score / (self._max_steps * self._env_handler.env.get_num_agents()),
                    "steps": count_steps / self._max_steps,
                    "loss": self._policy.stats['loss'],
                    "deadlocks": sum(info['deadlocks'].values()) / self._env_handler.env.get_num_agents(), #TODO Check deadlock count
                    "exploration_prob": self._policy.stats['eps_val'],
                    "exploration_count": self._policy.stats['eps_counter'] / np.sum(action_count)
                    # "min_steps": min_steps / ?
                }, **dict(zip(["act_" + str(i) for i in range(self._action_size)], action_probs))}, episode_idx)

                eval_score = score
                self._policy.episode_end()

                if episode_idx % self._trn_params['checkpoint_freq'] == 0:
                    end = "\n"
                    action_count = [1] * self._action_size

                    if self._training:
                        self._policy.save('./tmp/checkpoints/' + str(self._policy) + '-' + str(episode_idx) + '.pth/')
                        self._policy.save_best()

                else:
                    end = " "

                self._env_handler.print_results(episode_idx, self._logger.get_window('scores'), 
                    self._logger.get_window('completions'), action_probs, end)

                #TODO Evaluation once every tot

            self._logger.run_end(params, eval_score / (self._max_steps * self._env_handler.env.get_num_agents()), run_id)

        return time.time() - start_time

class EnvHandler:
    def __init__(self, env_filename, obs_builder, rendering=False):
        with open(env_filename) as json_file:
            self._full_env_params = json.load(json_file)

        self._obs_builder = obs_builder
        self._rendering = rendering
        self.deadlocks_detector = DeadlocksDetector()

    def update(self, env="r1.s", seed=None):
        self._params = self._full_env_params[env]

        self.x_dim = self._params['x_dim']
        self.y_dim = self._params['y_dim']
        self.n_cities = self._params['n_cities']
        self.grid_mode = self._params['grid_mode']
        self.max_rails_between_cities = self._params['max_rails_between_cities']
        self.max_rails_in_city = self._params['max_rails_in_city']
        self.n_agents = self._params['n_agents']
        min_mal, max_mal = self._params['malfunction_duration']
        self.mal_params = mal_gen.MalfunctionParameters(1 / self._params['min_malfunction_interval'], min_mal, max_mal)

        # Check for ParamMalfunctionGen existance for retrocompatibility purposes
        try:
            self.env = RailEnv(
                width=self.x_dim,
                height=self.y_dim,
                rail_generator=sparse_rail_generator(
                    max_num_cities=self.n_cities,
                    seed=seed,
                    grid_mode=self.grid_mode,
                    max_rails_between_cities=self.max_rails_between_cities,
                    max_rails_in_city=self.max_rails_in_city
                ),
                schedule_generator=sparse_schedule_generator(),
                number_of_agents=self.n_agents,
                obs_builder_object=self._obs_builder,
                malfunction_generator=mal_gen.ParamMalfunctionGen(self.mal_params),
            )
        except AttributeError:
            self.env = RailEnv(
                width=self.x_dim,
                height=self.y_dim,
                rail_generator=sparse_rail_generator(
                    max_num_cities=self.n_cities,
                    seed=seed,
                    grid_mode=self.grid_mode,
                    max_rails_between_cities=self.max_rails_between_cities,
                    max_rails_in_city=self.max_rails_in_city
                ),
                schedule_generator=sparse_schedule_generator(),
                number_of_agents=self.n_agents,
                obs_builder_object=self._obs_builder,
                malfunction_generator_and_process_data=mal_gen.malfunction_from_params(self.mal_params)
            )

        if self._rendering:
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
        next_obs, all_rewards, done, info = self.env.step(action_dict)

        # Compute deadlocks
        deadlocks = self.deadlocks_detector.step(self.env)
        info["deadlocks"] = {}
        for agent in self.get_agents_handle():
            info["deadlocks"][agent] = deadlocks[agent]

        if self._rendering:
            self._renderer.render_env(show=True, show_observations=True, show_predictions=False)

        return next_obs, all_rewards, done, info

    def get_num_agents(self):
        return self.n_agents

    def get_agents_handle(self):
        return self.env.get_agent_handles()

    def reset(self):
        obs, info = self.env.reset(regenerate_rail=True, regenerate_schedule=True)

        self.deadlocks_detector.reset(self.env.get_num_agents())
        info["deadlocks"] = {}

        for agent in self.get_agents_handle():
            info["deadlocks"][agent] = self.deadlocks_detector.deadlocks[agent]

        if self._rendering:
            self._renderer.reset()

        return obs, info
