import random
import matplotlib.pyplot as plt
import numpy as np

from collections import deque
from pathlib import Path

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool

from agent.random_agent import RandomAgent
from observation.observation_handler import ObservationHandler

action_size = 5 # The action space of flatland is 5 discrete actions

def train_agent(n_episodes, parameters, render=False):
    # And some variables to keep track of the progress
    # scores_window = deque(maxlen=100)  # todo smooth when rendering instead
    # completion_window = deque(maxlen=100)
    # scores = []
    # completion = []
    # update_values = False

    env, obs_handler, policy = prepare_training(parameters)
    max_steps = int(4 * 2 * (env.height + env.width + (parameters['env']['n_agents'] / parameters['env']['n_cities']))) # Max number of steps per episode
    env_renderer = RenderTool(env)

    run_episodes(policy, obs_handler, env, max_steps, n_episodes, parameters, env_renderer, render)

    # plot_results(scores, completion)

def prepare_training(parameters):
    # Set the seeds
    random.seed(parameters['env']['seed'])
    np.random.seed(parameters['env']['seed'])

    # Observation builder
    obs_handler = ObservationHandler(parameters['obs'])

    # Setup the environment
    env = RailEnv(
        width=parameters['env']['x_dim'],
        height=parameters['env']['y_dim'],
        rail_generator=sparse_rail_generator(
            max_num_cities=parameters['env']['n_cities'],
            seed=parameters['env']['seed'],
            grid_mode=False,
            max_rails_between_cities=parameters['env']['max_rails_between_cities'],
            max_rails_in_city=parameters['env']['max_rails_in_city']
        ),
        schedule_generator=sparse_schedule_generator(),
        number_of_agents=parameters['env']['n_agents'],
        obs_builder_object=obs_handler.builder
    )

    env.reset(True, True)

    # Policy
    policy = RandomAgent(obs_handler.get_state_size(env), action_size)

    return env, obs_handler, policy

def run_episodes(policy, obs_handler, env, max_steps, n_episodes, parameters, env_renderer, render):
    for episode_idx in range(n_episodes):
        # observation.get_state_size(env)
        score = 0
        action_dict = dict()
        action_count = [0] * action_size
        agent_obs = [None] * env.get_num_agents()
        agent_prev_obs = [None] * env.get_num_agents()
        agent_prev_action = [2] * env.get_num_agents()

        # Reset environment
        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
        if render:
            env_renderer.reset()

        # Build agent specific observations
        for agent in env.get_agent_handles():
            if obs[agent]:
                agent_obs[agent] = obs_handler.normalize(obs[agent])
                agent_prev_obs[agent] = agent_obs[agent].copy()


        # Epsilon decay
        eps_start = max(parameters['expl']['end'], parameters['expl']['decay'] * parameters['expl']['start'])

        # Collection information about training TODO: ???
        # tasks_finished = np.sum([int(done[idx]) for idx in env.get_agent_handles()])
        # completion_window.append(tasks_finished / max(1, env.get_num_agents()))
        # scores_window.append(score / (max_steps * env.get_num_agents()))
        # completion.append((np.mean(completion_window)))
        # scores.append(np.mean(scores_window))
        # action_probs = action_count / np.sum(action_count)

        # Run episode
        for step in range(max_steps - 1):
            for agent in env.get_agent_handles():
                if info['action_required'][agent]:
                    # If an action is required, we want to store the obs at that step as well as the action
                    update_values = True
                    action = policy.act(agent_obs[agent])
                    action_count[action] += 1
                else:
                    update_values = False
                    action = 0
                action_dict.update({agent: action})

            # Environment step
            next_obs, all_rewards, done, info = env.step(action_dict)
            if render:
                env_renderer.render_env(show=True, show_observations=True, show_predictions=False) #TODO Rendering

            # Update replay buffer and train agent
            for agent in range(env.get_num_agents()):
                # Only update the values when we are done or when an action was taken and thus relevant information is present
                if update_values or done[agent]:
                    policy.step(agent,
                                agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent],
                                agent_obs[agent], done[agent])

                    agent_prev_obs[agent] = agent_obs[agent].copy()
                    agent_prev_action[agent] = action_dict[agent]

                if next_obs[agent]:
                    agent_obs[agent] = obs_handler.normalize(next_obs[agent])

                score += all_rewards[agent]

            if done['__all__']:
                break

            if episode_idx % 100 == 0:
                end = "\n"
                policy.save('./checkpoints/single-' + str(episode_idx) + '.pth')
                action_count = [1] * action_size
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
        
def plot_results(scores, completion):
    # Plot overall training progress at the end
    plt.plot(scores)
    plt.show()

    plt.plot(completion)
    plt.show()