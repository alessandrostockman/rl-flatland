# Flatland Problem

Implementation of a Multi-Agent Reinforcement Learning solution for a vehicle scheduling problem consisting in multiple trains needing to reach a predetermined target station inside an arbitrarily large rail environment. 

Core aspects of the project:
- DDDQN with Noisy Networks and Prioritized Experience Replay
- PPO

![Execution Demo](https://github.com/alessandrostockman/rl-flatland/blob/master/res/demo.gif)

## Usage

The modules inside `fltlnd/` can be either executed by running `main.py` or included inside a notebook by instanciating and starting `fltlnd.handler.ExcHandler`.

The execution can be altered by changing some configuration files:
- `parameters/environments.json` - Predefined environments on top of which the system is executed
- `parameters/hp.json` - Definition of Tensorboard hyperparameter tuning values
- `parameters/setup.json` - Project and execution settings

### How to run

`python main.py [-E / --episodes EPISODES] [-R / --rendering]  [-S / --synclog] [-V / --verbose] [-T / --training TRAINING_MODE] [-C / --checkpoint CHECKPOINT] [-X / --seed] [-W / --world WORLD]`

- `--episodes`: Number of episodes to run
- `--rendering`: Renders the environment
- `--synclog`: Syncs logs on the cloud
- `--verbose`: Prints results on the console
- `--training`: Training modes: 
    - debug - Doesn't save checkpoints
    - eval - Executes the environment without training the model
    - tuning - Enables tuning of hyperparameters
    - best - Loads the best checkpoint for the chosen model (checkpoints/{model}) and trains it
    - fresh - Starts the training without loading a checkpoint
- `--checkpoint`: Cusotm checkpoint path
- `--seed`: Custom seed for current execution, overrides the defined setup
- `--world`: Custom environment for execution, overrides the defined setup
 