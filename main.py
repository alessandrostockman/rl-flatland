from argparse import ArgumentParser
from fltlnd.utils import TrainingMode
import json
from typing import Optional

from fltlnd.handler import ExcHandler

def main(episodes: int, training: str, rendering: bool, checkpoint: Optional[str], synclog: bool, verbose: bool, seed, world):
    with open("parameters/setup.json") as json_file:
        parameters = json.load(json_file)
        parameters['sys']['seed'] = int(seed)
        parameters['trn']['env'] = world

    ex = ExcHandler(parameters, {
        'eval': TrainingMode.EVAL,
        'tuning': TrainingMode.TUNING,
        'best': TrainingMode.BEST,
        'fresh': TrainingMode.FRESH,
        'debug': TrainingMode.DEBUG,
    }[training], rendering, checkpoint, synclog, verbose)
    ex.start(episodes)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-E", "--episodes", dest="episodes", help="Number of episodes to run", default=500, type=int)
    parser.add_argument('-R', '--rendering', dest="rendering", help="Renders the environment", default=False,
                        action='store_true')
    parser.add_argument('-S', '--synclog', dest="synclog", help="Syncs logs on the cloud", default=False,
                        action='store_true')
    parser.add_argument('-V', '--verbose', dest="verbose", help="Prints results on the console", default=False,
                        action='store_true')
    parser.add_argument('-T', '--training', dest="training", 
    help='''Training modes:
        debug - Doesn't save checkpoints |
        eval - Executes the environment without training the model |
        tuning - Enables tuning of hyperparameters |
        best - Loads the best checkpoint for the chosen model (checkpoints/{model}) and trains it |
        fresh - Starts the training without loading a checkpoint
    ''', default="debug", choices=['debug', 'eval', 'tuning', 'best', 'fresh'])
    parser.add_argument('-C', '--checkpoint', dest="checkpoint", help="Cusotm checkpoint path", default=None)
    parser.add_argument('-X', '--seed', dest="seed", help="", default=42)
    parser.add_argument('-W', '--world', dest="world", help="", default="t1.lx")
    args = parser.parse_args()

    main(args.episodes, args.training, args.rendering, args.checkpoint, args.synclog, args.verbose, args.seed, args.world)
