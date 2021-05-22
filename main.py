from argparse import ArgumentParser
from fltlnd.utils import TrainingMode
import json
from typing import Optional

from fltlnd.handler import ExcHandler

def main(episodes: int, training: str, rendering: bool, checkpoint: Optional[str]):
    with open("parameters/setup.json") as json_file:
        parameters = json.load(json_file)

    ex = ExcHandler(parameters, {
        'off': TrainingMode.OFF,
        'tuning': TrainingMode.TUNING,
        'best': TrainingMode.BEST,
    }[training], rendering, checkpoint)
    ex.start(episodes)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-E", "--episodes", dest="episodes", help="Number of episodes to run", default=500, type=int)
    parser.add_argument('-R', '--rendering', dest="rendering", help="Renders the environment", default=False,
                        action='store_true')
    parser.add_argument('-T', '--training', dest="training", 
    help='''Training modes:
        off - Executes the environment without training the model |
        tuning - Enables tuning of hyperparameters |
        best - Loads the best checkpoint for the chosen model (checkpoints/{model}) and trains it
    ''', default="best", choices=['off', 'tuning', 'best'])
    parser.add_argument('-C', '--checkpoint', dest="checkpoint", help="Cusotm checkpoint path", default=None)
    args = parser.parse_args()

    main(args.episodes, args.training, args.rendering, args.checkpoint)
