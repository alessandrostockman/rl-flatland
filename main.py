from argparse import ArgumentParser, Namespace
import time
import json

from train import train_agent

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-E", "--episodes", dest="episodes", help="Number of episodes to run", default=500, type=int)
    parser.add_argument("-P", "--parameters", dest="parameters", help="Parameter file to use", default='parameters/example.json')
    parser.add_argument('-R', '--rendering', dest="rendering", help="Renders the environment", default=False, action='store_true')
    args = parser.parse_args()

    with open(args.parameters) as json_file:
        parameters = json.load(json_file)
        # TODO Add to parameters:
        #
        #    "train": {
        #        'buffer_size': int(1e5),
        #        'batch_size': 32,
        #        'update_every': 8,
        #        'learning_rate': 0.5e-4,
        #        'tau': 1e-3,
        #        'gamma': 0.99,
        #        'buffer_min_size': 0,
        #        'hidden_size': 256,
        #        'use_gpu': False
        #    }

    start_time = time.time()
    train_agent(args.episodes, parameters, args.rendering)
    print("--- %s seconds ---" % (time.time() - start_time))