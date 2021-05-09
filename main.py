from argparse import ArgumentParser, Namespace
import time
import json

from fltlnd.handler import ExcHandler


def main(episodes, parameters_filename, training, rendering, interactive, checkpoint):
    with open(parameters_filename) as json_file:
        parameters = json.load(json_file)

    ex = ExcHandler(parameters, training == 'T', rendering, interactive, checkpoint)
    ex.start(episodes)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-E", "--episodes", dest="episodes", help="Number of episodes to run", default=500, type=int)
    parser.add_argument("-P", "--parameters", dest="parameters", help="Parameter file to use",
                        default='parameters/example.json')
    parser.add_argument('-R', '--rendering', dest="rendering", help="Renders the environment", default=False,
                        action='store_true')
    parser.add_argument('-T', '--training', dest="training", help="Enables training", default="T", 
                        choices=['T', 'F'])
    parser.add_argument('-I', '--interactive', dest="interactive", help="Executes in interactive mode", default=False,
                        action='store_true')
    parser.add_argument('-C', '--checkpoint', dest="checkpoint", help="Checkpoint file to be loaded", default=None)
    args = parser.parse_args()

    start_time = time.time()

    main(args.episodes, args.parameters, args.training, args.rendering, args.interactive, args.checkpoint)

    print("--- %s seconds ---" % (time.time() - start_time))
