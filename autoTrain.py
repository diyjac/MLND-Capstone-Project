#!/usr/bin/python
"""
autoTrainer.py: version 0.1.0

History:
2017/06/19: Initial version.
"""

# import some useful modules
import argparse
import time

# We have modularize our interface to the simulator
from lib.simenv import Drive

# We have modularize our agent and trainer embedded pipeline
from lib.agent import Agent
from lib.trainer import Trainer


# Our main CLI code, use --help to get full option list
if __name__ == "__main__":

    # initialize argparse to parse the CLI
    usage = 'python %(prog)s [options] gen_n_model gen_n_p_1_model'
    desc = 'DIYJAC\'s Udacity MLND Capstone Project: ' + \
           'Automated Behavioral Cloning'
    defaultInput = 'gen0/model.h5'
    inputHelp = 'gen0 model or another pre-trained model'
    defaultOutput = 'gen1'
    outputHelp = 'gen1 model or another target model'
    defaultMaxRecall = 500
    defaultSpeed = 30
    maxrecallHelp = 'maximum samples to collect for training session, ' + \
                    'defaults to 500'
    speedHelp = 'cruz control speed, defaults to 30'

    # set defaults
    parser = argparse.ArgumentParser(prog='autoTrain.py',
                                     usage=usage, description=desc)
    parser.add_argument('--maxrecall', type=int, default=defaultMaxRecall,
                        help=maxrecallHelp)
    parser.add_argument('--speed', type=int, default=defaultSpeed,
                        help=speedHelp)
    parser.add_argument('inmodel', type=str, default=defaultInput,
                        help=inputHelp)
    parser.add_argument('outmodel', type=str, default=defaultOutput,
                        help=outputHelp)
    args = parser.parse_args()

    # create a trainer
    trainer = Trainer(args.inmodel)

    # sleep - so keras and tensorflow will allow multiple models...
    time.sleep(1)

    # create an agent to train
    agent = Agent(args.outmodel+'/model-session{}.h5',
                  maxrecall=args.maxrecall)

    # sleep - so keras and tensorflow will allow multiple models...
    time.sleep(1)

    # Define environment/simulation
    sim = Drive(agent.width, agent.height, args.speed)

    # train in the environment
    sim.start_training(trainer, agent)
