#!/usr/bin/python
"""
agent.py: version 0.1.0

History:
2017/06/19: Initial version converted to a class
"""

# import some useful functions
import json
import numpy as np
import random
import importlib
import time
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras import backend as K
from lib.recall import Recall
import matplotlib.pyplot as plt


# Define the agent class
class Agent:

    # initialize the agent
    def __init__(self, model_path, maxrecall=500, log_metrics=True):
        # get model info
        (path, modelinstance) = os.path.split(model_path)
        model = importlib.import_module("{}.model".format(path))
        self.model = model.Model(model_path, log_metrics=log_metrics)
        self.model.create()
        self.log_metrics = log_metrics

        # environment and recall
        self.start_time = time.time()

        # Initialize experience replay object
        self.recall = Recall(maxmem=maxrecall, width=self.model.width,
                             height=self.model.height)

        # training parameters
        self.path = path
        self.tf_session = self.model.tf_session
        self.tf_graph = self.model.tf_graph
        self.batch_size = 50
        self.width = self.model.width
        self.height = self.model.height
        self.count = 0
        self.train_count = 0
        self.acte = 0.0
        self.tse = 0.0

    # Define a function that preprocess a image for the trainer and recall
    def preprocess(self, image_array):
        return self.model.preprocess(image_array)

    # Define a function that lets an agent predict
    def prediction(self, image):
        return self.model.predict(
            image.reshape(-1, self.height, self.width, 3))

    # Define a function that either stores or trains a model in an event loop
    def train(self, image, predicted_steer, cte, correct_steer, sim_over,
              training, testing, x, y):
        # store experience from last correction or training
        self.count += 1
        self.acte += abs(cte)
        self.tse += (predicted_steer-correct_steer)**2
        self.mse = self.tse/self.count
        self.success = (abs(cte) < 2.)
        if sim_over and not testing:
            self.model.log(
                self.train_count+1, testing, self.count,
                len(self.recall.X), self.acte, self.mse, self.success, x, y)
            self.count = 0
            self.tse = 0.0
            self.acte = 0.0
            return self.model_trainer()
        else:
            if training:
                self.model.log(
                    self.train_count+1, testing, self.count,
                    len(self.recall.X), self.acte, self.mse,
                    self.success, x, y)
                self.recall.remember(image, correct_steer)
            elif testing:
                self.model.log(
                    self.train_count, testing, self.count,
                    len(self.recall.X), self.acte, self.mse,
                    self.success, x, y)
            elif abs(predicted_steer-correct_steer) > 0.05:
                self.model.log(
                    self.train_count+1, testing, self.count,
                    len(self.recall.X), self.acte, self.mse,
                    self.success, x, y)
                self.recall.remember(image, correct_steer)
        if sim_over:
            self.count = 0
            self.tse = 0.0
            self.acte = 0.0
        return False

    # Define a function that trains a model in an event loop
    def model_trainer(self):
        with self.tf_session.as_default():
            with self.tf_graph.as_default():
                # start training
                if len(self.recall.X) > 100:
                    print("Model Trainer Starting...")
                    batch_size = 20
                    samples_per_epoch = int(len(self.recall.X)/batch_size)
                    val_size = int(samples_per_epoch/10)
                    if val_size < 10:
                        val_size = 10
                    nepoch = 100

                    # train on a fit generator
                    history = self.model.kmodel.fit_generator(
                                self.recall.batchgen(),
                                steps_per_epoch=samples_per_epoch,
                                epochs=nepoch,
                                validation_data=self.recall.batchgen(),
                                validation_steps=val_size,
                                verbose=1)

                    # save our new model instance
                    self.model.save(self.train_count+1)

                    # forget half of what we stored before.
                    self.recall.forget()
                    self.training_time = time.time() - self.start_time

                    # disable plotting of loss function - takes up too much space..
                    # save the training and validation loss for each epoch
                    # plotsave = '{}/sessionloss{}.png'.format(self.path, self.train_count+1)
                    # print("saving loss plot to:", plotsave)
                    # fig, ax1 = plt.subplots(1, 1, figsize=(10, 4))
                    # ax1.plot(history.history['loss'])
                    # ax1.plot(history.history['val_loss'])
                    # ax1.set_xlabel('epoch')
                    # ax1.set_ylabel('mean squared error loss')
                    # plt.title('model mean squared error loss session={}'.format(self.train_count+1))
                    # plt.legend(['training set', 'validation set'], loc='upper right')
                    # fig.savefig(plotsave)
                    # plt.close(fig)
                    
                else:
                    if len(self.recall.X) < 75:
                        self.training_time = time.time() - self.start_time
                        print("Training Complete!!!!")
                        print("Train count:", self.train_count,
                              "final len(X):", len(self.recall.X))
                        print("Total Training time (sec): ",
                              self.training_time,
                              "Average time per lap (sec): ",
                              self.training_time/self.train_count)
                        return True
        self.train_count += 1
        self.tse = 0.0
        self.acte = 0.0
        return False

    # close logging
    def close_log(self):
        self.model.close_log()
