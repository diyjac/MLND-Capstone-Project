#!/usr/bin/python
"""
recall.py: version 0.1.0

History:
2017/06/19: Initial version converted to a class
"""

# import some useful function
import numpy as np
import random


# Define a class that will handle remembering features and
# steering angles to be learn by the model.
class Recall:
    def __init__(self, maxmem=1000, width=320, height=160):

        # initialize the recall class with empty storage
        self.maxmem = maxmem
        self.X = []
        self.y = []
        self.width = width
        self.height = height
        self.input_size = width*height*3

    # store additional information for later retrieval
    def remember(self, X, y):
        self.X.append(X)
        self.y.append(y)
        if len(self.X) > self.maxmem:
            self.X = self.X[1:]
            self.y = self.y[1:]

    # forget half (first half - FIFO) of what we collected
    def forget(self):
        self.X = self.X[len(self.X)//2:]
        self.y = self.y[len(self.y)//2:]

    # the batch generator used by the fit generator
    def batchgen(self, batch_size=1):
        while 1:
            i = int(random.random()*len(self.X))
            image = self.X[i][None, :, :, :]
            y = np.array([self.y[i]])
            yield image, y
