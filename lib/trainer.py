#!/usr/bin/python
"""
trainer.py: version 0.1.0

History:
2017/06/19: Initial version converted to a class
"""

# import some useful modules
import os
import importlib


# Define a class that will handle loading the trainer model
# and get the steering angle predictions once requested
class Trainer:
    def __init__(self, model_path):
        (path, modelinstance) = os.path.split(model_path)
        model = importlib.import_module("{}.model".format(path))
        self.model = model.Model(model_path)
        self.model.load()
        print("starting trainer...")

    # Define a function that trains an agent in an event loop
    def get_steering(self, image_array):
        # get the preprocessed image
        image = self.model.preprocess(image_array)
        # store experience from last action
        return self.model.predict(image[None, :, :, :])
