import time
import tensorflow as tf
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras import backend as K
from keras import __version__ as keras_version
import h5py
import cv2
import csv
import os

class Model:
    def __init__(self, model_path, width=64, height=64, log_metrics=False):
        self.startTime = time.time()
        self.model_path = model_path
        self.width = width
        self.height = height
        self.imshape = (height, width, 3)

        # parameters
        self.tf_session = K.get_session()
        self.tf_graph = self.tf_session.graph
        self.lr = 0.000001
        self.epsilon = 1e-08

        # logger
        self.log_metrics = log_metrics
        if self.log_metrics:
            (path, modelinstance) = os.path.split(model_path)
            self.log_filename = os.path.join(path, "training-progress.csv")
            self.log_fields = ['session', 'testing', 'lap_timestep', 'training_sample', 'acte', 'mse', 'success', 'x', 'y']
            self.log_file = open(self.log_filename, 'w')
            self.log_writer = csv.DictWriter(self.log_file, fieldnames=self.log_fields)
            self.log_writer.writeheader()

    def create(self):
        with self.tf_session.as_default():
            with self.tf_graph.as_default():
                self.kmodel = Sequential()
                self.kmodel.add(Lambda(lambda x: ((x/255.0)-0.5), input_shape=self.imshape, name='lambda'))
                self.kmodel.add(Conv2D(filters=24, kernel_size=(5, 5), strides=[2,2], padding='valid', activation='elu', name='conv1'))
                self.kmodel.add(Conv2D(filters=36, kernel_size=(5, 5), strides=[2,2], padding='valid', activation='elu', name='conv2'))
                self.kmodel.add(Conv2D(filters=48, kernel_size=(5, 5), strides=[2,2], padding='valid', activation='elu', name='conv3'))
                self.kmodel.add(Conv2D(filters=64, kernel_size=(3, 3), strides=[1,1], padding='valid', activation='elu', name='conv4'))
                self.kmodel.add(Conv2D(filters=64, kernel_size=(3, 3), strides=[1,1], padding='valid', activation='elu', name='conv5'))
                self.kmodel.add(Flatten(name='flat'))
                self.kmodel.add(Dense(units=100, activation='elu', name='dense1'))
                self.kmodel.add(Dense(units=50, activation='elu', name='dense2'))
                self.kmodel.add(Dense(units=1, name='output'))
                adam = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=self.epsilon, decay=0.0)
                self.kmodel.compile(optimizer=adam, loss="mse")
                self.kmodel.summary()

    def load(self):
        global keras_version
        # check that model Keras version is same as local Keras version
        f = h5py.File(self.model_path, mode='r')
        model_version = f.attrs.get('keras_version')
        keras_version = str(keras_version).encode('utf8')

        if model_version != keras_version:
            print('You are using Keras version ', keras_version,
                  ', but the model was built using ', model_version)
        self.tf_session = K.get_session()
        self.tf_graph = self.tf_session.graph
        with self.tf_session.as_default():
            with self.tf_graph.as_default():
                self.kmodel = load_model(self.model_path)

    def save(self, session):
        modelFile = self.model_path.format(session)
        print("Saving model to disk: ",modelFile)
        self.kmodel.save(modelFile)

    def preprocess(self, image):
        # get shape and chop off 1/3 from the top and 1/5 from the bottom
        shape = image.shape
        # note: numpy arrays are (row, col)!
        image = image[shape[0]//3:shape[0]-shape[0]//5,:,:]
        return cv2.resize(image, (self.height, self.width), interpolation=cv2.INTER_AREA)

    # Define a function that trains an agent in an event loop
    def predict(self, image):
        # store experience from last action
        value = 0.
        try:
            with self.tf_session.as_default():
                with self.tf_graph.as_default():
                    value = self.kmodel.predict(image, batch_size=1)
        except:
            pass
        return float(value)

    # Define a function that collect metrics from training sessions
    def log(self, session, testing, lap_timestep, training_sample, acte, mse, success, x, y):
        if self.log_metrics:
            self.log_writer.writerow({
                'session': session,
                'testing': testing,
                'lap_timestep': lap_timestep,
                'training_sample': training_sample,
                'acte': acte,
                'mse': mse,
                'success': success,
                'x': x,
                'y': y
            })

    # Define a function that close the training sessions
    def close_log(self):
        if self.log_metrics:
            self.log_file.close()
