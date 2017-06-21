import time
from keras.models import load_model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras import backend as K
from keras import __version__ as keras_version
import h5py

class Model:
    def __init__(self, model_path, width=320, height=160):
        self.startTime = time.time()
        self.model_path = model_path
        self.width = width
        self.height = height
        self.imshape = (height, width, 3)

        # parameters
        self.tf_session = K.get_session()
        self.tf_graph = self.tf_session.graph
        self.lr = 0.00001

    def create(self):
        with self.tf_session.as_default():
            with self.tf_graph.as_default():
                self.kmodel = Sequential()
                self.kmodel.add(Lambda(lambda x: ((x/255.0)-0.5), input_shape=self.imshape, name='lambda'))
                self.kmodel.add(Conv2D(nb_filter=24, nb_row=5, nb_col=5, subsample=(2,2), border_mode='valid', activation='elu', name='conv1'))
                self.kmodel.add(Conv2D(nb_filter=36, nb_row=5, nb_col=5, subsample=(2,2), border_mode='valid', activation='elu', name='conv2'))
                self.kmodel.add(Conv2D(nb_filter=48, nb_row=5, nb_col=5, subsample=(2,2), border_mode='valid', activation='elu', name='conv3'))
                self.kmodel.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1,1), border_mode='valid', activation='elu', name='conv4'))
                self.kmodel.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1,1), border_mode='valid', activation='elu', name='conv5'))
                self.kmodel.add(Flatten(name='flat'))
                self.kmodel.add(Dense(100, activation='elu', name='dense1'))
                self.kmodel.add(Dense(50, activation='elu', name='dense2'))
                self.kmodel.add(Dense(1, name='output'))
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
        modelFile = 'model-session{}.h5'.format(session)
        print("Saving model to disk: ",modelFile)
        self.kmodel.save(modelFile)

    def preprocess(self, image):
        # no preprocessing in this model...
        return image

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

