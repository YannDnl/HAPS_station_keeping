import os
from keras.layers import Flatten, Dense, InputLayer
from keras.models import Sequential
from keras.optimizers.legacy import Adam
from keras.losses import MeanSquaredError
import random as rd

h,w = 10, 50

def create_model(input_shape=2+2+h*w, num_actions=3, summary=False):
    '''
    Crée un NN à utiliser pour approcher Q(s,a)
    '''
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(num_actions, activation='relu'))
    model.build(input_shape)
    model.compile(optimizer=Adam(), loss=MeanSquaredError())
    if summary :
        model.summary()
    return model

def create_model_sigm(input_shape=2+2+h*w, num_actions=3, summary=False):
    '''
    Crée un NN à utiliser pour approcher Q(s,a)
    '''
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(24, activation='tanh'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='sigmoid'))
    model.add(Dense(num_actions, activation='relu'))
    model.build(input_shape)
    model.compile(optimizer=Adam(), loss=MeanSquaredError())
    if summary :
        model.summary()
    return model

def load_best_weights(model, dir="chckpt"):
    chkps_dir = os.path.join(os.curdir, dir)
    weights = os.listdir(chkps_dir)
    if len(weights) > 0:
        losses = [float(wght.split("-")[-1].split(".")[0] + "." + wght.split("-")[-1].split(".")[1]) for wght in weights]
        argmax = losses.index(max(losses))
        model.load_weights(os.path.join(chkps_dir, weights[argmax]))

def load_random_weights(model, dir="chckpt"):
    chkps_dir = os.path.join(os.curdir, dir)
    weights = os.listdir(chkps_dir)
    if len(weights) > 0:
        model.load_weights(os.path.join(chkps_dir, rd.choice(weights)))