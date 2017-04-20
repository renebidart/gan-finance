import keras
from keras.models import Sequential
from keras.layers import Dropout, Flatten
from keras.layers.core import Activation, Dense, Reshape
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np 
import random


## Stock model based on: https://github.com/jacobgil/keras-dcgan/blob/master/dcgan.py
def gen_model_stock(lr=.01):
    model = Sequential()
    model.add(Dense(units=64, input_dim=32))
    model.add(Activation('tanh'))
    model.add(Dense(248))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((62, 4), input_shape=(248,)))
    model.add(UpSampling1D(size=2))
    model.add(Conv1D(16, 3, padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling1D(size=2))
    model.add(Conv1D(1, 3, padding='same'))
    model.add(Activation('tanh'))
    gen_optim = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=gen_optim) ### lr=0.01
    return model

def desc_model_stock(lr=0.0005):
    model = Sequential()
    model.add(Conv1D(8, 3, padding='same', input_shape=(248, 1)))
    model.add(Activation('tanh'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(16, 3))
    model.add(Activation('tanh'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(248))
    model.add(Activation('tanh'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    d_optim = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=d_optim)
    return model

def build_gen_desc_stock(gen_model_stock, desc_model_stock, lr=0.0005):
    model = Sequential()
    model.add(gen_model_stock)
    gen_model_stock.trainable = False
    model.add(desc_model_stock)
    g_optim = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=g_optim)
    return model


###### Make a few changes that should help the training
def gen_model_standard(lr=.01):
    model = Sequential()
    model.add(Dense(units=64, input_dim=32))
    model.add(Activation('relu'))
    model.add(Dense(248))
    model.add(BatchNormalization())

    model.add(Activation('relu'))
    model.add(Reshape((62, 4), input_shape=(248,)))
    model.add(UpSampling1D(size=2))
    model.add(Conv1D(16, 3, padding='same'))
    #model.add(BatchNormalization())

    model.add(Activation('relu'))
    model.add(UpSampling1D(size=2))
    model.add(Conv1D(1, 3, padding='same'))
    model.add(Activation('tanh'))
    Adam = keras.optimizers.Adam(lr=lr)    
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model

def desc_model_standard(lr=0.0005):
    model = Sequential()
    model.add(Conv1D(8, 3, padding='same', input_shape=(248, 1)))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(16, 3))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(248))
    #model.add(BatchNormalization())

    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    Adam = keras.optimizers.Adam(lr=lr)    
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model

def build_gen_desc_standard(gen_model_stock, desc_model_stock, lr=0.0005):
    model = Sequential()
    model.add(gen_model_stock)
    gen_model_stock.trainable = False
    model.add(desc_model_stock)
    Adam = keras.optimizers.Adam(lr=lr)    
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model

#########  Build a small model that seems resonbable
def gen_model_v1(lr_gen=0.001):  # For the model they used lr 20*desc=10*both=gen=default
    model = Sequential()
    model.add(Dense(units=64, input_dim=32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(248))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Reshape((62, 4), input_shape=(248,)))
    model.add(UpSampling1D(size=2))

    model.add(Conv1D(16, 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(UpSampling1D(size=2))
    model.add(Conv1D(1, 3, padding='same'))
    model.add(Activation('softmax'))

    Adam = keras.optimizers.Adam(lr=lr_gen)    
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model

def desc_model_v1(lr_desc=0.0001):
    model = Sequential()
    model.add(Conv1D(8, 3, padding='same', input_shape=(248, 1)))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=(2))) 

    model.add(Conv1D(16, 3))
    model.add(keras.layers.normalization.BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=(2))) 

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    Adam = keras.optimizers.Adam(lr=lr_desc)    
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model

def build_gen_desc_v1(gen_model, desc_model):
    lr_both=0.0001
    model = Sequential()
    model.add(gen_model)
    gen_model.trainable = False
    model.add(desc_model)
    Adam = keras.optimizers.Adam(lr=lr_both)    
    model.compile(loss="categorical_crossentropy", optimizer=Adam, metrics=['accuracy'])
    return model
