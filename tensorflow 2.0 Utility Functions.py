# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 21:59:22 2019

@author: Rashid Haffadi
"""

import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
def set_cmap(cmap:str):
    plt.rcParams['image.cmap'] = cmap
    
def begin():
    print("tensorflow: {}, keras: {}, gpu:  {}".format(tf.__version__, keras.__version__, tf.test.is_gpu_available()))
    
config = {
        {
                'name': 'Input',
                'type': keras.layers.Conv2D,
                'input_shape': (32, 32, 1),
                'activation': 'relu',
                'filters': 6,
                'kernel_size': (3, 3),
                
        },
        {
                'name': 'Pool1',
                'pool': (2, 2)
             
        }
}

def get_layers(config=None):
    layers = []
    layers.append(keras.layers.Conv2d(6, kernel_size=(3, 3)))
    
    return layers

def build_model(layers):
    layers = get_layers()
    model = keras.Sequential(*layers)
    
def compile_model():
    pass
    
    
def train_model():
    pass

def plot_curve(history:tf.python.keras.callbacks.History, metrics=['accuracy', 'loss'], valid:bool=False):
    history = history.history
    for metric in metrics:
        plot_metric(history, metric, valid)
            
def plot_metric(history, metric, valid, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
        ax.plot(history[metric])
        if valid:
            ax.plot(history['val_' + metric])
        ax.set_title('Model ' + metric.title() + ':')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.title())
        ax.legend(['Train', 'Valid'], loc='upper left')

def most_confused(n, model, valid_data, loss):
    x_true, y_true = valid_data
    y_pred = model.predict_classes(x_true)
    losses = loss(y_true, y_pred)
    sort = np.argsort(losses)
    sort = sort[-n:]
    return x_true[sort], y_true[sort], y_pred[sort], losses[sort]
    
#lenet5 = 