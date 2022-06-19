import pickle
import random
import logging
import numpy as np
from numpy import load
from sklearn import metrics
from numpy.random import randn
from connection import DBConnection
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *

from spektral.layers import GraphConvSkip, GlobalAvgPool
from spektral.layers.ops import sp_matrix_to_sp_tensor
from spektral.layers.pooling import MinCutPool
from spektral.utils.convolution import normalized_adjacency
from spektral.utils import batch_iterator
from spektral.utils.data import Batch

class AdversarialAttack:
    def __init__(self):
        # Load pre-trained models
        base_model = load_model('moraphishdet.h5', custom_objects={'GraphConvSkip': GraphConvSkip, 'MinCutPool': MinCutPool,
                                                        'GlobalAvgPool': GlobalAvgPool})
        base_model.trainable = False

        inputs = Input(shape=(64))
        x = inputs
        x = base_model.get_layer('output') (x)
        self.model = Model(inputs, x)

        self.latent_dim = 128
        self.generator = load_model('model/generator-gan.h5')
        self.generator.trainable = False

    def get_prediction(self, data):
        return self.model.predict(data)

    def generate_data(self, n):
        # generate points in latent space
        x_input = self.generate_latent_points(n)
        # predict outputs
        x_fake = self.generator.predict(x_input)
        return x_fake

    # generate points in latent space as input for the generator
    def generate_latent_points(self, n):
    	# generate points in the latent space
    	x_input = randn(self.latent_dim * n)
    	# reshape into a batch of inputs for the network
    	x_input = x_input.reshape(n, self.latent_dim)
    	return x_input


attack = AdversarialAttack()
data = attack.generate_data(5)
predictions = attack.get_prediction(data)

connection = DBConnection("smartiphish").get_connection()
cursor = connection.cursor(buffered=True)

#save states to adv_attack table
sql = "TRUNCATE adv_attack"
cursor.execute(sql)
connection.commit()

for pred in predictions:
    rank = random.uniform(0, 0.0005)
    rank = rank if (rank < 0.00024) else 0
    state = str(pred[1]) + ',' + '0' + ',' + str(rank)
    
    #save states to adv_attack table
    sql = "INSERT INTO adv_attack(state) VALUES(%s)"
    val = (str(state), )
    cursor.execute(sql, val)
    connection.commit()

cursor.close()
connection.close()
