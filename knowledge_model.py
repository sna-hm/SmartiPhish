import pickle
import logging
import numpy as np
from numpy import load
from sklearn import metrics
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

class KnowledgeModel:
    def __init__(self):
        # Load pre-trained models
        self.model = load_model('moraphishdet.h5', custom_objects={'GraphConvSkip': GraphConvSkip, 'MinCutPool': MinCutPool,
                                                        'GlobalAvgPool': GlobalAvgPool})
        self.model.trainable = False

    async def get_km_prediction(self, A_list, X_list, X_A_list):
        batches = batch_iterator([A_list, X_list, X_A_list], batch_size = 1)
        pred = []
        for b in batches:
            X, A, I = Batch(b[0], b[1]).get('XAI')
            A = sp_matrix_to_sp_tensor(A)
            X_A = b[2]
            pred.append(tf.get_static_value(self.model([X_A, X, A, I], training=False)[0]))
        return pred
