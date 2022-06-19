import os
import sys
import codecs
import pickle
import gensim
import urllib
import logging

import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from connection import DBConnection

import numpy as np
from numpy import load
from numpy import save
from numpy import asarray
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.text import Tokenizer

from spektral.layers import GraphConvSkip, GlobalAvgPool
from spektral.layers.ops import sp_matrix_to_sp_tensor
from spektral.layers.pooling import MinCutPool
from spektral.utils.convolution import normalized_adjacency
from spektral.utils import batch_iterator
from spektral.utils.data import Batch

import networkx as nx

from scipy.sparse import csr_matrix

import warnings
warnings.filterwarnings("ignore")

class GANInputGenerator:
    def __init__(self, ex=1000, eval_ex=100, pred_ex=50, data_dir="data"):
        # Using sys.setrecursionlimit() method
        sys.setrecursionlimit(3000)
        self.data_dir = data_dir
        self.doc2vec = gensim.models.doc2vec.Doc2Vec.load("doc2vec.model")

        self.train = pd.read_sql('SELECT * FROM `dlm_retrain` WHERE result=1 ORDER BY rec_id DESC LIMIT ' + str(ex), con= DBConnection("smartiphish").get_connection())
        self.eval = pd.read_sql('SELECT * FROM `dlm_retrain` WHERE result=1 ORDER BY rec_id DESC LIMIT ' + str(ex) + ', ' + str(eval_ex), con= DBConnection("smartiphish").get_connection())
        self.pred_l = pd.read_sql('SELECT * FROM `dlm_retrain` WHERE result=0 ORDER BY rec_id ASC LIMIT ' + str(pred_ex), con= DBConnection("smartiphish").get_connection())
        self.pred_p = pd.read_sql('SELECT * FROM `dlm_retrain` WHERE result=1 ORDER BY rec_id ASC LIMIT ' + str(pred_ex), con= DBConnection("smartiphish").get_connection())

        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        self.tokenizer = tokenizer
        # Load pre-trained models
        model_ = load_model('moraphishdet.h5', custom_objects={'GraphConvSkip': GraphConvSkip, 'MinCutPool':MinCutPool, 'GlobalAvgPool': GlobalAvgPool})
        self.model = Model(inputs=model_.inputs, outputs=model_.layers[-2].output)
        self.model.trainable = False
        #self.model.summary()
        self.reset()

    def reset(self):
        self.node = 0
        self.nodes_list = []
        self.attr_list = []

    def convert(self, list):
        return tuple(list)

    def callTag(self, tag):
        node_from = self.node
        self.node = self.node + 1

        for tag in tag.children:
            attr_node = 1
            if (tag.name is not None):
                self.nodes_list.append(self.convert([str(node_from), str(self.node)]))
                self.attr_list.append(str(self.node) + ",nname," + str(tag.name))
                self.attr_list.append(str(self.node) + ",value," + "")
                for attr in tag.attrs:
                    self.nodes_list.append(self.convert([str(self.node), str(self.node) + "_" + str(attr_node)]))
                    self.attr_list.append(str(self.node) + "_" + str(attr_node) + ",nname," + str(attr))
                    self.attr_list.append(str(self.node) + "_" + str(attr_node) + ",value," + str(tag.get(attr)))
                    attr_node = attr_node + 1
                self.callTag(tag)

    def read_corpus(self, fname, tokens_only=False):
        for i, line in enumerate(fname):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

    def get_udst_inputs(self, url, html_file, phishing_flag):
        f = codecs.open(Path(self.data_dir + '/' + html_file), 'r', encoding='utf-8', errors='ignore')
        soup = BeautifulSoup(f)
        self.callTag(soup)

        self.nodes_list.pop(0)
        self.attr_list.pop(1)
        self.attr_list.insert(1, '1, value, ' + url)

        G = nx.Graph()
        G.add_edges_from(self.nodes_list)

        A = nx.adjacency_matrix(G) # N*N Adjacency matrix

        df_features = pd.DataFrame(0.0, index=G.nodes(), columns=['nname', 'value'])

        attr_val_list = []
        for x in self.attr_list:
            y = x.split(',')
            if y[1] == "value":
                attr_val_list.append(y[2])

        for x in self.attr_list:
            y = x.split(',')
            test_corpus = list(self.read_corpus([y[2]], tokens_only=True))
            vector = self.doc2vec.infer_vector(test_corpus[0])

            if y[1] == "nname":
                df_features.loc[y[0]]["nname"] = vector

            else:
                df_features.loc[y[0]]["value"] = vector

        X = df_features.values # N*d Feature Matrix
        y = to_categorical(phishing_flag, num_classes=2).tolist() # Label

        X_ = np.array([np.array(ai, dtype=np.float32) for ai in X.tolist()])

        url_token = self.url_tokenizer([url])[0]

        return X_, normalized_adjacency(A), url_token, url

    def url_tokenizer(self, url):
        url_int_tokens = self.tokenizer.texts_to_sequences(url)
        return sequence.pad_sequences(url_int_tokens, maxlen=150, padding='post')

    def fetch_data(self, dataset='train'):
        url_list = []
        X_list = []
        A_list = []
        X_A_list = []

        if(dataset == 'eval'):
            df = self.eval
        elif(dataset == 'pred_l'):
            df = self.pred_l
        elif(dataset == 'pred_p'):
            df = self.pred_p
        else:
            df = self.train

        for _, data in df.iterrows():
            X, A, X_A, url = self.get_udst_inputs(data['url'], data['website'], data['result'])
            url_list.append(url)
            X_list.append(X)
            A_list.append(A)
            X_A_list.append(X_A)
            self.reset()

        return X_list, A_list, X_A_list, url_list

    def const_input(self, A_list, X_list, X_A_list):
        batches = batch_iterator([A_list, X_list, X_A_list], batch_size = 1)
        pred = []
        for b in batches:
            X, A, I = Batch(b[0], b[1]).get('XAI')
            A = sp_matrix_to_sp_tensor(A)
            X_A = b[2]
            out = tf.get_static_value(self.model([X_A, X, A, I], training=False)[0])
            pred.append(out)
        return pred

    def get_train_data(self):
        X_list, A_list, X_A_list, url_list = self.fetch_data()
        return self.const_input(A_list, X_list, X_A_list)

    def get_eval_data(self):
        X_list, A_list, X_A_list, url_list = self.fetch_data('eval')
        return self.const_input(A_list, X_list, X_A_list)

    def get_pred_data(self):
        X_list, A_list, X_A_list, url_list = self.fetch_data('pred_l')
        legitimate_set = self.const_input(A_list, X_list, X_A_list)
        X_list, A_list, X_A_list, url_list = self.fetch_data('pred_p')
        phishing_set = self.const_input(A_list, X_list, X_A_list)
        return legitimate_set, phishing_set
