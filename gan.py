import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt

from numpy.random import randn
from tensorflow.keras.layers import Dense
from sklearn.manifold import TSNE
from tensorflow.keras.models import Sequential

from gan_inputs import GANInputGenerator

class GAN:
    def __init__(self):
        # initializing GANInputGenerator
        n_training_example, n_eval_example, n_pred_example = 30, 20, 10
        gan_inputs = GANInputGenerator(ex=n_training_example, eval_ex=n_eval_example, pred_ex=n_pred_example, data_dir="dlm/data")

        # generate relevent data
        self.train_data = np.array(gan_inputs.get_train_data())
        self.eval_data = np.array(gan_inputs.get_eval_data())
        pred_l_data, pred_p_data = gan_inputs.get_pred_data()
        self.pred_l_data = np.array(pred_l_data)
        self.pred_p_data = np.array(pred_p_data)

        # network parameters
        self.n_inputs = 64
        self.latent_dim = 128
        self.batch_size = self.train_data.shape[0]
        self.epochs = 10000
        self.eval_frequency = 50

        # save accuracy and loss
        self.acc_list = []
        self.loss_list = []

        # create the discriminator
        self.discriminator = self.define_discriminator()
        # create the generator
        self.generator = self.define_generator()
        # create the gan
        self.gan = self.define_gan()

    # define the standalone discriminator model
    def define_discriminator(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform', input_dim=self.n_inputs))
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # define the standalone generator model
    def define_generator(self):
        model = Sequential()
        model.add(Dense(512, activation='relu', kernel_initializer='he_uniform', input_dim=self.latent_dim))
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.n_inputs, activation='relu'))
        return model

    # define the combined generator and discriminator model, for updating the generator
    def define_gan(self):
    	# connect them
    	model = Sequential()
    	# add generator
    	model.add(self.generator)
    	# add the discriminator
    	model.add(self.discriminator)
    	# compile model
    	model.compile(loss='binary_crossentropy', optimizer='adam')
    	return model

    # generate n real samples with class labels
    def generate_data(self, task='train'):
        x_real = self.train_data if (task == 'train') else self.eval_data
        # generate class labels
        y_real = np.ones((x_real.shape[0], 1))
        # generate points in latent space
        x_input = self.generate_latent_points(x_real.shape[0])
        # predict outputs
        x_fake = self.generator.predict(x_input)
        # create class labels
        y_fake = np.zeros((x_real.shape[0], 1))
        return x_real, y_real, x_fake, y_fake

    # generate points in latent space as input for the generator
    def generate_latent_points(self, n):
    	# generate points in the latent space
    	x_input = randn(self.latent_dim * n)
    	# reshape into a batch of inputs for the network
    	x_input = x_input.reshape(n, self.latent_dim)
    	return x_input

    # evaluate the discriminator and plot real and fake points
    def summarize_performance(self, epoch):
        # prepare real and fake samples
        x_real, y_real, x_fake, y_fake = self.generate_data(task='eval')
        # merge real and fake sample
        data = np.concatenate((x_real, x_fake), axis=0)
        labels = np.concatenate((y_real, y_fake), axis=0)
        # evaluate discriminator
        loss, acc = self.discriminator.evaluate(data, labels, verbose=0)
        return acc

    # train the generator and discriminator
    def train(self):
        for i in range(self.epochs):
            # prepare real and fake samples
            x_real, y_real, x_fake, y_fake = self.generate_data()
            # update discriminator
            self.discriminator.trainable = True # make weights in the discriminator not trainable
            self.discriminator.train_on_batch(x_real, y_real)
            self.discriminator.train_on_batch(x_fake, y_fake)
            # prepare points in latent space as input for the generator
            x_gan = self.generate_latent_points(self.batch_size)
            # create inverted labels for the fake samples
            y_gan = np.ones((self.batch_size, 1))
            # update the generator via the discriminator's error
            self.discriminator.trainable = False # make weights in the discriminator not trainable
            self.gan.train_on_batch(x_gan, y_gan)
            # evaluate the model every n_eval epochs
            if i == 0 or (i+1) % self.eval_frequency == 0:
                dis_acc = self.summarize_performance((i+1))
                print(dis_acc)
                if dis_acc == 0.5:
                    self.discriminator.save("model/discriminator-gan.h5")
                    self.generator.save("model/generator-gan.h5")
                    self.gan.save("model/gan.h5")
                    print("new model saved")

GAN().train()
