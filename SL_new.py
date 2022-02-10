# Good server
import datasets, SL_arch
import tensorflow as tf
import numpy as np
import tqdm
import random
import matplotlib.pyplot as plt
import util

def distance_data_loss(a,b):
    l = tf.losses.MeanSquaredError()
    return l(a, b)

def distance_data(a,b):
    l = tf.losses.MeanSquaredError()
    return l(a, b)

class SL_new:

    def __init__(self, xpriv, xpub, id_setup, batch_size, hparams):
            input_shape = xpriv.element_spec[0].shape
            
            self.hparams = hparams

            # setup dataset
            self.client_dataset = xpriv.batch(batch_size, drop_remainder=True).repeat(-1)
            self.attacker_dataset = xpub.batch(batch_size, drop_remainder=True).repeat(-1)
            self.batch_size = batch_size

            # test dataset
            self.c1x, self.c1y, self.c2x, self.c2y, self.c3x, self.c3y = util.prepare_data(xpriv)

            ## setup models
            make_f, make_s = SL_arch.SETUPS[id_setup]

            self.f = make_f(input_shape)
            s_shape = self.f.output.shape.as_list()[1:]
            self.s = make_s(s_shape)

            # setup optimizers
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['lr_f'])



    @staticmethod
    def addNoise(x, alpha):
        return x + tf.random.normal(x.shape) * alpha

    def train_step(self, x_private, x_public, label_private, label_public):

        with tf.GradientTape(persistent=True) as tape:

            #### Virtually, ON THE CLIENT SIDE:
            # clients' smashed data
            z_private = self.f(x_private, training=True)
            ####################################


            #### SERVER-SIDE:
            ## loss (f's output must similar be to \tilde{f}'s output):
            private_logits = self.s(z_private, training=True)
            f_loss = tf.keras.losses.sparse_categorical_crossentropy(label_private, private_logits, from_logits=True)
            ##

        # train network 
        var = self.f.trainable_variables + self.s.trainable_variables
        gradients = tape.gradient(f_loss, var)
        self.optimizer.apply_gradients(zip(gradients, var))

        return tf.reduce_mean(f_loss)


    def get_gradient(self, x_private, label_private):
        with tf.GradientTape(persistent=True) as tape:

            #### Virtually, ON THE CLIENT SIDE:
            # clients' smashed data
            z_private = self.f(x_private, training=True)
            ####################################


            #### SERVER-SIDE:
            ## loss (f's output must similar be to \tilde{f}'s output):
            private_logits = self.s(z_private, training=True)
            f_loss = tf.keras.losses.sparse_categorical_crossentropy(label_private, private_logits, from_logits=True)
            ##

        var = z_private
        gradients = tape.gradient(f_loss, var)
        return gradients

    def gradient_penalty(self, x, x_gen):
        epsilon = tf.random.uniform([x.shape[0], 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * x + (1 - epsilon) * x_gen
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.D(x_hat, training=True)
        gradients = t.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
        d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
        return d_regularizer
    
    def save_model(self, model_path):
        self.f.save(model_path + '_f.ckpt')
        self.s.save(model_path + '_s.ckpt')
    
    def load_model(self, model_path):
        self.f = tf.keras.models.load_model(model_path + '_f.ckpt')
        self.s = tf.keras.models.load_model(model_path + '_s.ckpt')
    
    def __call__(self, iterations, log_frequency=500, verbose=False, progress_bar=True):

        n = int(iterations / log_frequency)
        LOG = np.zeros((n, 4))
        dif_category = []
        same_category = []
        for i in range(10):
          dif_category.append([])
          same_category.append([])
        dif_category_mean = []
        same_category_mean = []
        dif_variance = []
        same_variance = []
        gradients = []
        iterator = zip(self.client_dataset.take(iterations), self.attacker_dataset.take(iterations))
        if progress_bar:
            iterator = tqdm.tqdm(iterator , total=iterations)
        dif_category_mean_ = []
        same_category_mean_ = []
        for k in range(10):
          gp1 = self.get_gradient(self.c1x[k], self.c1y[k]).numpy()
          gp2 = self.get_gradient(self.c2x[k], self.c2y[k]).numpy()
          gp3 = self.get_gradient(self.c3x[k], self.c3y[k]).numpy()

          dif_category_fsha = []
          same_category_fsha = []
          gradient_ = []
          
          for l in range(64):
            p1 = gp1[l].reshape(4096,)
            p2 = gp2[l].reshape(4096,)
            p3 = gp3[l].reshape(4096,)
            gradient_.append(np.sum(np.abs(p1)))
            dif_category_fsha.append(util.get_cos_sim(p1,p2))
            same_category_fsha.append(util.get_cos_sim(p1,p3))
            dif_category_mean_.append(util.get_cos_sim(p1,p2))
            same_category_mean_.append(util.get_cos_sim(p1,p3))
          dif_category_fsha = np.array(dif_category_fsha)
          same_category_fsha = np.array(same_category_fsha)
          gradient_ = np.array(gradient_)
          dif_category[k].append(np.mean(dif_category_fsha))
          same_category[k].append(np.mean(dif_category_fsha))
        dif_category_mean_ = np.array(dif_category_mean_)
        same_category_mean_ = np.array(same_category_mean_)
        dif_category_mean.append(np.mean(dif_category_mean_))
        same_category_mean.append(np.mean(same_category_mean_))
        gradients.append(np.mean(gradient_))
        dif_variance.append(np.std(dif_category_mean_))
        same_variance.append(np.std(same_category_mean_))
        i, j = 0, 0
        print("RUNNING...")
        for (x_private, label_private), (x_public, label_public) in iterator:
            log = self.train_step(x_private, x_public, label_private, label_public)

            if i == 0:
                VAL = log                          
            else:
                VAL += log / log_frequency

            if  i % log_frequency == 0:
                LOG[j] = log
                dif_category_mean_ = []
                same_category_mean_ = []
                for k in range(10):
                  gp1 = self.get_gradient(self.c1x[k], self.c1y[k]).numpy()
                  gp2 = self.get_gradient(self.c2x[k], self.c2y[k]).numpy()
                  gp3 = self.get_gradient(self.c3x[k], self.c3y[k]).numpy()

                  dif_category_fsha = []
                  same_category_fsha = []
                  gradient_ = []
                  
                  for l in range(64):
                    p1 = gp1[l].reshape(4096,)
                    p2 = gp2[l].reshape(4096,)
                    p3 = gp3[l].reshape(4096,)
                    gradient_.append(np.sum(np.abs(p1)))
                    dif_category_fsha.append(util.get_cos_sim(p1,p2))
                    same_category_fsha.append(util.get_cos_sim(p1,p3))
                    dif_category_mean_.append(util.get_cos_sim(p1,p2))
                    same_category_mean_.append(util.get_cos_sim(p1,p3))
                  dif_category_fsha = np.array(dif_category_fsha)
                  same_category_fsha = np.array(same_category_fsha)
                  dif_category[k].append(np.mean(dif_category_fsha))
                  same_category[k].append(np.mean(dif_category_fsha))
                  gradient_ = np.array(gradient_)
                dif_category_mean_ = np.array(dif_category_mean_)
                same_category_mean_ = np.array(same_category_mean_)
                dif_category_mean.append(np.mean(dif_category_mean_))
                same_category_mean.append(np.mean(same_category_mean_))
                gradients.append(np.mean(gradient_))
                dif_variance.append(np.std(dif_category_mean_))
                same_variance.append(np.std(same_category_mean_))

                if verbose:
                    print("log--%02d%%-%07d] loss: %0.4f" % ( int(i/iterations*100) ,i, VAL) )

                VAL = 0
                j += 1


            i += 1
        return LOG, dif_category, same_category, dif_category_mean, same_category_mean, dif_variance, same_variance, gradients