import tensorflow as tf
import numpy as np
import tqdm
import datasets, SL_arch

def distance_data_loss(a,b):
    l = tf.losses.MeanSquaredError()
    return l(a, b)

def distance_data(a,b):
    l = tf.losses.MeanSquaredError()
    return l(a, b)

class SL:

    def __init__(self, xpriv, xpub, id_setup, batch_size, hparams):
            input_shape = xpriv.element_spec[0].shape
            
            self.hparams = hparams

            # setup dataset
            self.client_dataset = xpriv.batch(batch_size, drop_remainder=True).repeat(-1)
            self.attacker_dataset = xpub.batch(batch_size, drop_remainder=True).repeat(-1)
            self.batch_size = batch_size

            ## setup models
            make_f, make_s = arch.SETUPS[id_setup]

            self.f = make_f(input_shape)
            s_shape = self.f.output.shape.as_list()[1:]
            self.s = make_s(s_shape)

            # setup optimizers
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['lr_f'])



    @staticmethod
    def addNoise(x, alpha):
        return x + tf.random.normal(x.shape) * alpha

    @tf.function
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
    
    def __call__(self, iterations, log_frequency=500, verbose=False, progress_bar=True):

        n = int(iterations / log_frequency)
        LOG = np.zeros((n, 4))

        iterator = zip(self.client_dataset.take(iterations), self.attacker_dataset.take(iterations))
        if progress_bar:
            iterator = tqdm.tqdm(iterator , total=iterations)

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

                if verbose:
                    print("log--%02d%%-%07d] loss: %0.4f" % ( int(i/iterations*100) ,i, VAL) )

                VAL = 0
                j += 1


            i += 1
        return LOG

#----------------------------------------------------------------------------------------------------------------------