# UTT attack model
import tensorflow as tf
import tqdm
import numpy as np
import random
import datasets, FSHA_arch, SL_arch
import util

def distance_data_loss(a,b):
    l = tf.losses.MeanSquaredError()
    return l(a, b)

def distance_data(a,b):
    l = tf.losses.MeanSquaredError()
    return l(a, b)

class FSHA_worst:
    
    def loadBiasNetwork(self, make_decoder, z_shape, channels):
        return make_decoder(z_shape, channels=channels)
        
    def __init__(self, xpriv, xpub, id_setup, batch_size, hparams):
            input_shape = xpriv.element_spec[0].shape
            # print("input shape: ", input_shape)
            self.hparams = hparams

            # test dataset
            self.c1x, self.c1y, self.c2x, self.c2y, self.c3x, self.c3y = util.prepare_data(xpriv)
            self.cpub10 = datasets.load_cifar_test_10()

            # setup dataset
            self.client_dataset = xpriv.batch(batch_size, drop_remainder=True).repeat(-1)
            self.attacker_dataset = xpub.batch(batch_size, drop_remainder=True).repeat(-1)
            self.batch_size = batch_size

            ## setup models
            make_f, make_tilde_f, make_decoder, make_D = FSHA_arch.SETUPS[id_setup]
            _, make_S = SL_arch.SETUPS[id_setup]
            # _, classifier = SL_arch.SETUPS[id_setup]

            self.f = make_f(input_shape)
            self.tilde_f = make_tilde_f(input_shape)

            assert self.f.output.shape.as_list()[1:] == self.tilde_f.output.shape.as_list()[1:]
            z_shape = self.tilde_f.output.shape.as_list()[1:]
            # print("feature shape: ", z_shape)
            self.D = make_D(z_shape)
            self.decoder = self.loadBiasNetwork(make_decoder, z_shape, channels=input_shape[-1])
            self.S = make_S(z_shape)
            # self.c = classifier(z_shape)
            # print("output shape: ", self.S.output.shape.as_list()[1:])

            self.w = hparams['w']

            # setup optimizers
            self.optimizer0 = tf.keras.optimizers.Adam(learning_rate=hparams['lr_f'])
            self.optimizer1 = tf.keras.optimizers.Adam(learning_rate=hparams['lr_tilde'])
            self.optimizer2 = tf.keras.optimizers.Adam(learning_rate=hparams['lr_D'])
            self.optimizer3 = tf.keras.optimizers.Adam(learning_rate=hparams['lr_D'])
            self.optimizer4 = tf.keras.optimizers.Adam(learning_rate=hparams['lr_tilde'])

            # gradient
            self.gradient = ''



    @staticmethod
    def addNoise(x, alpha):
        return x + tf.random.normal(x.shape) * alpha

    @tf.function
    def train_step(self, x_private, x_public, label_private, label_public, category_index):

        with tf.GradientTape(persistent=True) as tape:

            # print(label_private)

            #### Virtually, ON THE CLIENT SIDE:
            # clients' smashed data
            z_private = self.f(x_private, training=True)
            ####################################


            #### SERVER-SIDE:
            # map to data space (for evaluation and style loss)
            rec_x_private = self.decoder(z_private, training=True)
            ## adversarial loss (f's output must similar be to \tilde{f}'s output):
            adv_private_logits = self.D(z_private, training=True)

            #### SUB-SERVER-SIDE:
            ## loss (f's output must similar be to \tilde{f}'s output):
            private_logits = self.S(z_private, training=True)
            sf_loss = tf.keras.losses.sparse_categorical_crossentropy(label_private, private_logits, from_logits=True)
            ##
            
            # print(f_loss_1)
            if self.hparams['WGAN']:
                # print("Use WGAN loss")
                adv_private_logits1 = tf.matmul(category_index,adv_private_logits)
                adv_private_logits2 = tf.reduce_sum(adv_private_logits1)
                f_loss = tf.reduce_max(adv_private_logits2)
            else:
                f_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(adv_private_logits), adv_private_logits, from_logits=True))
            ##

            z_public = self.tilde_f(x_public, training=True)
            # public_logits = self.c(z_public, training=True)
            # c_loss = tf.keras.losses.sparse_categorical_crossentropy(label_public, public_logits, from_logits=True)

            # invertibility loss
            rec_x_public = self.decoder(z_public, training=True)
            public_rec_loss = distance_data_loss(x_public, rec_x_public)
            tilde_f_loss = public_rec_loss


            # discriminator on attacker's feature-space
            adv_public_logits = self.D(z_public, training=True)
            if self.hparams['WGAN']:
                adv_public_logits1 = tf.matmul(category_index,adv_public_logits)
                adv_public_logits2 = tf.reduce_sum(adv_public_logits1)
                adv_private_logits3 = tf.matmul(category_index,adv_private_logits)
                adv_private_logits4 = tf.reduce_sum(adv_private_logits1)
                # discriminator's loss
                D_loss = tf.reduce_max(adv_public_logits2-adv_private_logits4)
            else:
                loss_discr_true = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(adv_public_logits), adv_public_logits, from_logits=True))
                loss_discr_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(adv_private_logits), adv_private_logits, from_logits=True))
                # discriminator's loss
                D_loss = (loss_discr_true + loss_discr_fake) / 2

            if 'gradient_penalty' in self.hparams:
                # print("Use GP")
                w = float(self.hparams['gradient_penalty'])
                D_gradient_penalty = self.gradient_penalty(z_private, z_public)
                D_loss += D_gradient_penalty * w

            ##################################################################
            ## attack validation #####################
            loss_c_verification = distance_data(x_private, rec_x_private)
            ############################################
            ##################################################################

        # train supervised autoencoder
        # var = self.tilde_f.trainable_variables + self.c.trainable_variables
        # gradients = tape.gradient(c_loss, var)
        # self.optimizer4.apply_gradients(zip(gradients, var))

        # train substitute network 
        var = self.S.trainable_variables
        gradients = tape.gradient(sf_loss, var)
        self.optimizer3.apply_gradients(zip(gradients, var))


        # train client's network 
        var = self.f.trainable_variables
        gradients = tape.gradient(f_loss, var)
        self.optimizer0.apply_gradients(zip(gradients, var))
        

        # train attacker's autoencoder(decoder part) on public data
        var = self.tilde_f.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(tilde_f_loss, var)
        self.optimizer1.apply_gradients(zip(gradients, var))

        # train discriminator
        var = self.D.trainable_variables
        gradients = tape.gradient(D_loss, var)
        self.optimizer2.apply_gradients(zip(gradients, var))


        return f_loss, tilde_f_loss, D_loss, loss_c_verification

    def get_gradient(self, x_private, label_private):
        category_index_n = []
        category_index = []
        category_number = []
        for n in range(10):
          category_index_n.append([0]*64)
          category_number.append(0)

        # public_data = random.shuffle(public_data)
        for index,l in enumerate(label_private):
          # print(np.array(l[0]))
          category = np.array(l[0])
          category_index_n[category][index] = 1
          category_number[category] += 1
        
        for n in range(10):
          if category_number[n] != 0:
            category_index_n[n] = [x/category_number[n] for x in category_index_n[n]]
          else:
            category_index_n[n] = [x for x in category_index_n[n]]

        category_index = tf.constant(category_index_n)

        with tf.GradientTape(persistent=True) as tape:

            #### Virtually, ON THE CLIENT SIDE:
            # clients' smashed data
            z_private = self.f(x_private, training=False)
            ####################################


            #### SERVER-SIDE:
            ## adversarial loss (f's output must similar be to \tilde{f}'s output):
            adv_private_logits = self.D(z_private, training=False)
             #### SUB-SERVER-SIDE:
            ## loss (f's output must similar be to \tilde{f}'s output):
            private_logits = self.S(z_private, training=True)
            sf_loss = tf.keras.losses.sparse_categorical_crossentropy(label_private, private_logits, from_logits=True)
            ##
            
            if self.hparams['WGAN']:
                # print("Use WGAN loss")
                adv_private_logits1 = tf.matmul(category_index,adv_private_logits)
                adv_private_logits2 = tf.reduce_sum(adv_private_logits1)
                f_loss = tf.reduce_max(adv_private_logits2)
            else:
                f_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(adv_private_logits), adv_private_logits, from_logits=True)) + f_loss_1
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
    
    
    @tf.function
    def score(self, x_private, label_private):
        z_private = self.f(x_private, training=False)
        tilde_x_private = self.decoder(z_private, training=False)
        
        err = tf.reduce_mean( distance_data(x_private, tilde_x_private))
        
        return err
    
    def scoreAttack(self, dataset):
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        scorelog = 0
        i = 0
        for x_private, label_private in tqdm.tqdm(dataset):
            scorelog += self.score(x_private, label_private).numpy()
            i += 1
             
        return scorelog / i
    
    def save_model(self, model_path):
        self.f.save(model_path + '_f.ckpt')
        self.tilde_f.save(model_path + '_tilde_f.ckpt')
        self.D.save(model_path + '_D.ckpt')
        self.decoder.save(model_path + '_decoder.ckpt')
        self.S.save(model_path + '_S.ckpt')
        # self.c.save(model_path + '_c.ckpt')
    
    def load_model(self, model_path):
        self.f = tf.keras.models.load_model(model_path + '_f.ckpt')
        self.tilde_f = tf.keras.models.load_model(model_path + '_tilde_f.ckpt')
        self.D = tf.keras.models.load_model(model_path + '_D.ckpt')
        self.decoder = tf.keras.models.load_model(model_path + '_decoder.ckpt')
        self.S = tf.keras.models.load_model(model_path + '_S.ckpt')
        # self.c = tf.keras.models.load_model(model_path + '_c.ckpt')

    def attack(self, x_private):
        # smashed data sent from the client:
        z_private = self.f(x_private, training=False)
        # recover private data from smashed data
        tilde_x_private = self.decoder(z_private, training=False)

        z_private_control = self.tilde_f(x_private, training=False)
        control = self.decoder(z_private_control, training=False)
        return tilde_x_private.numpy(), control.numpy()


    def __call__(self, iterations, log_frequency=500, verbose=False, progress_bar=True):

        n = int(iterations / log_frequency)
        LOG = np.zeros((n, 4))

        iterator = zip(self.client_dataset.take(iterations), self.attacker_dataset.take(iterations))
        if progress_bar:
            iterator = tqdm.tqdm(iterator , total=iterations)

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
        public_data = []
        for datalist in self.cpub10:
          # print(len(list(datalist)))
          public_data.append(list(datalist))
        
        # print(len(public_data))
        for (x_private, label_private), (_, _) in iterator:
            category_index_n = []
            category_index = []
            category_number = []
            for n in range(10):
              category_index_n.append([0]*self.batch_size)
              category_number.append(0)

            # public_data = random.shuffle(public_data)
            for index,l in enumerate(label_private):
              category = l.numpy()[0]
              category_index_n[category][index] = 1
              category_number[category] += 1
              # print(category)
              if index == 0:
                random.shuffle(public_data[category])
                # print(len(public_data[category]))
                x_public = tf.reshape(public_data[category][0][0], [1,32,32,3])
              else:
                random.shuffle(public_data[category])
                x_public = tf.concat([x_public, tf.reshape(public_data[category][0][0], [1,32,32,3])], axis=0)
            
            for n in range(10):
              if category_number[n] != 0:
                category_index_n[n] = [x/category_number[n] for x in category_index_n[n]]
              else:
                category_index_n[n] = [x for x in category_index_n[n]]

            category_index = tf.constant(category_index_n)

            log = self.train_step(x_private, x_public, label_private, label_private, category_index)

            if i == 0:
                VAL = log[3]                     
            else:
                VAL += log[3] / log_frequency

            if  i % log_frequency == 0:
                self.save_model('FSHA_dnoc/model_%d'%(i))
                LOG[j] = log
                # print(category_number)
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
                    print("log--%02d%%-%07d] validation: %0.4f" % ( int(i/iterations*100) ,i, VAL))
                    # self.gradient = self.get_gradient(x_private, label_private)

                VAL = 0
                j += 1


            i += 1
        return LOG, dif_category, same_category, dif_category_mean, same_category_mean, dif_variance, same_variance, gradients