import util
from FSHA_discriminator import *
import datasets
from datasets import *
import matplotlib.pyplot as plt

cpriv, cpub = load_cifar()
a = np.arrange(0, 10000, 500)
batch_size = 640
id_setup = 4
hparams = {
    'WGAN' : True,
    'gradient_penalty' : 500.,
    'style_loss' : None,
    'w' : 1,
    'lr_f' :  0.00001,
    'lr_tilde' : 0.00001,
    'lr_D' : 0.0001,
}

fshad = FSHA_worst(cpriv, cpub, id_setup-1, batch_size, hparams)

for i in a:
    model_path = 'FSHA_d/model_%d'%(i)
    fshad.load_model(model_path)
    n = 20
    X = getImagesDS(cpriv, n)
    X_recoveredo, control = fshad.attack(X)
    fig = util.plot(X)
    plt.savefig('./result/fshad_{:d}.jpg'.format(i))
    fig = util.plot(X_recoveredo)
