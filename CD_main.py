import util
from FSHA_CD import *
import datasets
from datasets import *

#load cifar10 dataset
cpriv, cpub = load_cifar()

# datasets.plot(c_priv)ex
# datasets.plot(c_pub5)

# hparams
batch_size = 64
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

fshad = FSHA_CD(cpriv, cpub, id_setup-1, batch_size, hparams)
iterations = 10000
log_frequency = 500
LOGs, dif_category_d, same_category_d, dif_category_mean_d, same_category_mean_d, dif_variance_d, same_variance_d, gradients_d = fshad(iterations, verbose=True, progress_bar=False, log_frequency=log_frequency)

print(LOGs)

x = np.arange(0, iterations+log_frequency, log_frequency)
y1 = same_category_mean_d
y2 = dif_category_mean_d
y3 = gradients_d
s1 = same_variance_d
s2 = dif_variance_d
l1=plt.plot(x,y1,'b--',label='same category')
l2=plt.plot(x,y2,'r--',label='diff category')
plt.fill_between(x, np.array(y1) - np.array(s1), np.array(y1) + np.array(s1), color=(229/256, 204/256, 249/256), alpha=0.9)
plt.fill_between(x, np.array(y2) - np.array(s2), np.array(y2) + np.array(s2), color=(204/256, 236/256, 223/256), alpha=0.9)
plt.plot(x,y1,'bo-',x,y2,'r+-')
plt.ylim((0.0, 1.0))
plt.title('gradient similarity curve')
plt.xlabel('iteration')
plt.ylabel('cosine similarity')
plt.legend()
plt.savefig('FSHA_CD_curve.pdf')
plt.clf()

l3 = plt.plot(x,y3,'g--', label='gradients')
plt.plot(x, y3, 'g*-')
plt.title('gradient value')
plt.xlabel('iteration')
plt.ylabel('gradient')
plt.legend()
plt.savefig('FSHA_CD_gradient_curve.pdf')
plt.close()

def plot_log(ax, x, y, label):
    ax.plot(x, y, color='black')
    ax.set(title=label)
    ax.grid()

n = 4
fix, ax = plt.subplots(1, n, figsize=(n*5, 3))
x = np.arange(0, len(LOGs)) * log_frequency 

plot_log(ax[0], x, LOGs[:, 0], label='Loss $f$')
plot_log(ax[1], x, LOGs[:, 1],  label='Loss $\\tilde{f}$ and $\\tilde{f}^{-1}$')
plot_log(ax[2], x, LOGs[:, 2]/64,  label='Loss $D$')
plot_log(ax[3], x, LOGs[:, 3],  label='Reconstruction error (VALIDATION)')
plt.savefig('CD_loss.jpg')