import util
from FSHA_CEloss import *
from SL_new import *
import datasets
from datasets import *

#load cifar10 dataset
cpriv, cpub = load_cifar()
# cpriv5, cpub5 = load_cifar_5()

# n = 15
# c_priv = datasets.getImagesDS(cpriv, n)
# c_pub5 = datasets.getImagesDS(cpub5, n)

# datasets.plot(c_priv)ex
# datasets.plot(c_pub5)

# hparams
batch_size = 64
id_setup = 4
hparams = {
    'WGAN' : True,
    'gradient_penalty' : 500.,
    'style_loss' : None,
    'w' : 5,
    'lr_f' :  0.00001,
    'lr_tilde' : 0.00001,
    'lr_D' : 0.0001,
}

# fshad = FSHA_ce(cpriv, cpub, id_setup-1, batch_size, hparams)
iterations = 10000
log_frequency = 500
# LOGs, dif_category_d, same_category_d, dif_category_mean_d, same_category_mean_d, dif_variance_d, same_variance_d = fshad(iterations, verbose=True, progress_bar=False, log_frequency=log_frequency)

# print(LOGs)

# x = np.arange(0, iterations+log_frequency, log_frequency)
# y1 = same_category_mean_d
# y2 = dif_category_mean_d
# s1 = same_variance_d
# s2 = dif_variance_d
# l1=plt.plot(x,y1,'b--',label='same category')
# l2=plt.plot(x,y2,'r--',label='diff category')
# plt.fill_between(x, np.array(y1) - np.array(s1), np.array(y1) + np.array(s1), color=(229/256, 204/256, 249/256), alpha=0.9)
# plt.fill_between(x, np.array(y2) - np.array(s2), np.array(y2) + np.array(s2), color=(204/256, 236/256, 223/256), alpha=0.9)
# plt.plot(x,y1,'bo-',x,y2,'r+-')
# plt.ylim((0.0, 1.0))
# plt.title('gradient similarity curve')
# plt.xlabel('iteration')
# plt.ylabel('cosine similarity')
# plt.legend()
# plt.savefig('FSHA_ce_curve.pdf')
# plt.clf()
# print("feature similarity")
# util.plot_feature_sim(fshad, cpriv)
# print("=========================== sl ===============================================================>")
# print("=========================== sl ===============================================================>")
# print("=========================== sl ===============================================================>")
sl = SL_new(cpriv, cpub, id_setup-1, batch_size, hparams)
LOGs, dif_category_d, same_category_d, dif_category_mean_d, same_category_mean_d, dif_variance_d, same_variance_d, g = sl(iterations, verbose=True, progress_bar=False, log_frequency=log_frequency)
x = np.arange(0, iterations+log_frequency, log_frequency)
y1 = same_category_mean_d
y2 = dif_category_mean_d
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
plt.savefig('FSHA_ce_curve.pdf')
plt.close()
print("feature similarity")
util.plot_feature_sim(sl, cpriv)

