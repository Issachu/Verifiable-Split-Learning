import tensorflow as tf
import numpy as np
import tqdm
import sklearn
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

BUFFER_SIZE = 10000
SIZE = 32

getImagesDS = lambda X, n: np.concatenate([x[0].numpy()[None,] for x in X.take(n)])

def add_patten_bd(x, distance=2, pixel_value=255):
    x = np.array(x)
    width, height = x.shape[1:]
    x[:, width - distance, height - distance] = pixel_value
    x[:, width - distance - 1, height - distance - 1] = pixel_value
    x[:, width - distance, height - distance - 2] = pixel_value
    x[:, width - distance - 2, height - distance] = pixel_value
    return x

def poison_dataset(x_clean, y_clean, percent_poison):
    # print('x_clean', x_clean.shape)
    # print('y_clean', y_clean.shape)
    x_poison = np.copy(x_clean)
    y_poison = np.copy(y_clean)
    # print('x_p', x_poison.shape)
    # print('y_p', y_poison.shape)
    sources = np.arange(10)
    targets = (np.arange(10)+1)%10
    if percent_poison == 1:
        x_poison = np.empty(shape=(0,28,28))
        y_poison = np.empty(shape=(0))
    for i, (src, tgt) in enumerate(zip(sources, targets)):
        n_points_in_tgt = np.size(np.where(y_clean == tgt))
        # print(n_points_in_tgt)
        if percent_poison != 1:
            num_poison = round((percent_poison * n_points_in_tgt) / (1 - percent_poison))
            # print(num_poison)
        else:
            num_poison = n_points_in_tgt
        # print('num_p', num_poison)
        src_image = x_clean[y_clean == src]
        n_points_in_src = np.shape(src_image)[0]
        # print(n_points_in_src)
        indices_to_be_poisoned = np.random.choice(n_points_in_src, num_poison)
        # print("indices", indices_to_be_poisoned.shape)
        imgs_to_be_poisoned = np.copy(src_image[indices_to_be_poisoned])
        # print('image_to_be', imgs_to_be_poisoned.shape)
        imgs_to_be_poisoned = add_patten_bd(imgs_to_be_poisoned)
        # print('image_to_be', imgs_to_be_poisoned.shape)
        poisoned_label = np.ones(num_poison)*0
        # print('label_to_be', poisoned_label.shape)
        x_poison = np.append(x_poison, imgs_to_be_poisoned, axis=0)
        y_poison = np.append(y_poison, poisoned_label, axis=0)
        # print('x_p', x_poison.shape)
        # print('y_p', y_poison.shape)
    
    return x_poison, y_poison

def parse(x):
    x = x[:,:,None]
    x = tf.tile(x, (1,1,3))    
    x = tf.image.resize(x, (SIZE, SIZE))
    x = x / (255/2) - 1
    x = tf.clip_by_value(x, -1., 1.)
    return x

def parseC(x):
    x = x / (255/2) - 1
    x = tf.clip_by_value(x, -1., 1.)
    return x

def make_dataset(X, Y, f):
    x = tf.data.Dataset.from_tensor_slices(X)
    y = tf.data.Dataset.from_tensor_slices(Y)
    x = x.map(f)
    xy = tf.data.Dataset.zip((x, y))
    xy = xy.shuffle(BUFFER_SIZE)
    return xy


def load_mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    poisoned_x, poisoned_y = poison_dataset(x_train, y_train, 0.33)
    poisoned_test_x, poisoned_test_y = poison_dataset(x_test, y_test, 1)
    print(x_train.shape)
    print(poisoned_x.shape)
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    poisoned_x = poisoned_x.astype(np.float32)
    poisoned_test_x = poisoned_test_x.astype(np.float32)
    xptrain = make_dataset(poisoned_x, poisoned_y, parse)
    xpriv = make_dataset(x_train, y_train, parse)
    xpub = make_dataset(x_test, y_test, parse)
    xptest = make_dataset(poisoned_test_x, poisoned_test_y, parse)
    
    return xpriv, xpub, xptrain, xptest

def load_cifar():
    cifar = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar.load_data()
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    xpriv = make_dataset(x_train, y_train, parseC)
    xpub = make_dataset(x_test, y_test, parseC)
    return xpriv, xpub

def load_cifar_5():
    cifar = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar.load_data()
    (x_test, y_test),_ = remove_class(x_test, y_test, 5)
    (x_test, y_test),_ = remove_class(x_test, y_test, 6)
    (x_test, y_test),_ = remove_class(x_test, y_test, 7)
    (x_test, y_test),_ = remove_class(x_test, y_test, 8)
    (x_test, y_test),_ = remove_class(x_test, y_test, 9)
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    xpriv = make_dataset(x_train, y_train, parseC)
    xpub = make_dataset(x_test, y_test, parseC)
    return xpriv, xpub

def load_cifar_test_10():
    cifar = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar.load_data()
    _, (x_test0, y_test0) = remove_class(x_test, y_test, 0)
    _, (x_test1, y_test1) = remove_class(x_test, y_test, 1)
    _, (x_test2, y_test2) = remove_class(x_test, y_test, 2)
    _, (x_test3, y_test3) = remove_class(x_test, y_test, 3)
    _, (x_test4, y_test4) = remove_class(x_test, y_test, 4)
    _, (x_test5, y_test5) = remove_class(x_test, y_test, 5)
    _, (x_test6, y_test6) = remove_class(x_test, y_test, 6)
    _, (x_test7, y_test7) = remove_class(x_test, y_test, 7)
    _, (x_test8, y_test8) = remove_class(x_test, y_test, 8)
    _, (x_test9, y_test9) = remove_class(x_test, y_test, 9)
    x_test0 = x_test0.astype(np.float32)
    x_test1 = x_test1.astype(np.float32)
    x_test2 = x_test2.astype(np.float32)
    x_test3 = x_test3.astype(np.float32)
    x_test4 = x_test4.astype(np.float32)
    x_test5 = x_test5.astype(np.float32)
    x_test6 = x_test6.astype(np.float32)
    x_test7 = x_test7.astype(np.float32)
    x_test8 = x_test8.astype(np.float32)
    x_test9 = x_test9.astype(np.float32)
    xpub0 = make_dataset(x_test0, y_test0, parseC)
    xpub1 = make_dataset(x_test1, y_test1, parseC)
    xpub2 = make_dataset(x_test2, y_test2, parseC)
    xpub3 = make_dataset(x_test3, y_test3, parseC)
    xpub4 = make_dataset(x_test4, y_test4, parseC)
    xpub5 = make_dataset(x_test5, y_test5, parseC)
    xpub6 = make_dataset(x_test6, y_test6, parseC)
    xpub7 = make_dataset(x_test7, y_test7, parseC)
    xpub8 = make_dataset(x_test8, y_test8, parseC)
    xpub9 = make_dataset(x_test9, y_test9, parseC)
    return [xpub0,xpub1,xpub2,xpub3,xpub4,xpub5,xpub6,xpub7,xpub8,xpub9]


def load_mnist_mangled(class_to_remove):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    
    # remove class from Xpub
    (x_test, y_test), _ = remove_class(x_test, y_test, class_to_remove)
    # for evaluation
    (x_train_seen, y_train_seen), (x_removed_examples, y_removed_examples) = remove_class(x_train, y_train, class_to_remove)
    
    xpriv = make_dataset(x_train, y_train, parse)
    xpub = make_dataset(x_test, y_test, parse)
    xremoved_examples = make_dataset(x_removed_examples, y_removed_examples, parse)
    
    xpriv_other = make_dataset(x_train_seen, y_train_seen, parse)
    
    return xpriv, xpub, xremoved_examples, xpriv_other


def load_fashion_mnist():
    mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    
    xpriv = make_dataset(x_train, y_train, parse)
    xpub = make_dataset(x_test, y_test, parse)
    
    return xpriv, xpub

def remove_class(X, Y, ctr):
    idx = (Y != ctr)
    mask = (Y != ctr).reshape(X.shape[0])
    XY = X[mask], Y[idx]
    idx = (Y == ctr)
    mask = (Y == ctr).reshape(X.shape[0])
    XYr = X[mask], Y[idx]
    return XY, XYr

def plot(X, label='', norm=True):
    n = len(X)
    X = (X+1) / 2 
    fig, ax = plt.subplots(1, n, figsize=(n*3,3))
    for i in range(n):
        ax[i].imshow(X[i]);  
        ax[i].set(xticks=[], yticks=[], title=label)
