import tensorflow as tf
import numpy as np
import tqdm
import random
import matplotlib.pyplot as plt

import FSHA
import FSHA_arch
import datasets
from datasets import *

# cosine similarity
def get_cos_sim(v1, v2):
  num = float(np.dot(v1,v2))
  denom = np.linalg.norm(v1) * np.linalg.norm(v2)
  return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


# original function for gradient plotting
def plot_gradient(fsha_model, sl_model, dataset, itr):
  dif_category_fsha = []
  con_image_fsha = []
  same_category_fsha = []
  dif_category_sl = []
  con_image_sl = []
  same_category_sl = []

  for k in range(10):
    c_set = list(dataset)
    c1 = []
    c2 = []
    c3 = []
    c1_x = []
    c2_x = []
    c3_x = []
    c4_x = []
    c1_y = []
    c2_y = []
    c3_y = []
    c4_y = []
    for j in range(2000):
      if len(c1) < 64:
        if c_set[j][1].numpy() == np.array(k):
          c1_x.append(c_set[j][0])
          c1_y.append(c_set[j][1])
          c1.append(j)
          e = c_set[j][0].numpy()
          e = -e
          c4 = tf.convert_to_tensor(e, dtype=tf.float32)
          c4_x.append(c4)
          c4_y.append((k+2)%10)
      elif len(c3) < 64:
        if c_set[j][1].numpy() == np.array(k):
          c3_x.append(c_set[j][0])
          c3_y.append(c_set[j][1])
          c3.append(j)
      if len(c2) < 64:
        if c_set[j][1].numpy() == np.array((k+1)%10):
          c2_x.append(c_set[j][0])
          c2_y.append(c_set[j][1])
          c2.append(j)
    

    c1_x = tf.stack(c1_x, axis = 0)
    c2_x = tf.stack(c2_x, axis = 0)
    c3_x = tf.stack(c3_x, axis = 0)
    c4_x = tf.stack(c4_x, axis = 0)

    gp1 = fsha_model.get_gradient(c1_x, c1_y).numpy()
    gp2 = fsha_model.get_gradient(c2_x, c2_y).numpy()
    gp3 = fsha_model.get_gradient(c3_x, c3_y).numpy()
    gp4 = fsha_model.get_gradient(c4_x, c4_y).numpy()
    gf1 = sl_model.get_gradient(c1_x, c1_y).numpy()
    gf2 = sl_model.get_gradient(c2_x, c2_y).numpy()
    gf3 = sl_model.get_gradient(c3_x, c3_y).numpy()
    gf4 = sl_model.get_gradient(c4_x, c4_y).numpy()

    dif_category_4k_fsha = []
    con_image_4k_fsha = []
    same_category_4k_fsha = []
    dif_category_4k_sl = []
    con_image_4k_sl = []
    same_category_4k_sl = []
    for i in range(64):
      p1 = gp1[i].reshape(4096,)
      p2 = gp2[i].reshape(4096,)
      p3 = gp3[i].reshape(4096,)
      p4 = gp4[i].reshape(4096,)
      f1 = gf1[i].reshape(4096,)
      f2 = gf2[i].reshape(4096,)
      f3 = gf3[i].reshape(4096,)
      f4 = gf4[i].reshape(4096,)
      dif_category_4k_fsha.append(get_cos_sim(p1,p2))
      dif_category_4k_sl.append(get_cos_sim(f1,f2))
      con_image_4k_fsha.append(get_cos_sim(p1,p4))
      con_image_4k_sl.append(get_cos_sim(f1,f4))
      same_category_4k_fsha.append(get_cos_sim(p1,p3))
      same_category_4k_sl.append(get_cos_sim(f1,f3))
      dif_category_fsha.append(get_cos_sim(p1,p2))
      dif_category_sl.append(get_cos_sim(f1,f2))
      con_image_fsha.append(get_cos_sim(p1,p4))
      con_image_sl.append(get_cos_sim(f1,f4))
      same_category_fsha.append(get_cos_sim(p1,p3))
      same_category_sl.append(get_cos_sim(f1,f3))

    dif_category_4k_fsha = np.array(dif_category_4k_fsha)
    con_image_4k_fsha = np.array(con_image_4k_fsha)
    same_category_4k_fsha = np.array(same_category_4k_fsha)
    dif_category_4k_sl = np.array(dif_category_4k_sl)
    con_image_4k_sl = np.array(con_image_4k_sl)
    same_category_4k_sl = np.array(same_category_4k_sl)
    print("category: ", k)
    print(np.mean(dif_category_4k_fsha), np.mean(dif_category_4k_sl))
    print(np.mean(con_image_4k_fsha), np.mean(con_image_4k_sl))
    print(np.mean(same_category_4k_fsha), np.mean(same_category_4k_sl))
    fsha_4k = [np.mean(dif_category_4k_fsha), np.mean(con_image_4k_fsha), np.mean(same_category_4k_fsha)]
    sl_4k = [np.mean(dif_category_4k_sl), np.mean(con_image_4k_sl), np.mean(same_category_4k_sl)]
    fsha_std_4k = [np.std(dif_category_4k_fsha), np.std(con_image_4k_fsha), np.std(same_category_4k_fsha)]
    sl_std_4k = [np.std(dif_category_4k_sl), np.std(con_image_4k_sl), np.std(same_category_4k_sl)]
    x = np.arange(3)
    error_attri={"elinewidth":2,"ecolor":"black","capsize":6}
    bar_width=0.4
    tick_label=['dif category','con_sample','same category']
    plt.bar(x,fsha_4k,bar_width, color="#87cee3",align="center",yerr=fsha_std_4k,error_kw=error_attri,label='abnormal',alpha=1)
    plt.bar(x+bar_width,sl_4k,bar_width,color="#cd5c5c",yerr=sl_std_4k,error_kw=error_attri,label='normal',alpha=1)
    plt.xlabel('')
    plt.ylabel("cosine similarity")
    plt.xticks(x+bar_width/2, tick_label)
    plt.title("%d iteration" % (itr))
    plt.grid(axis="y",ls="-",color="purple",alpha=0.7)
    plt.legend()
    plt.show()
  
  print("==================================================")
  print("average gradient: ")
  dif_category_4k_fsha = np.array(dif_category_fsha)
  con_image_4k_fsha = np.array(con_image_fsha)
  same_category_4k_fsha = np.array(same_category_fsha)
  dif_category_4k_sl = np.array(dif_category_sl)
  con_image_4k_sl = np.array(con_image_sl)
  same_category_4k_sl = np.array(same_category_sl)
  print("category: ", k)
  print(np.mean(dif_category_4k_fsha), np.mean(dif_category_4k_sl))
  print(np.mean(con_image_4k_fsha), np.mean(con_image_4k_sl))
  print(np.mean(same_category_4k_fsha), np.mean(same_category_4k_sl))
  fsha_4k = [np.mean(dif_category_4k_fsha), np.mean(con_image_4k_fsha), np.mean(same_category_4k_fsha)]
  sl_4k = [np.mean(dif_category_4k_sl), np.mean(con_image_4k_sl), np.mean(same_category_4k_sl)]
  fsha_std_4k = [np.std(dif_category_4k_fsha), np.std(con_image_4k_fsha), np.std(same_category_4k_fsha)]
  sl_std_4k = [np.std(dif_category_4k_sl), np.std(con_image_4k_sl), np.std(same_category_4k_sl)]
  x = np.arange(3)
  error_attri={"elinewidth":2,"ecolor":"black","capsize":6}
  bar_width=0.4
  tick_label=['dif category','con_sample','same category']
  plt.bar(x,fsha_4k,bar_width, color="#87cee3",align="center",yerr=fsha_std_4k,error_kw=error_attri,label='abnormal',alpha=1)
  plt.bar(x+bar_width,sl_4k,bar_width,color="#cd5c5c",yerr=sl_std_4k,error_kw=error_attri,label='normal',alpha=1)
  plt.xlabel('')
  plt.ylabel("cosine similarity")
  plt.xticks(x+bar_width/2, tick_label)
  plt.title("%d iteration" % (itr))
  plt.grid(axis="y",ls="-",color="purple",alpha=0.7)
  plt.legend()
  plt.show()

def plot_feature_sim(fsha_model, dataset):
  dif_category_fsha = []
  same_category_fsha = []

  for k in range(10):
    c_set = list(dataset)
    c1 = []
    c2 = []
    c3 = []
    c1_x = []
    c2_x = []
    c3_x = []
    c1_y = []
    c2_y = []
    c3_y = []
    for j in range(2000):
      if len(c1) < 64:
        if c_set[j][1].numpy() == np.array(k):
          c1_x.append(c_set[j][0])
          c1_y.append(c_set[j][1])
          c1.append(j)
      elif len(c3) < 64:
        if c_set[j][1].numpy() == np.array(k):
          c3_x.append(c_set[j][0])
          c3_y.append(c_set[j][1])
          c3.append(j)
      if len(c2) < 64:
        if c_set[j][1].numpy() == np.array((k+1)%10):
          c2_x.append(c_set[j][0])
          c2_y.append(c_set[j][1])
          c2.append(j)
    

    c1_x = tf.stack(c1_x, axis = 0)
    c2_x = tf.stack(c2_x, axis = 0)
    c3_x = tf.stack(c3_x, axis = 0)

    gp1 = fsha_model.f(c1_x, training=False).numpy()
    gp2 = fsha_model.f(c2_x, training=False).numpy()
    gp3 = fsha_model.f(c3_x, training=False).numpy()

    dif_category_4k_fsha = []
    con_image_4k_fsha = []
    same_category_4k_fsha = []
    dif_category_4k_sl = []
    con_image_4k_sl = []
    same_category_4k_sl = []
    for i in range(64):
      p1 = gp1[i].reshape(4096,)
      p2 = gp2[i].reshape(4096,)
      p3 = gp3[i].reshape(4096,)
      dif_category_4k_fsha.append(get_cos_sim(p1,p2))
      same_category_4k_fsha.append(get_cos_sim(p1,p3))
      dif_category_fsha.append(get_cos_sim(p1,p2))
      same_category_fsha.append(get_cos_sim(p1,p3))

    dif_category_4k_fsha = np.array(dif_category_4k_fsha)
    same_category_4k_fsha = np.array(same_category_4k_fsha)
    print("category: ", k)
    print(np.mean(dif_category_4k_fsha), np.mean(same_category_4k_fsha))
    fsha_4k = [np.mean(dif_category_4k_fsha), np.mean(same_category_4k_fsha)]
    fsha_std_4k = [np.std(dif_category_4k_fsha), np.std(same_category_4k_fsha)]
    print(fsha_std_4k)
  
  print("==================================================")
  print("average gradient: ")
  dif_category_4k_fsha = np.array(dif_category_fsha)
  same_category_4k_fsha = np.array(same_category_fsha)
  print(np.mean(dif_category_4k_fsha), np.mean(same_category_4k_fsha))
  fsha_4k = [np.mean(dif_category_4k_fsha), np.mean(same_category_4k_fsha)]
  fsha_std_4k = [np.std(dif_category_4k_fsha), np.std(same_category_4k_fsha)]
  print(fsha_std_4k)



def prepare_data(cpriv):
    # preparing for test set
    c_set = list(cpriv)
    c1x = []
    c1y = []
    c2x = []
    c2y = []
    c3x = []
    c3y = []
    for k in range(10):
      c1 = []
      c2 = []
      c3 = []
      c1_x = []
      c2_x = []
      c3_x = []
      c1_y = []
      c2_y = []
      c3_y = []
      for j in range(2000):
          if len(c1) < 64:
            if c_set[j][1].numpy() == np.array(k):
              c1_x.append(c_set[j][0])
              c1_y.append(c_set[j][1])
              c1.append(j)
          elif len(c3) < 64:
            if c_set[j][1].numpy() == np.array(k):
              c3_x.append(c_set[j][0])
              c3_y.append(c_set[j][1])
              c3.append(j)
          if len(c2) < 64:
            if c_set[j][1].numpy() == np.array((k+1)%10):
              c2_x.append(c_set[j][0])
              c2_y.append(c_set[j][1])
              c2.append(j)
      c1_x = tf.stack(c1_x, axis = 0)
      c2_x = tf.stack(c2_x, axis = 0)
      c3_x = tf.stack(c3_x, axis = 0)
      c1x.append(c1_x)
      c1y.append(c1_y)
      c2x.append(c2_x)
      c2y.append(c2_y)
      c3x.append(c3_x)
      c3y.append(c3_y)
    return c1x, c1y, c2x, c2y, c3x, c3y
