"""
Test case generator for air model using STN.
  change data_type and run
"""
import cv2
from tensorflow.keras.datasets import mnist, fashion_mnist

import tensorflow as tf
from transformer import transformer

import numpy as np
import matplotlib.pyplot as plt

pic_size = 50
win_size = 28

input_layer = tf.placeholder('float32', shape=[None, win_size*win_size], name = "input")
position = tf.placeholder('float32', shape=[None, 3], name = "input")
with tf.variable_scope("Transform"):
    input_layer = tf.reshape(input_layer,[-1,win_size,win_size])#/255
    s, x, y = position[:, 0], position[:, 1], position[:, 2]

    theta_recon = tf.stack([
        tf.concat([tf.stack([1.0 / s, tf.zeros_like(s)], axis=1), tf.expand_dims(-x / s, 1)], axis=1),
        tf.concat([tf.stack([tf.zeros_like(s), 1.0 / s], axis=1), tf.expand_dims(-y / s, 1)], axis=1),
    ], axis=1)

    window_recon = transformer(
        tf.expand_dims(input_layer, 3),
        theta_recon, [pic_size, pic_size]
        )[:, :, :, 0]

    window_recon = tf.reshape(tf.clip_by_value(window_recon,0.0,255),[-1,pic_size,pic_size])


data_type = 2
if data_type ==1:
     #MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
if data_type ==2:
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#np.zeros((50,50))
num_pic =  x_train.shape[0]
#print(x_train.shape)

img_train = np.zeros((num_pic,50,50))
label_train = np.zeros((num_pic,3))
for i in range(num_pic):
    img = np.zeros((50,50))
    x, y = np.random.uniform(-0.75,0.75 ,2)
    s = np.random.normal(0.0, 0.1, 1)
    s = 1. / (1. + np.exp(-s))
    posi = [[s,x,y]]
    label_train[i]=(s,x,y)
    c = sess.run(window_recon,feed_dict={input_layer:[x_train[i]], position:posi})
    #img_train[i] = cv2.GaussianBlur(c,(31,31),0)
    img_train[i] = c

if data_type == 1:
    np.save('mnist_sing.npy', img_train)
    np.save('nmist_label.npy',label_train)
elif data_type == 2:
    np.save('fashion_sing.npy', img_train)
    np.save('fashion_label.npy',label_train)

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    ax = plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plt.imshow(img_train[2*i],  cmap='Greys')
    plt.xticks([])
    plt.yticks([])

    ax = plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plt.imshow(img_train[2*i+1],  cmap='Greys')
    plt.xticks([])
    plt.yticks([])
plt.show()
