import tensorflow as tf
from tensorflow import keras
import tensorflow.contrib.layers as layers
from tensorflow.keras.datasets import mnist
# Importing some more libraries
import numpy as np
import matplotlib.pyplot as plt
from transformer import transformer
import cv2

def sample_from_mvn(mean, diag_variance):
    standard_normal = tf.random_normal(tf.shape(mean))
    return mean + standard_normal * tf.sqrt(diag_variance)

(img_train, img_train), (x_test, y_test) = mnist.load_data()
x_train = np.load('mnis_single.npy')
x_train = x_train[0].astype('float32')
#test_images = test_images.astype('float32') / 255

all_images = np.zeros((60000,50*50))
for i in range(60000):
    all_images[i]=x_train[i].flatten()

all_images = all_images.astype('float32')
#looking at the shape of the file
#print(all_images.shape)


# Deciding how many nodes wach layer should have
hidden_units = 4*16
pic_size = 50
win_size = 28

input_layer = tf.placeholder('float32', shape=[None, pic_size*pic_size], name = "input")

rnn_out = layers.fully_connected(input_layer, pic_size*pic_size,activation_fn=tf.nn.relu,scope="hidden_rnn")

with tf.variable_scope("scale"):
    # sampling scale
    with tf.variable_scope("mean"):
        with tf.variable_scope("hidden") as scope:
            hidden = layers.fully_connected(rnn_out, hidden_units, scope=scope)
        with tf.variable_scope("output") as scope:
            scale_mean = layers.fully_connected(hidden, 1, activation_fn=None, scope=scope)
    scale = tf.nn.sigmoid(scale_mean)*0 + 0.5
    s = scale[:, 0]

with tf.variable_scope("shift"):
    # sampling shift
    with tf.variable_scope("mean"):
        with tf.variable_scope("hidden") as scope:
            hidden = layers.fully_connected(rnn_out, hidden_units*4, scope=scope)
        with tf.variable_scope("output") as scope:
            shift_mean = layers.fully_connected(hidden, 2, activation_fn=None, scope=scope)
    shift = tf.nn.tanh(shift_mean)
    x, y = shift[:, 0], shift[:, 1]


with tf.variable_scope("st_forward"):
    theta = tf.stack([
        tf.concat([tf.stack([s, tf.zeros_like(s)], axis=1), tf.expand_dims(x, 1)], axis=1),
        tf.concat([tf.stack([tf.zeros_like(s), s], axis=1), tf.expand_dims(y, 1)], axis=1),
    ], axis=1)

    window = transformer(
        tf.expand_dims(tf.reshape(input_layer, [-1, pic_size, pic_size]), 3),
        theta, [win_size, win_size]
        )[:, :, :, 0]
    window = tf.clip_by_value(window,0.0,1.0)

with tf.variable_scope("st_bypart"):
    theta_recon = tf.stack([
        tf.concat([tf.stack([1.0 / s, tf.zeros_like(s)], axis=1), tf.expand_dims(-x / s, 1)], axis=1),
        tf.concat([tf.stack([tf.zeros_like(s), 1.0 / s], axis=1), tf.expand_dims(-y / s, 1)], axis=1),
    ], axis=1)

    window_recon = transformer(
        tf.expand_dims(window, 3),
        theta_recon, [pic_size, pic_size]
        )[:, :, :, 0]
    window_recon = tf.reshape(tf.clip_by_value(window_recon,0.0,1.0),[-1,pic_size*pic_size])

# output_true shall have the original image for error calculations
output_true = tf.placeholder('float32', [None, pic_size*pic_size], name = "Truth")


with tf.variable_scope("recon_loss"):
    # define our cost function
    #meansq =    tf.reduce_mean(tf.square(rec - output_true))
    meansq =    tf.reduce_mean(tf.square(window_recon - output_true))
    meansq *= win_size*win_size
    binarcs = -tf.reduce_mean(
        output_true * tf.log(window_recon+ 10e-10) +
        (1.0 - output_true) * tf.log(1.0 - window_recon+ 10e-10))
    vae_loss = meansq-tf.reduce_mean(tf.square(window))*10
    #vae_loss = binarcs*0.0001-tf.reduce_mean(tf.square(window))


learn_rate = 0.001   # how fast the model should learn
#optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(vae_loss)
optimizer = tf.train.AdamOptimizer(learn_rate).minimize(vae_loss)
#optimizer = tf.train.RMSPropOptimizer(learn_rate, momentum=.9).minimize(vae_loss)


# initialising stuff and starting the session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#tf.summary.FileWriter (n)
writer = tf.summary.FileWriter("demo")
writer.add_graph(sess.graph)
summary = tf.summary.merge([
                tf.summary.scalar("loss", vae_loss),
                tf.summary.scalar("mean_sq", meansq),
                tf.summary.scalar("binary_cross", binarcs),
                tf.summary.image("recon", tf.reshape(window_recon, [-1, pic_size, pic_size, 1]) ),
                tf.summary.image("window", tf.reshape(window, [-1, win_size, win_size, 1])),
                ])

# defining batch size, number of epochs and learning rate
batch_size = 500  # how many images to use together for training
hm_epochs =100  # how many times to go through the entire dataset
tot_images = 60000 # total number of images
# running the model for a 1000 epochs taking 100 images in batches
# total improvement is printed out after each epoch
kl = 0
for epoch in range(hm_epochs):
    epoch_loss = 0    # initializing error as 0
    for i in range(int(tot_images/batch_size)):
        epoch_x = all_images[ i*batch_size : (i+1)*batch_size ]
        _,c = sess.run([optimizer,vae_loss],feed_dict={input_layer:epoch_x, output_true:epoch_x})
        epoch_loss += c
    epoch_x = all_images[10:300]
    summ = sess.run(summary, feed_dict={input_layer: epoch_x, \
       output_true: epoch_x})
    writer.add_summary(summ,epoch)
    if epoch%20==0:
        print('Epoch', epoch, '/', hm_epochs, 'loss:',epoch_loss)



def plot_results(model_name="vae_mnist"):
    import os
    import matplotlib.pyplot as plt
    img = np.zeros((50,50))
    for i in range(50):
        img[:,i] = (i%5)/5

    x_true = all_images[10:30]
    x_true[14]=img.flatten()
    filename = os.path.join(model_name, "compair.png")
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        j = i
        any_image = x_true[j]
        x_decoded,error,sh,sc = sess.run([window_recon,meansq,shift,scale],\
                       feed_dict={input_layer:[any_image], output_true:[any_image]})
        x_dec = x_decoded[0].reshape(pic_size, pic_size)

        #print(x_decoded)
        print(sh[0])
        sc = sc[0]
        sh = sh[0]
        sx, sy = sc*pic_size/2.0  , sc*pic_size/2.0
        cx, cy = (1+sh[0])*pic_size/2.0, (1+sh[1])*pic_size/2.0
        lx = cx - sx
        ly = cy - sy
        rx = cx + sx
        ry = cy + sy

        x_tt = x_true[j].reshape(pic_size, pic_size)
        #print(int(lx),int(ly),int(rx),int(ry),int(sx),int(sy))

        for k in range(int(2*sx)):
            x_tt[max(0,min(49,int(ly))),max(0,min(49,int(lx+k)))] = 0.5
            x_tt[max(0,min(49,int(ry))),max(0,min(49,int(lx+k)))] = 0.5
            x_dec[max(0,min(49,int(ly))),max(0,min(49,int(lx+k)))] = 0.5
            x_dec[max(0,min(49,int(ry))),max(0,min(49,int(lx+k)))] = 0.5

        for k in range(int(2*sy)):
            x_tt[max(0,min(49,int(ly+k))),max(0,min(49,int(lx)))] = 0.5
            x_tt[max(0,min(49,int(ly+k))),max(0,min(49,int(rx)))] = 0.5
            x_dec[max(0,min(49,int(ly+k))),max(0,min(49,int(lx)))] = 0.5
            x_dec[max(0,min(49,int(ly+k))),max(0,min(49,int(rx)))] = 0.5


        ax = plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plt.imshow(x_tt ,  cmap='Greys')
        plt.xlabel(error)
        plt.xticks([])
        plt.yticks([])
        ax = plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plt.imshow(x_dec,  cmap='Greys')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(filename)
    plt.show()
    #plt.figure(figsize=(2,1))
    #ax = plt.subplot(1, 2, 1)
    #plt.imshow(x_tt ,  cmap='Greys')
    #plt.xlabel("eccror:"+str(np.sum(np.square(x_dec-x_tt))))
    #ax = plt.subplot(1, 2, 2)
    #plt.imshow(x_dec,  cmap='Greys')
    #plt.xlabel("norm"+str(np.sum(np.square(x_tt))))
    #plt.show()

plot_results(model_name="vae_test")
