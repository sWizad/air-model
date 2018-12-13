import tensorflow as tf
from tensorflow import keras
import tensorflow.contrib.layers as layers
from tensorflow.keras.datasets import mnist, fashion_mnist
# Importing some more libraries
import numpy as np
import matplotlib.pyplot as plt
from transformer import transformer
from vae import vae, encoder, decoder
import cv2

def gaussian_log_likelihood(x, mean, var, eps=1e-8):
    # compute log P(x) for diagonal Guassian
    # -1/2 log( (2pi)^k sig_1 * sig_2 * ... * sig_k ) -  sum_i 1/2sig_i^2 (x_i - m_i)^2
    bb = tf.square(x-mean)
    bb /=(var + eps)
    return -0.5 * tf.reduce_sum( tf.log(2.*np.pi*var + eps)
                               + bb, axis=1)

(img_train, img_test), (img_test, y_test) = mnist.load_data()
#x_train = np.load('mnis_single.npy')
#x_train = x_train[0].astype('float32')
img_train = img_train.astype('float32') /255

all_images = np.zeros((60000,28*28))
x_test = np.zeros((10000,28*28))
for i in range(60000):
    all_images[i]=img_train[i].flatten()
for i in range(10000):
    x_test[i]=img_test[i].flatten()
all_images = all_images.astype('float32')
x_test = x_test.astype('float32')

# Deciding how many nodes wach layer should have
#hidden_units = 4*16
rec_hidden_units = (512, 256)
vae_generative_units=(256, 512)
vae_likelihood_std=0.3
latent_dim = 2
pic_size = 50
win_size = 28

input_layer = tf.placeholder('float32', shape=[None, win_size*win_size], name = "input")

with tf.variable_scope("vae"):
    #rec, rec_mean, rec_log_var, rec_sample = vae(input_layer,win_size**2,rec_hidden_units,
    #                       latent_dim,vae_generative_units,vae_likelihood_std)

    rec_mean, rec_log_var = encoder(input_layer,win_size**2,rec_hidden_units,latent_dim)
    with tf.variable_scope("rec_sample"):
        standard_normal_sample = tf.random_normal([tf.shape(input_layer)[0], latent_dim])
        rec_sample = rec_mean + 1*standard_normal_sample * tf.sqrt(tf.exp(rec_log_var))

    rec                   = decoder(rec_sample,win_size**2,vae_generative_units, latent_dim)

    #rec = tf.clip_by_value(rec,0.0,1.0)
# output_true shall have the original image for error calculations
output_true = tf.placeholder('float32', [None, win_size*win_size], name = "Truth")


with tf.variable_scope("recon_loss"):
    # define our cost function
    meansq =    tf.reduce_mean(tf.square(rec - output_true))
    meansq *= win_size*win_size
    binarcs = -tf.reduce_mean(
        output_true * tf.log(rec+ 10e-10) +
        (1.0 - output_true) * tf.log(1.0 - rec+ 10e-10))
    vae_kl = 0.5 * tf.reduce_sum( 0.0 - rec_log_var - 1.0 + tf.exp(rec_log_var)  +
        tf.square(rec_mean - 0.0) , 1)
    vae_kl = tf.reduce_mean(vae_kl)
    vae_kl2 = tf.reduce_mean(
                - gaussian_log_likelihood(rec_sample, 0.0, 1.0, eps=0.0) \
                + gaussian_log_likelihood(rec_sample, rec_mean,  (tf.exp(rec_log_var)) )
                )
    vae_loss = meansq + vae_kl#*0.0035#-tf.reduce_mean(tf.square(window))*0.5#meansq
    #vae_loss = binarcs*0.0001-tf.reduce_mean(tf.square(window))


learn_rate = 0.0001   # how fast the model should learn
#optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(vae_loss)
optimizer = tf.train.AdamOptimizer(learn_rate).minimize(vae_loss)
#optimizer = tf.train.RMSPropOptimizer(learn_rate, momentum=.9).minimize(vae_loss)


# initialising stuff and starting the session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#tf.summary.FileWriter (n)
writer = tf.summary.FileWriter("demo/2")
writer.add_graph(sess.graph)
summary = tf.summary.merge([
                tf.summary.scalar("loss", vae_loss),
                tf.summary.scalar("mean_sq", meansq),
                tf.summary.scalar("binary_cross", binarcs),
                tf.summary.image("recon", tf.reshape(rec, [-1, win_size, win_size, 1]) ),
                #tf.summary.image("window", tf.reshape(, [-1, win_size, win_size, 1])),
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
    epoch_x = all_images[10:30]
    summ = sess.run(summary, feed_dict={input_layer: epoch_x, \
       output_true: epoch_x})
    writer.add_summary(summ,epoch)
    if epoch%20==0:
        print('Epoch', epoch, '/', hm_epochs, 'loss:',epoch_loss)
print('Epoch', epoch+1, '/', hm_epochs, 'loss:',epoch_loss)


def plot_results(model_name="vae_mnist"):
    import os
    import matplotlib.pyplot as plt
    x_true = all_images[10:30]

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean = sess.run(rec_mean, feed_dict={input_layer:x_test[1:5000]})
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test[1:5000])
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 15
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4*4, 4*4, n)
    grid_y = np.linspace(-4*4, 4*4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = sess.run(rec, feed_dict={rec_sample:z_sample})
            #x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys')
    plt.savefig(filename)


    filename = os.path.join(model_name, "compair.png")
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        j = i
        any_image = x_true[j]
        x_decoded,error,x_encoded= sess.run([rec,meansq,rec_mean],\
                       feed_dict={input_layer:[any_image], output_true:[any_image]})
        x_tt = x_true[j].reshape(win_size, win_size)
        x_dec = x_decoded[0].reshape(win_size, win_size)
        #print(x_encoded.shape)
        sar = [str(int(a*10)/10) for a in x_encoded[0]]
        ax = plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plt.imshow(x_tt ,  cmap='Greys')
        #plt.xlabel(error)
        plt.xlabel('z = ['+", ".join(sar)+']')
        plt.xticks([])
        plt.yticks([])
        ax = plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plt.imshow(x_dec,  cmap='Greys')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(filename)
    #plt.show()
    xx = np.zeros((4,28,28))
    x_tt = x_true[5].reshape(win_size, win_size)
    x_decoded,x_encoded= sess.run([rec,rec_mean],\
                   feed_dict={input_layer:[x_true[5]]})
    x_dec = x_decoded[0].reshape(win_size, win_size)
    xx[0] = x_tt
    xx[1] = x_dec
    x_tt = x_true[6].reshape(win_size, win_size)
    x_decoded,x_encoded= sess.run([rec,rec_mean],\
                   feed_dict={input_layer:[x_true[6]]})
    x_dec = x_decoded[0].reshape(win_size, win_size)
    xx[2] = x_tt
    xx[3] = x_dec

    plt.figure()
    ax = plt.subplot(1, 2, 1)
    plt.imshow(x_tt ,  cmap='Greys')
    plt.xticks([])
    plt.yticks([])
    ax = plt.subplot(1, 2, 2)
    plt.imshow(x_dec,  cmap='Greys')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    np.save('num4.npy', xx)

plot_results(model_name="vae_test")
