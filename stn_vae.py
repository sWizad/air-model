"""
STN helping VAE
vae using slim
"""
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist
import tensorflow.contrib.slim as slim
import numpy as np
# Importing some more libraries
from transformer import transformer
from vae import  encoder, decoder

data_type = 1
if data_type == 1:
    (img_train, img_test), (img_test, y_test) = mnist.load_data()
elif data_type ==2:
    (img_train, img_test), (img_test, y_test) = fashion_mnist.load_data()

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
rnn_hidden_units = (8,16,32)
rec_hidden_units = (32,16)#(512, 256)
vae_generative_units = (16,32)#(256, 512)
hidden_units = 32
vae_likelihood_std=0.3
latent_dim = 2
win_size = 28

def plot_results(model_name="vae_mnist",index = 0):
    import os
    import matplotlib.pyplot as plt
    if not os.path.exists(model_name):
        os.makedirs(model_name)

    filename = os.path.join(model_name, "vae_mean.png")
    x_true = all_images[10:30]
    # display a 2D plot of the digit classes in the latent space
    z_mean = sess.run(rec_mean, feed_dict={input_layer:x_test[1:5000], is_train:False})
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
    grid_x = np.linspace(-1.5, 1.5, n)
    grid_y = np.linspace(-1.5, 1.5, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.zeros([1,latent_dim])
            z_sample[0,0] = xi
            z_sample[0,1] = yi
            #z_sample = np.array([[xi, yi]])
            x_decoded = sess.run(rec, feed_dict={rec_sample:z_sample, is_train:False})
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


    filename = os.path.join(model_name, "compair%02d.png"%(index))
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        j = i
        any_image = x_true[j]
        x_decoded,error,x_encoded,sh,sw,px,py,rr= sess.run([rec,meansq,rec_mean,h,w,x,y,r],\
                       feed_dict={input_layer:[any_image], output_true:[any_image], is_train:False})
        x_tt = x_true[j].reshape(win_size, win_size)
        x_dec = x_decoded[0].reshape(win_size, win_size)
        #print(x_encoded.shape)
        sar = [str(int(a*10)/10) for a in x_encoded[0]]
        ax = plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plt.imshow(x_tt ,  cmap='Greys')
        #plt.xlabel(error)
        #plt.xlabel('z = ['+", ".join(sar)+']')
        plt.xlabel("(%.2f,%2f), (%.2f, %.2f), %.2f" % (sh,sw,px,py ,rr))
        plt.xticks([])
        plt.yticks([])
        ax = plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plt.imshow(x_dec,  cmap='Greys')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(filename)
    #plt.show()

    plt.figure()
    ax = plt.subplot(1, 2, 1)
    plt.imshow(x_tt ,  cmap='Greys')
    plt.xticks([])
    plt.yticks([])
    ax = plt.subplot(1, 2, 2)
    plt.imshow(x_dec,  cmap='Greys')
    plt.xticks([])
    plt.yticks([])
    plt.close('all')
    #plt.show()
    #np.save('num4.npy', xx)

# VAE model
input_layer = tf.placeholder('float32', shape=[None, win_size*win_size], name = "input")
is_train = tf.placeholder(tf.bool, name='is_train')

with tf.variable_scope("pre-transofrm"):
    net = tf.reshape(input_layer, shape=[-1, win_size,win_size, 1])
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        normalizer_fn=slim.batch_norm,
        activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
        normalizer_params={"is_training": is_train}):

        for i in range(len(rnn_hidden_units)):
            net = slim.conv2d(net, rnn_hidden_units[i], [3, 3], scope="conv%d" % (i*2+1))
            net = slim.conv2d(net, rnn_hidden_units[i], [3, 3], stride=2, scope="conv%d" % (i*2+2))

        net = slim.flatten(net)
        rnn_out = slim.fully_connected(net, 64)

    with tf.variable_scope("scale"):
        # sampling scale
        with tf.variable_scope("mean"):
            with tf.variable_scope("hidden") as scope:
                hidden = slim.fully_connected(rnn_out, hidden_units*2, scope=scope)
            with tf.variable_scope("output") as scope:
                scale_mean = slim.fully_connected(hidden, 2, activation_fn=None, scope=scope)
        scale = tf.nn.sigmoid(scale_mean)*0.25+0.75
        h, w = scale[:, 0], scale[:, 1]

    with tf.variable_scope("shift"):
        # sampling shift
        with tf.variable_scope("mean"):
            with tf.variable_scope("hidden") as scope:
                hidden = slim.fully_connected(rnn_out, hidden_units*2, scope=scope)
            with tf.variable_scope("output") as scope:
                shift_mean = slim.fully_connected(hidden, 2, activation_fn=None, scope=scope)
        shift = tf.nn.tanh(shift_mean)*0.25
        x, y = shift[:, 0], shift[:, 1]

    with tf.variable_scope("roll"):
        # sampling shift
        with tf.variable_scope("mean"):
            with tf.variable_scope("hidden") as scope:
                hidden = slim.fully_connected(rnn_out, hidden_units, scope=scope)
            with tf.variable_scope("output") as scope:
                roll_mean = slim.fully_connected(hidden, 1, activation_fn=None, scope=scope)
        roll = tf.nn.tanh(roll_mean)
        r = roll[:, 0]

with tf.variable_scope("transformer"):
    theta = tf.stack([
        tf.concat([tf.stack([h*tf.math.cos(r), tf.math.sin(r)], axis=1), tf.expand_dims(x, 1)], axis=1),
        tf.concat([tf.stack([-tf.math.sin(r), w*tf.math.cos(r)], axis=1), tf.expand_dims(y, 1)], axis=1),
    ], axis=1)

    window = transformer(
        tf.expand_dims(tf.reshape(input_layer, [-1, win_size, win_size]), 3),
        theta, [win_size, win_size]
        )[:, :, :, 0]
    window = tf.clip_by_value(window,0.0,1.0)


with tf.variable_scope("vae"):
    with tf.variable_scope("rsh"):
        net = tf.reshape(window,[-1,win_size,win_size])
    net = tf.reshape(net,[-1,win_size,win_size,1])
    rec_mean, rec_log_var = encoder(net,is_train,rec_hidden_units,latent_dim)
    with tf.variable_scope("rec_sample"):
        standard_normal_sample = tf.random_normal([tf.shape(input_layer)[0], latent_dim])
        rec_sample = rec_mean + 1*standard_normal_sample * tf.sqrt(tf.exp(rec_log_var))

    rec  = decoder(rec_sample,is_train, [0,7,7,16],vae_generative_units, latent_dim)
    rec = tf.reshape(rec,[-1,win_size*win_size])
# output_true shall have the original image for error calculations
output_true = tf.placeholder('float32', [None, win_size*win_size], name = "Truth")

def gaussian_log_likelihood(x, mean, var, eps=1e-8):
    # compute log P(x) for diagonal Guassian
    # -1/2 log( (2pi)^k sig_1 * sig_2 * ... * sig_k ) -  sum_i 1/2sig_i^2 (x_i - m_i)^2
    bb = tf.square(x-mean)
    bb /=(var + eps)
    return -0.5 * tf.reduce_sum( tf.log(2.*np.pi*var + eps)
                               + bb, axis=1)

with tf.variable_scope("loss_function"):
    # define our cost function
    with tf.variable_scope("recon_loss"):
        meansq =    tf.reduce_mean(tf.square(rec - tf.reshape(window,[-1,win_size*win_size])))
        meansq *= win_size*win_size
        binarcs = -tf.reduce_mean(
            output_true * tf.log(rec+ 10e-10) +
        (1.0 - output_true) * tf.log(1.0 - rec+ 10e-10))

    with tf.variable_scope("kl_loss"):
        vae_kl = 0.5 * tf.reduce_sum( 0.0 - rec_log_var - 1.0 + tf.exp(rec_log_var)  +
            tf.square(rec_mean - 0.0) , 1)
        vae_kl = tf.reduce_mean(vae_kl)
        vae_kl2 = tf.reduce_mean(
                    - gaussian_log_likelihood(rec_sample, 0.0, 1.0, eps=0.0) \
                    + gaussian_log_likelihood(rec_sample, rec_mean,  (tf.exp(rec_log_var)) )
                    )
    vae_loss = meansq + vae_kl
    #vae_loss = binarcs*0.0001-tf.reduce_mean(tf.square(window))



learn_rate = 0.001   # how fast the model should learn
slimopt = slim.learning.create_train_op(vae_loss, tf.train.AdamOptimizer(learn_rate))

# initialising stuff and starting the session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()


writer = tf.summary.FileWriter("demo/2")
writer.add_graph(sess.graph)
summary = tf.summary.merge([
                tf.summary.scalar("loss_total", vae_loss),
                tf.summary.scalar("mean_sq", meansq),
                tf.summary.scalar("binary_cross", binarcs),
                tf.summary.image("recon", tf.reshape(rec, [-1, win_size, win_size, 1]) ),
                tf.summary.image("crop", tf.reshape(window, [-1, win_size, win_size, 1]) ),
                tf.summary.image("original", tf.reshape(input_layer, [-1, win_size, win_size, 1]) ),
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
        _,c = sess.run([slimopt,vae_loss],feed_dict={input_layer:epoch_x, output_true:epoch_x, is_train:True})
        epoch_loss += c
    epoch_x = all_images[10:30]
    summ = sess.run(summary, feed_dict={input_layer: epoch_x, \
       output_true: epoch_x, is_train:False})
    writer.add_summary(summ,epoch)
    if epoch%10==0:
        print('Epoch', epoch, '/', hm_epochs, 'loss:',epoch_loss)
        plot_results(model_name="vae_test/",index =epoch)
print('Epoch', epoch+1, '/', hm_epochs, 'loss:',epoch_loss)


plot_results(model_name="vae_test")
