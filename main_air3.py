"""
AIR model using slim
    Fully with r-step (Repeat step)
"""
# Importing tensorflow
import tensorflow as tf
from tensorflow import keras
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as layers
from tensorflow.keras.datasets import mnist, fashion_mnist
# Importing some more libraries
import numpy as np
import matplotlib.pyplot as plt
from unity import  colored_hook,encoder, decoder
from transformer import spatial_transformer_network
import tensorflow.contrib.slim as slim
import os,sys
#loading the images
#all_images = np.loadtxt('fashion-mnist_train.csv',\
#                  delimiter=',', skiprows=1)[:,1:]
def sample_from_mvn(mean, diag_variance):
    # sampling from the multivariate normal
    # with given mean and diagonal covaraince
    standard_normal = tf.random_normal(tf.shape(mean))
    return mean + standard_normal * tf.sqrt(diag_variance)

def gaussian_log_likelihood(x, mean, var, eps=1e-8):
    # compute log P(x) for diagonal Guassian
    # -1/2 log( (2pi)^k sig_1 * sig_2 * ... * sig_k ) -  sum_i 1/2sig_i^2 (x_i - m_i)^2
    bb = tf.square(x-mean)
    bb /=(var + eps)
    return -0.5 * tf.reduce_sum( tf.log(2.*np.pi*var + eps)
                               + bb, axis=1)
def concrete_binary_pre_sigmoid_sample(log_odds, temperature, eps=10e-10):
    count = tf.shape(log_odds)[0]

    u = tf.random_uniform([count], minval=0, maxval=1)
    noise = tf.log(u + eps) - tf.log(1.0 - u + eps)
    y = (log_odds + noise) / temperature

    return y

def concrete_binary_kl_mc_sample(y,
                                 prior_log_odds, prior_temperature,
                                 posterior_log_odds, posterior_temperature,
                                 eps=10e-10):

    y_times_prior_temp = y * prior_temperature
    log_prior = tf.log(prior_temperature + eps) - y_times_prior_temp + prior_log_odds - \
        2.0 * tf.log(1.0 + tf.exp(-y_times_prior_temp + prior_log_odds) + eps)

    y_times_posterior_temp = y * posterior_temperature
    log_posterior = tf.log(posterior_temperature + eps) - y_times_posterior_temp + posterior_log_odds - \
        2.0 * tf.log(1.0 + tf.exp(-y_times_posterior_temp + posterior_log_odds) + eps)

    return log_posterior - log_prior

data_type = 1
if data_type == 1:
    x_train = np.load('/home/suttisak/research/air-present/mnist_sing.npy')/255
    y_train = np.load('/home/suttisak/research/air-present/nmist_label.npy')
    (train_images, train_labels), (test_images, test_labels)  = mnist.load_data()
elif data_type == 2:
    x_train = np.load('/tmp/multi-mnist/fashion_sing.npy')/255
    y_train = np.load('/tmp/multi-mnist/fashion_label.npy')
    (train_images, train_labels), (test_images, test_labels)  = fashion_mnist.load_data()


all_images = np.zeros((60000,50*50))
for i in range(60000):
    all_images[i]=x_train[i].flatten()

all_images = all_images.astype('float32')
#looking at the shape of the file
#print(all_images.shape)


# Deciding how many nodes wach layer should have
rnn_hidden_units = (4,8,16,32)
rec_hidden_units = (32, 16) #(512, 256)
vae_generative_units= (16,32)#256, 512)
vae_likelihood_std=0.3
hidden_units = 64
latent_dim = 10
pic_size = 50
win_size = 28
T = 2

z_pres_temperature = 1.0

batch_size = 256   # how many images to use together for training
hm_epochs = 101     # how many times to go through the entire dataset
tot_images = 60000 # total number of images



def main(arv):
    DO_SHARE = None
    lstm_rnn = tf.contrib.rnn.LSTMCell(256, state_is_tuple=True)

    input_layer = tf.placeholder('float32', shape=[None, pic_size*pic_size], name = "input")
    is_train = tf.placeholder(tf.bool, name='is_train')
    #rnn_out = layers.fully_connected(input_layer, pic_size*pic_size,activation_fn=tf.nn.relu,scope="hidden_rnn")
    def hidden_rnn(input,x_hat,state):
        with tf.variable_scope("hidden_rnn",reuse=DO_SHARE):
            net = tf.concat([input, x_hat],axis=3)
            with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                normalizer_fn=slim.batch_norm,
                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                normalizer_params={"is_training": is_train}):

                for i in range(len(rnn_hidden_units)):
                    net = slim.conv2d(net, rnn_hidden_units[i], [3, 3], stride=2, scope="conv%d" % (i*2+2))
                net = slim.flatten(net)
            return lstm_rnn(net, state)

    def comp_scale(rnn_out,scale_kl):
        with tf.variable_scope("scale",reuse=DO_SHARE):
            # sampling scale
            with tf.variable_scope("mean"):
                with tf.variable_scope("hidden") as scope:
                    hidden = layers.fully_connected(rnn_out, hidden_units, scope=scope)
                with tf.variable_scope("output") as scope:
                    scale_mean = layers.fully_connected(hidden, 1, activation_fn=None, scope=scope)
            with tf.variable_scope("log_variance"):
                with tf.variable_scope("hidden") as scope:
                    hidden = layers.fully_connected(rnn_out, hidden_units, scope=scope)
                with tf.variable_scope("output") as scope:
                    scale_log_variance = layers.fully_connected(hidden, 1, activation_fn=None, scope=scope)
            scale_variance = tf.exp(scale_log_variance)
            scale = tf.nn.sigmoid(sample_from_mvn(scale_mean, scale_variance))
            #scale = tf.nn.sigmoid(scale_mean)  #with out randomness
            s = scale[:, 0]
            scale_kl += tf.reduce_mean(0.5 * tf.reduce_sum(tf.log(0.01)- scale_log_variance - 1.0 + scale_variance/0.01
                        + tf.square(scale_mean - 0.0)/0.01 , 1))
        return s, scale_kl

    def comp_shift(rnn_out,shift_kl):
        with tf.variable_scope("shift",reuse=DO_SHARE):
            # sampling shift
            with tf.variable_scope("mean"):
                with tf.variable_scope("hidden") as scope:
                    hidden = layers.fully_connected(rnn_out, hidden_units*2, scope=scope)
                with tf.variable_scope("output") as scope:
                    shift_mean = layers.fully_connected(hidden, 2, activation_fn=None, scope=scope)
            with tf.variable_scope("log_variance"):
                with tf.variable_scope("hidden") as scope:
                    hidden = layers.fully_connected(rnn_out, hidden_units, scope=scope)
                with tf.variable_scope("output") as scope:
                    shift_log_variance = layers.fully_connected(hidden, 2, activation_fn=None, scope=scope)
            shift_variance = tf.exp(shift_log_variance)
            shift = tf.nn.tanh(sample_from_mvn(shift_mean, shift_variance))
            #shift = tf.nn.tanh(shift_mean) #with out randomness
            x, y = shift[:, 0], shift[:, 1]
            shift_kl += tf.reduce_mean(0.5 * tf.reduce_sum( tf.log(2.1) - shift_log_variance - 1.0 + shift_variance/2.1 +
                tf.square(shift_mean - 0.0)/2.1 , 1))

        return x,y , shift_kl

    def comp_para(rnn_out):
        with tf.variable_scope("para",reuse=DO_SHARE):
            with tf.variable_scope("mean"):
                with tf.variable_scope("hidden") as scope:
                    hidden = layers.fully_connected(rnn_out, hidden_units*2, scope=scope)
                with tf.variable_scope("output") as scope:
                    sigma_mean = layers.fully_connected(hidden, 2, activation_fn=None, scope=scope)
            sigma2 = tf.nn.sigmoid(sigma_mean)*0.5
            sigma2,gamma = sigma2[:,0],sigma2[:,1]
        return sigma2,gamma

    def comp_pres(rnn_out,z_pres_kl):
        z_pres_prior_log_odds = -2.0
        with tf.variable_scope("z_pres", reuse=DO_SHARE):
            with tf.variable_scope("hidden") as scope:
                hidden = layers.fully_connected(rnn_out, hidden_units, scope=scope)
            with tf.variable_scope("output") as scope:
                z_pres_log_odds = layers.fully_connected(hidden, 1, activation_fn=None, scope=scope)[:, 0]
        with tf.variable_scope("gumbel"):
            z_pres_pre_sigmoid = concrete_binary_pre_sigmoid_sample(
                z_pres_log_odds, z_pres_temperature )
            z_pres = tf.nn.sigmoid(z_pres_pre_sigmoid)
            if not is_train:
                z_pres = tf.round(z_pres)
            z_pres = tf.reshape(z_pres, [-1, 1, 1, 1])
            z_pres_kl += concrete_binary_kl_mc_sample(
                z_pres_pre_sigmoid,
                z_pres_prior_log_odds, 1.0,
                z_pres_log_odds, 1.0 )
            #print("z_pres ",z_pres_kl.shape)
            z_pres_kl = tf.reduce_mean(z_pres_kl)
        return z_pres, z_pres_kl


    def read(input,theta):
        with tf.variable_scope("st_forward"):
            window = spatial_transformer_network(input, theta, (win_size, win_size))
            window = tf.clip_by_value(window,0.0,1.0)
        return window

    def vae(window,window_hat, vae_kl):
        with tf.variable_scope("vae",reuse=DO_SHARE):
            window_v = tf.concat([window,window_hat],axis=3)
            #window_v = tf.reshape(window,[-1,win_size,win_size,1])
            rec_mean, rec_log_var = encoder(window_v,is_train,rec_hidden_units,latent_dim)

            with tf.variable_scope("rec_sample"):
                standard_normal_sample = tf.random_normal([tf.shape(input_layer)[0], latent_dim])
                rec_sample = rec_mean + 1*standard_normal_sample * tf.sqrt(tf.exp(rec_log_var))

            rec                   = decoder(rec_sample, is_train, [0,7,7,16],vae_generative_units, latent_dim)
            rec = tf.reshape(rec,[-1,win_size,win_size,1])
            vae_kl += tf.reduce_mean(0.5 * tf.reduce_sum( tf.log(1.1) - rec_log_var - 1.0 + tf.exp(rec_log_var)/1.1 +
                tf.square(rec_mean - 0.0)/1.1 , 1))
        return rec, vae_kl

    def write(rec, itheta,sigma2):
        with tf.variable_scope("st_backward"):
            window_recon = spatial_transformer_network(rec, itheta, (pic_size, pic_size),sigma2)
            #window_recon = tf.Print(window_recon, ["win_rec", tf.shape(window_recon)])
            #window_recon = tf.reshape(tf.clip_by_value(window_recon,0.0,1.0),[-1,pic_size*pic_size])
        return window_recon

    """
    with tf.variable_scope("st_bypart"):
        window_recon2 = spatial_transformer_network(window, itheta, (pic_size, pic_size),sigma2)
        window_recon2 = tf.reshape(tf.clip_by_value(window_recon2,0.0,1.0),[-1,pic_size*pic_size])
    """

    input = tf.reshape(input_layer, shape=[-1, pic_size,pic_size, 1])

    scale_kl, shift_kl, z_pres_kl, vae_kl = 0.0, 0.0, 0.0, 0.0
    cs = [0]*T
    s,x,y = [0]*T, [0]*T, [0]*T
    z_pres = 1.0
    rnn_state = lstm_rnn.zero_state(tf.shape(input_layer)[0],tf.float32)

    for t in range(T):
        c_prev = tf.zeros((tf.shape(input_layer)[0],pic_size,pic_size,1))-5.5 if t==0 else cs[t-1]
        x_hat = input - tf.sigmoid(c_prev)
        rnn_out,rnn_state = hidden_rnn(input,x_hat,rnn_state)
        s[t], shift_kl = comp_scale(rnn_out,shift_kl)
        x[t], y[t], scale_kl = comp_shift(rnn_out,scale_kl)
        sigma2,gamma = comp_para(rnn_out)
        z_pres, z_pres_kl = comp_pres(rnn_out, z_pres_kl)
        #z_pres *= z_pres_t #z_pres = z_pres_t
        zeros = tf.zeros_like(s[t])
        theta = tf.stack([s[t], zeros, x[t], zeros, s[t], y[t]], 1)
        itheta = tf.stack([1.0/s[t], zeros, -x[t]/s[t], zeros, 1.0/s[t], -y[t]/s[t]], 1)
        window = read(input,theta)
        window_hat = read(x_hat,theta)

        rec, vae_kl= vae(window,window_hat,vae_kl)
        cs[t] = c_prev + z_pres*write(rec, itheta, sigma2)#tf.clip_by_value(c_prev + write(rec, itheta, sigma2),0.0,1.0)
        DO_SHARE=True

    window_recon = tf.reshape(tf.sigmoid(cs[T-1]),[-1,pic_size*pic_size])

    # output_true shall have the original image for error calculations
    output_true = tf.placeholder('float32', [None, pic_size*pic_size], name = "Truth")
    position = tf.placeholder('float32', shape=[None, 3], name = "Position")

    with tf.variable_scope("recon_loss1"):
        # define our cost function
        #meansq =    tf.reduce_mean(tf.square(rec - output_true))
        meansq =    tf.reduce_mean(tf.square(window_recon - output_true))
        meansq *= win_size*win_size#*0.01
        meansq2 =    tf.reduce_mean(tf.square(rec - window))
        binarcs = -tf.reduce_mean(
            output_true * tf.log(window_recon+ 10e-10) +
            (1.0 - output_true) * tf.log(1.0 - window_recon+ 10e-10))
        binarcs *=win_size
        if data_type == 1 or data_type == 2:
            vae_loss = binarcs + meansq +scale_kl +shift_kl+vae_kl*0.1 + z_pres_kl
        elif data_type == 2:
            vae_loss = meansq +scale_kl +shift_kl+vae_kl*0.01


    with tf.variable_scope("trick_term"):
        # define our cost function
        trick = -tf.reduce_mean(tf.square(window))#*60*5#*.01
        trick2 = tf.reduce_mean(tf.square(s[0] - position[:,0]))*0.75
        trick2 += 0.5*tf.reduce_mean(tf.square(x[0] - position[:,1])+tf.square(y[0] - position[:,2]))
        trick_loss = meansq*0.01+trick2 #+scale_kl +shift_kl

    t_vars = tf.trainable_variables()
    vae_vars = [var for var in t_vars if 'vae' in var.name]
    # define our optimizer
    learn_rate = 0.0002   # how fast the model should learn

    slimopt = slim.learning.create_train_op(vae_loss, tf.train.AdamOptimizer(0.0005))
    slimopt2= slim.learning.create_train_op(trick_loss, tf.train.AdamOptimizer(0.001))

    # initialising stuff and starting the session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    saver = tf.train.Saver()


    #tf.summary.FileWriter (n)
    writer = tf.summary.FileWriter("TensorB")
    writer.add_graph(sess.graph)
    #tf.summary.histogram("loss", vae_loss)
    #merged_summary = tf.summary.merge_all()
    summary = tf.summary.merge([
                    tf.summary.scalar("loss", vae_loss),
                    tf.summary.scalar("l2_big", meansq),
                    tf.summary.scalar("l2_small", meansq2),
                    tf.summary.scalar("binary_cross", binarcs),
                    tf.summary.scalar("scale_kl", scale_kl),
                    tf.summary.scalar("shift_kl", shift_kl),
                    tf.summary.scalar("vae_kl", vae_kl),
                    tf.summary.scalar("Trick", trick),
                    tf.summary.scalar("Position_loss", trick2),
                    tf.summary.image("rec_window", tf.reshape(rec, [-1, win_size, win_size, 1])),
                    #tf.summary.image("sum_map", tf.reshape(sss, [-1, pic_size, pic_size, 1])),
                    tf.summary.image("cs0", tf.reshape(tf.sigmoid(cs[0]), [-1, pic_size, pic_size, 1]) ),
                    tf.summary.image("recon", tf.reshape(window_recon, [-1, pic_size, pic_size, 1]) ),
                    tf.summary.image("window", tf.reshape(window, [-1, win_size, win_size, 1])),
                    #tf.summary.image("bypart", tf.reshape(window_recon2, [-1, pic_size, pic_size, 1]) ),
                    ])


    def plot_results(model_name="vae_mnist",index = 0):
        import matplotlib.pyplot as plt
        if not os.path.exists(model_name):
            os.makedirs(model_name)
        """
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
        """

        x_true = all_images[10:30].copy()
        filename = os.path.join(model_name, "compair%02d.png"%(index))
        num_rows = 5
        num_cols = 3
        num_images = num_rows*num_cols
        plt.figure(figsize=(2*2*num_cols, 2*num_rows))
        #print(x_encoded[0].shape[0])

        for i in range(num_images):
            j = i
            any_image = x_true[j]
            x_decoded,error,sh1,sh2,sc = sess.run([window_recon,meansq
                            ,x[0],y[0],s[0]],\
                           feed_dict={input_layer:[any_image], output_true:[any_image], is_train: False})
            x_dec = x_decoded[0].reshape(pic_size, pic_size)

            #print(x_decoded)

            sc[0] = sc[0]
            sh1[0] = sh1[0]
            sh2[0] = sh2[0]
            sx, sy = sc[0]*pic_size/2.0  , sc[0]*pic_size/2.0
            cx, cy = (1+sh1[0])*pic_size/2.0, (1+sh2[0])*pic_size/2.0
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
        plt.close('all')
        #plt.show()

    # defining batch size, number of epochs and learning rate


    kl = 0
    for epoch in range(hm_epochs):
        epoch_loss = 0    # initializing error as 0
        for i in range(int(tot_images/batch_size)):
            epoch_x = all_images[ i*batch_size : (i+1)*batch_size ]
            posi    = y_train[ i*batch_size : (i+1)*batch_size ]
            #_,c = sess.run([optimizer2,vae_loss],feed_dict={input_layer:epoch_x, output_true:epoch_x})
            if 0==1 and (epoch<1 or (epoch%5<=1 and epoch<11) or (epoch%5==0 and epoch<31)):
                _,c = sess.run([slimopt2,vae_loss],feed_dict={input_layer:epoch_x,
                      output_true:epoch_x , is_train:True, position:posi})
            else:
                _,a,c = sess.run([slimopt,z_pres,vae_loss],feed_dict={input_layer:epoch_x,
                      output_true:epoch_x, is_train:True})
            epoch_loss += c

        epoch_x = all_images[10:300]
        posi    = y_train[10:300]
        summ = sess.run(summary, feed_dict={input_layer: epoch_x, \
           output_true: epoch_x, is_train: False, position:posi})
        writer.add_summary(summ,epoch)
        if epoch%10==0:
            print('Epoch', epoch, '/', hm_epochs, 'loss:',epoch_loss)
            plot_results(model_name="air_test/",index =epoch)
        if epoch%100 == 0:
            if not os.path.exists('./model'):
                os.makedirs('./model')
            saver.save(sess, './model/' + str(i))

    plot_results(model_name="air_test",index=100)
#plt.show()
if __name__ == "__main__":
    sys.excepthook = colored_hook(
        os.path.dirname(os.path.realpath(__file__)))
    tf.app.run()
