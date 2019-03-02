"""
AIR model using slim
    Fully with r-step (Repeat step)
    Colorful images batch
    Use Gaussian-Mixture and Gumbel-softmax to find the box location
"""
# Importing tensorflow
import tensorflow as tf
from tensorflow import keras
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as layers
from tensorflow.keras.datasets import mnist, fashion_mnist
import tensorflow.contrib.slim as slim
import os,sys
import cv2
# Importing some more libraries
import numpy as np
import matplotlib.pyplot as plt
from unity import  colored_hook,encoder, decoder
from transformer import spatial_transformer_network


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
    (train_images, train_labels), (test_images, test_labels)  = mnist.load_data()
elif data_type == 2:
    (train_images, train_labels), (test_images, test_labels)  = fashion_mnist.load_data()


# Deciding how many nodes wach layer should have
rnn_hidden_units = (5, 3, 3, 5, 5, 3, 3, 1, 1)
rec_hidden_units = (32, 16) #(512, 256)
vae_generative_units= (16,32)#256, 512)
vae_likelihood_std=0.3
hidden_units = 64
latent_dim = 40
pic_size = 56
win_size = 28
T = 3

z_pres_temperature = 1.0

batch_size = 256   # how many images to use together for training
hm_epochs = 101     # how many times to go through the entire dataset
tot_images = 60000 # total number of images

seed  = 301

Lena = cv2.imread('img/Lena1.jpg')
Lena = cv2.cvtColor(Lena, cv2.COLOR_BGR2RGB)
#1728 792 3
#60000 28 28

def get_batch(batch_size = 256, begin_point = 0, is_train=True):
    def psu_rand(is_train):
        if is_train:
            return np.random.uniform(0,1)
        else:
            global seed
            mode = 9629
            seed = (seed*8837+5)%mode
            return seed/mode


    batch = np.zeros((batch_size, pic_size, pic_size, 3))
    global seed
    seed  = 301
    for i in range(batch_size):
        x = int(psu_rand(is_train)*711)
        y = int(psu_rand(is_train)*732)
        bg = np.copy(Lena[x:x+pic_size,y:y+pic_size,:])
        bg = cv2.GaussianBlur(bg,(5,5),0)
        for i2 in range(3):
            bg[:,:,i2] = (bg[:,:,i2]+ (psu_rand(is_train)*255.0)) / 2.0
        for i1 in range(T-1):
            x = psu_rand(is_train)*2*.75-.75
            y = psu_rand(is_train)*2*.75-.75
            s = 1. / (1. + np.exp(np.sqrt(-2*np.log(psu_rand(is_train)))*np.cos(np.pi*2*psu_rand(is_train))*0.1))+0.5
            if i1 == 0:
                digit_img = train_images[begin_point+i]
            else:
                digit_img = train_images[int(psu_rand(is_train)*tot_images)]
            M = np.float32([[s,0,(x+0.5)*(pic_size-win_size)],[0,s,(y+0.5)*(pic_size-win_size)]])
            c1 = cv2.warpAffine(digit_img,M,(pic_size,pic_size))
            #c1 = cv2.resize(digit_img,(2*win_size, 2*win_size), interpolation = cv2.INTER_LINEAR)
            pres = 1 if i1==0 else (psu_rand(is_train)<0.5)
            for i2 in range(3):
                bg[:,:,i2] = bg[:,:,i2]*(1-c1/255)+(c1)*psu_rand(is_train)*pres+(1-pres)*bg[:,:,i2]*(c1/255)
        batch[i] =bg[:,:,:]/255
    #plt.imshow(c1)
    #plt.show()
    return batch
'''#test batch
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch = get_batch(100,is_train=True)
num_rol = 3
num_col = 5
num_pl = num_col*num_rol
for i in range(num_pl):
    plt.subplot(num_rol,num_col,i+1)
    plt.imshow(batch[i])
    plt.xticks([])
    plt.yticks([])

plt.show()
exit()
'''



def main(arv):
    DO_SHARE = None
    lstm_rnn = tf.contrib.rnn.LSTMCell(14*14, state_is_tuple=True)

    input_layer = tf.placeholder('float32', shape=[None, pic_size,pic_size,3], name = "input")
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

                for i in range(2):
                    net = slim.conv2d(net, 1, [3, 3], stride=2, scope="conv%d" % (i+1))
                rnn_out1, state = lstm_rnn(slim.flatten(net), state)
                #net = tf.reshape(rnn_out1,[-1,int(pic_size/4),int(pic_size/4),1])
                #for i in range(2):
                #    net  = slim.conv2d_transpose(net , 2, [3, 3], stride=2, scope="convt%d" % (i+1))

                net = tf.concat([input,x_hat],axis=3)
                for i in range(len(rnn_hidden_units)):
                    net = slim.conv2d(net, rnn_hidden_units[i], [3, 3], scope="conva%d" % (i+1))
            return rnn_out1,net, state

    def comp_scale(rnn_out,scale_kl):
        with tf.variable_scope("scale",reuse=DO_SHARE):
            # sampling scale
            #with tf.variable_scope("mean"):
            #    with tf.variable_scope("hidden") as scope:
            #        hidden = layers.fully_connected(rnn_out, hidden_units, scope=scope)
            #    with tf.variable_scope("output") as scope:
            #        scale_mean = layers.fully_connected(hidden, 1, activation_fn=None, scope=scope)
            #with tf.variable_scope("log_variance"):
            #    with tf.variable_scope("hidden") as scope:
            #        hidden = layers.fully_connected(rnn_out, hidden_units, scope=scope)
            #    with tf.variable_scope("output") as scope:
            #        scale_log_variance = layers.fully_connected(hidden, 1, activation_fn=None, scope=scope)
            #scale_variance = tf.exp(scale_log_variance)
            #scale = tf.nn.sigmoid(sample_from_mvn(scale_mean, scale_variance))
            #scale = tf.nn.sigmoid(scale_mean)  #with out randomness
            #s = scale[:, 0]

            if t ==0:
                s = tf.ones([tf.shape(rnn_out)[0],])*1
            #    scale_kl = 0.0
            else:
                s = tf.ones([tf.shape(rnn_out)[0],])*0.5
            #    scale_kl += tf.reduce_mean(0.5 * tf.reduce_sum(tf.log(0.1)- scale_log_variance - 1.0 + scale_variance/0.1
            #                + tf.square(scale_mean - 0.0)/0.1 , 1))

        return s, scale_kl
    """
    def comp_scale(rnn_out2,scale_kl):
        with tf.variable_scope("scale",reuse=DO_SHARE):
            prob = slim.conv2d( rnn_out2, 1, [3, 3], rate=1, scope="prob1", activation_fn=None)
            prob = slim.conv2d( prob, 1, [3, 3], rate=1, scope="prob2", activation_fn=None)
            s = 0.5 + slim.conv2d(rnn_out2, 1, [3, 3], rate=1, scope="conv_s", activation_fn=None)
            s = tf.transpose(s, [0, 3, 1, 2])
            s = tf.reduce_sum(prob * s, axis=[2, 3])[:, 0]
        return s, scale_kl
    """
    """
    def comp_shift(input,x_hat):
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
            #x, y = shift[:, 0], shift[:, 1]
            #shift_kl += tf.reduce_mean(0.5 * tf.reduce_sum( tf.log(2.1) - shift_log_variance - 1.0 + shift_variance/2.1 +
            #    tf.square(shift_mean - 0.0)/2.1 , 1))

        return x,y #, shift_kl
        """
    def comp_shift2(rnn_out2):
        with tf.variable_scope("shift",reuse=DO_SHARE):
            def meshgrid(h):
              r = np.arange(0.5, h, 1) / (h / 2) - 1
              ranx, rany = tf.meshgrid(r, r)
              return tf.to_float(ranx), tf.to_float(rany)
            #print("rnn_out",rnn_out.shape)
            prob = slim.conv2d( rnn_out2, 1, [3, 3], rate=1, scope="prob1", activation_fn=None)
            prob = slim.conv2d( prob, 1, [3, 3], rate=1, scope="prob2", activation_fn=None)
            #prob = prob
            prob = tf.transpose(prob, [0, 3, 1, 2])

            prob = tf.reshape(prob, [-1,pic_size * pic_size])
            #prob = tf.nn.softmax(prob, name="softmax")
            prob = tf.nn.softplus(prob)
            prob_sum = tf.maximum(tf.expand_dims(tf.reduce_sum(prob,1),1),1e-10)
            prob = prob/prob_sum
            #print("prob_sum",prob.shape)

            ranx, rany = meshgrid(pic_size)
            prob = tf.reshape(prob, [-1,pic_size, pic_size])
            #print("prob",prob.shape)
            #print("ranx",ranx.shape)


            #print("x",x.shape)
            if t == 0:
                num_clusters = 0
                x = tf.zeros([tf.shape(rnn_out2)[0],])
                y = tf.zeros([tf.shape(rnn_out2)[0],])
            else:
                num_clusters = T-t
                if num_clusters == 1:
                    mean1 = tf.expand_dims(tf.reduce_sum(prob * ranx, axis=[1, 2]),axis=1)#[:, 0]
                    mean2 = tf.expand_dims(tf.reduce_sum(prob * rany, axis=[1, 2]),axis=1)#[:, 0]
                    prob = tf.reshape(prob, [-1, pic_size * pic_size])
                    ranx = tf.reshape(ranx,[1,pic_size * pic_size])
                    rany = tf.reshape(rany,[1,pic_size * pic_size])
                    covar = tf.reduce_sum(prob*(tf.square(ranx-mean1)+tf.square(rany-mean2))*0.5,axis=1)
                    x = sample_from_mvn(mean1[:,0], covar)
                    y = sample_from_mvn(mean2[:,0], covar)

                else:
                    ranx = tf.reshape(ranx,[1,pic_size * pic_size])
                    rany = tf.reshape(rany,[1,pic_size * pic_size])
                    ranx = ranx*tf.ones(tf.shape(rnn_out2)[0])
                    prob = tf.reshape(prob, [-1, pic_size * pic_size])
                    weights = tf.ones((tf.shape(rnn_out2)[0],num_clusters))*(1/num_clusters)
                    covar = tf.ones((tf.shape(rnn_out2)[0],num_clusters,2))*(1/10)
                    index = ( tf.range(0,num_clusters,1)/num_clusters)
                    #mean = tf.ones((tf.shape(rnn_out2)[0],num_clusters,2))*(0.5)
                    multiply = tf.constant([1])*tf.shape(rnn_out2)[0]
                    mean1 = tf.cast(tf.cos(index*6.283185)*0.5,tf.float32)
                    mean2 = tf.cast(tf.sin(index*6.283185)*0.5,tf.float32)#tf.stack([tf.sin(index*6.283185)*0.5,tf.cos(index*6.283185)*0.5])
                    #mean = tf.to_float(mean)#tf.cast(mean,tf.float32)
                    #print("mean",mean1.shape)
                    mean1 = tf.tile(mean1,multiply)
                    mean1 = tf.reshape(mean1,[-1,num_clusters])
                    mean2 = tf.tile(mean2,multiply)
                    mean2 = tf.reshape(mean2,[-1,num_clusters])
                    mean = tf.stack([mean1,mean2],2)
                    for i in range(5):
                        #E-step
                        #for j in range(pic_size * pic_size):
                        for k in range(num_clusters):

                            cap =-(tf.square(ranx-mean[:,k,0:1])/(2*covar[:,k:k+1,0])+tf.square(rany-mean[:,k,1:2])/(2*covar[:,k:k+1,1]))
                            #cap = tf.Print(cap,[i,k,"cap",cap[0,0]])
                            cap = weights[:,k:k+1]*tf.exp(cap)/tf.sqrt(2*np.pi*covar[:,k:k+1,0])/tf.sqrt(2*np.pi*covar[:,k:k+1,1])*prob
                            #cap = tf.Print(cap,[i,k,"cap",cap[0,0]])
                            cap = tf.maximum(tf.expand_dims(cap,axis=2),1e-10)
                            #print("cap",cap.shape)
                            #print("ranx",ranx.shape)
                            #print("covar",covar[:,k:k+1,0].shape)
                            if k == 0:
                                resp = cap
                            else:
                                resp = tf.concat([resp,cap],axis=2)

                            #print("resp",resp.shape)
                        resp_sum = tf.reduce_sum(resp, 2, keepdims=True)
                        resp = resp/ resp_sum
                        #resp = tf.Print(resp,[i,"resp",resp[0,0]])
                        #print("resp",resp.shape)
                        #M-step
                        count = tf.reduce_sum(resp, 1)
                        weights  = count/(pic_size * pic_size)
                        #print("resp",resp.shape)
                        #print("count",count.shape)
                        #print("mean",mean.shape)
                        for k in range(num_clusters):
                            mean_sum1 = tf.expand_dims(tf.reduce_sum(resp[:,:,k]*ranx*prob,1, keepdims=True),axis = 1)
                            mean_sum2 = tf.expand_dims(tf.reduce_sum(resp[:,:,k]*rany*prob,1, keepdims=True),axis = 1)
                            weighted_sum = tf.concat([mean_sum1,mean_sum2],axis=2)/tf.expand_dims(count[:,k:k+1],axis = 1)
                            cover_sum1 = tf.expand_dims(tf.reduce_sum(resp[:,:,k]*prob*tf.square(ranx-mean[:,k,0:1]),1, keepdims=True),axis = 1)
                            cover_sum2 = tf.expand_dims(tf.reduce_sum(resp[:,:,k]*prob*tf.square(rany-mean[:,k,1:2]),1, keepdims=True),axis = 1)
                            cover_sum = tf.concat([cover_sum1,cover_sum2],axis=2)/tf.expand_dims(count[:,k:k+1],axis = 1)
                            #cover_sum = (cover_sum1+cover_sum2)*0.5
                            #covar_sum1 = tf.expand_dims(tf.reduce_sum(resp[:,:,k]*ranx*prob,1),axis = 1)
                            #print("weighted_sum",weighted_sum.shape)
                            #print("cover_sum1",cover_sum1.shape)
                            if k==0:
                                mean_temp = weighted_sum
                                cover_temp = cover_sum
                                #cover_sum = (cover_sum1+cover_sum2)*0.5/count[:,k:k+1]
                            else:
                                mean_temp = tf.concat([mean_temp,weighted_sum],axis=1)
                                cover_temp = tf.concat([cover_temp,cover_sum],axis=1)
                            #print("cover_sum",cover_sum.shape)
                            #print("cover_temp",cover_temp.shape)
                        mean = mean_temp
                        covar = cover_temp
                        #exit()
                    mean1,mean2 = mean[:,0,0], mean[:,0,1]
                    #print("mean",mean.shape)
                    #print("covar",covar.shape)
                    U = tf.random_uniform(tf.shape(weights),minval=0,maxval=1)
                    sample_gumbel = -tf.log(-tf.log(U + 1e-20)+1e-20)
                    gumbel_softsample = tf.nn.softmax(weights+sample_gumbel)
                    #print("weights",weights.shape)
                    sample = mean +tf.random_normal(tf.shape(mean))* tf.sqrt(covar)
                    sample = tf.reduce_sum(tf.expand_dims(weights,axis=2)*sample,axis=1)
                    #print("sample",sample.shape)
                    x,y = sample[:,0],sample[:,1]



                    #x = mean[:,0,0]
                prob = tf.reshape(prob, [-1, pic_size, pic_size])
            print("x",x.shape)
                #print("mean",mean[:,0,0].shape)

        return x, y, prob #, mean

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
            #if not is_train:
            #    z_pres = tf.round(z_pres)
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

            rec                   = decoder(rec_sample, is_train, [3,7,7,16],vae_generative_units, latent_dim)
            #rec = tf.reshape(rec,[-1,win_size,win_size,1])
            vae_kl += tf.reduce_mean(0.5 * tf.reduce_sum( tf.log(10.1) - rec_log_var - 1.0 + tf.exp(rec_log_var)/10.1 +
                tf.square(rec_mean - 0.0)/10.1 , 1))
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

    input = input_layer #tf.reshape(input_layer, shape=[-1, pic_size,pic_size, 3])
    #with tf.variable_scope("bg_recon"):
    #    rec_mean, rec_log_var = encoder(input_layer,is_train,(24, 16, 8),latent_dim)
    #    input                 = decoder(rec_mean, is_train, [3,7,7,8],(8, 16, 24), latent_dim)
    #    input = tf.sigmoid(input)


    scale_kl, shift_kl, z_pres_kl, vae_kl = 0.0, 0.0, 0.0, 0.0
    cs = [0]*T
    s,x,y = [0]*T, [0]*T, [0]*T
    m = [0]*T
    z_pres = 1.0
    rnn_state = lstm_rnn.zero_state(tf.shape(input_layer)[0],tf.float32)

    for t in range(T):
        c_prev = tf.zeros((tf.shape(input_layer)[0],pic_size,pic_size,1))if t==0 else cs[t-1]
        x_hat = input - tf.sigmoid(c_prev)
        rnn_out1,rnn_out2, state = hidden_rnn(input,x_hat,rnn_state)
        s[t], scale_kl = comp_scale(rnn_out1,scale_kl)
        #x[t], y[t], scale_kl = comp_shift(rnn_out,scale_kl)
        x[t], y[t], prob = comp_shift2(rnn_out2)
        sigma2,gamma = comp_para(rnn_out1)
        z_pres, z_pres_kl = comp_pres(rnn_out1, z_pres_kl)
        #z_pres *= z_pres_t #z_pres = z_pres_t
        zeros = tf.zeros_like(s[t])
        theta = tf.stack([s[t], zeros, x[t], zeros, s[t], y[t]], 1)
        itheta = tf.stack([1.0/s[t], zeros, -x[t]/s[t], zeros, 1.0/s[t], -y[t]/s[t]], 1)
        window = read(input,theta)
        window_hat = read(x_hat,theta)
        rec, vae_kl= vae(window,window_hat,vae_kl)
        cs[t] = c_prev + z_pres*write(rec, itheta, sigma2)#tf.clip_by_value(c_prev + write(rec, itheta, sigma2),0.0,1.0)
        DO_SHARE=True

    window_recon = tf.sigmoid(cs[T-1])
    #print("window_recon",window_recon.shape)

    # output_true shall have the original image for error calculations
    output_true = tf.placeholder('float32', [None, pic_size,pic_size,3], name = "Truth")

    with tf.variable_scope("recon_loss1"):
        # define our cost function
        #meansq =    tf.reduce_mean(tf.square(rec - output_true))
        trick = - tf.reduce_mean(tf.square(window_hat))*500
        meansq =    tf.reduce_mean(tf.square(window_recon - output_true))
        meansq *= win_size*win_size#*0.01
        meansq2 =    tf.reduce_mean(tf.square(input - output_true))
        binarcs = -tf.reduce_mean(
            output_true * tf.log(window_recon+ 10e-10) +
            (1.0 - output_true) * tf.log(1.0 - window_recon+ 10e-10))
        binarcs *=win_size
        if data_type == 1 or data_type == 2:
            vae_loss = binarcs + meansq*2.5  +scale_kl +shift_kl+vae_kl*0.0 + z_pres_kl*0.1 +trick*0.0
        elif data_type == 2:
            vae_loss = meansq +scale_kl +shift_kl+vae_kl*0.01

    t_vars = tf.trainable_variables()
    vae_vars = [var for var in t_vars if 'vae' in var.name]
    # define our optimizer
    learn_rate = 0.001#0.0005   # how fast the model should learn
    slimopt = slim.learning.create_train_op(vae_loss, tf.train.AdamOptimizer(learn_rate))

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
                    #tf.summary.scalar("shift_kl", shift_kl),
                    tf.summary.scalar("z_pres_kl", z_pres_kl),
                    tf.summary.scalar("vae_kl", vae_kl),
                    tf.summary.scalar("tricky_term", trick),
                    #tf.summary.histogram("scale0", s[0]),
                    #tf.summary.histogram("scale1", s[1]),
                    tf.summary.image("rec_window", tf.reshape(rec, [-1, win_size, win_size, 3])),
                    #tf.summary.image("sum_map", tf.reshape(sss, [-1, pic_size, pic_size, 1])),
                    tf.summary.image("cs0", tf.reshape(tf.sigmoid(cs[0]), [-1, pic_size, pic_size, 3]) ),
                    #tf.summary.image("input_recon", tf.reshape(input, [-1, pic_size, pic_size, 3]) ),
                    tf.summary.image("recon", tf.reshape(window_recon, [-1, pic_size, pic_size, 3]) ),
                    tf.summary.image("window", tf.reshape(window, [-1, win_size, win_size, 3])),
                    tf.summary.image("prob", tf.reshape(prob, [-1, pic_size, pic_size,1])),
                    tf.summary.image("error", tf.reshape(tf.reduce_sum(tf.square(x_hat),3, keepdims=True), [-1, pic_size, pic_size, 1])),
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

        x_true = get_batch(200,10,False)
        filename = os.path.join(model_name, "compair%02d.png"%(index))
        num_rows = 5
        num_cols = 3
        num_images = num_rows*num_cols
        plt.figure(figsize=(2*2*num_cols, 2*num_rows))
        #print(x_encoded[0].shape[0])

        for i in range(num_images):
            j = i
            any_image = x_true[j]
            x_decoded,p_map,error,sh1,sh2,sc = sess.run([window_recon,prob,meansq
                            ,x[2],y[2],s[2]],\
                           feed_dict={input_layer:[any_image], output_true:[any_image], is_train: False})
            x_dec = x_decoded[0]#.reshape(pic_size, pic_size)
            p_map = p_map[0]
            #print(p_map.shape)

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

            x_tt = x_true[j]#.reshape(pic_size, pic_size)
            #print(int(lx),int(ly),int(rx),int(ry),int(sx),int(sy))
            x_tt[max(0,min(pic_size-1,int(ly))):max(0,min(pic_size-1,int(ry))),max(0,min(pic_size-1,int(lx))),:] = .5
            x_tt[max(0,min(pic_size-1,int(ly))):max(0,min(pic_size-1,int(ry))),max(0,min(pic_size-1,int(rx))),:] = .5
            x_tt[max(0,min(pic_size-1,int(ly))),max(0,min(pic_size-1,int(lx))):max(0,min(pic_size-1,int(rx))),:] = .5
            x_tt[max(0,min(pic_size-1,int(ry))),max(0,min(pic_size-1,int(lx))):max(0,min(pic_size-1,int(rx))),:] = .5

            x_dec[max(0,min(pic_size-1,int(ly))):max(0,min(pic_size-1,int(ry))),max(0,min(pic_size-1,int(lx))),:] = .5
            x_dec[max(0,min(pic_size-1,int(ly))):max(0,min(pic_size-1,int(ry))),max(0,min(pic_size-1,int(rx))),:] = .5
            x_dec[max(0,min(pic_size-1,int(ly))),max(0,min(pic_size-1,int(lx))):max(0,min(pic_size-1,int(rx))),:] = .5
            x_dec[max(0,min(pic_size-1,int(ry))),max(0,min(pic_size-1,int(lx))):max(0,min(pic_size-1,int(rx))),:] = .5

            p_map[max(0,min(pic_size-1,int(ly))):max(0,min(pic_size-1,int(ry))),max(0,min(pic_size-1,int(lx)))] = .02
            p_map[max(0,min(pic_size-1,int(ly))):max(0,min(pic_size-1,int(ry))),max(0,min(pic_size-1,int(rx)))] = .02
            p_map[max(0,min(pic_size-1,int(ly))),max(0,min(pic_size-1,int(lx))):max(0,min(pic_size-1,int(rx)))] = .02
            p_map[max(0,min(pic_size-1,int(ry))),max(0,min(pic_size-1,int(lx))):max(0,min(pic_size-1,int(rx)))] = .02

            plt.title('%2d step' %index)
            ax = plt.subplot(num_rows, 2*num_cols, 2*i+1)
            plt.imshow(x_tt ,  cmap='Greys')
            plt.xlabel("(%.2f %.2f) %.2f"% (sh1,sh2,sc))
            plt.xticks([])
            plt.yticks([])
            ax = plt.subplot(num_rows, 2*num_cols, 2*i+2)
            plt.xlabel(error)
            plt.imshow(x_dec,  cmap='winter')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.savefig(filename)
        plt.close('all')
        #plt.show()

    # defining batch size, number of epochs and learning rate

    vari = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print("variable = ", vari)
    kl = 0
    all_images = get_batch(tot_images,0,True)
    test_imges = get_batch(200,10,False)
    for epoch in range(hm_epochs):
        epoch_loss = 0    # initializing error as 0
        for i in range(int(tot_images/batch_size)):
            #epoch_x = get_batch(batch_size,batch_size*i,True)
            epoch_x = all_images[ i*batch_size : (i+1)*batch_size ]
            _,c,pmap = sess.run([slimopt,vae_loss,prob],feed_dict={input_layer:epoch_x,
                      output_true:epoch_x, is_train:True})
            epoch_loss += c
            #print(np.sum(pmap[0])*28)
            #plt.imshow(pmap[0])
            #plt.show()
            #exit()
        #epoch_x = get_batch(200,10,False)
        epoch_x = test_imges
        summ = sess.run(summary, feed_dict={input_layer: epoch_x, \
           output_true: epoch_x, is_train: False})
        writer.add_summary(summ,epoch)
        if epoch%5==0:
            #r = np.arange(0.5, pic_size, 1) / (pic_size / 2) - 1
            #ranx, rany = np.meshgrid(r, r)
            #print("sum prob",np.sum(pmap[0]))
            #print("Ex",np.sum(pmap[0]*ranx),np.sum(pmap[0]*rany))
            #print("Ex",cc[0],np.sum(pmap[0]*rany))
            print('Epoch', epoch, '/', hm_epochs, 'loss:',epoch_loss)
            plot_results(model_name="air_test/",index =epoch)
        if epoch%20==19:
            all_images = get_batch(tot_images,0,True)
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
