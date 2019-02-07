import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim

def vae(inputs, input_dim, rec_hidden_units, latent_dim,
        gen_hidden_units, likelihood_std=0.0, activation=tf.nn.softplus):

    input_size = tf.shape(inputs)[0]
    next_layer = inputs
    for i in range(len(rec_hidden_units)):
        with tf.variable_scope("recognition_" + str(i+1)) as scope:
            next_layer = layers.fully_connected(
                next_layer, rec_hidden_units[i], activation_fn=activation, scope=scope
            )

    with tf.variable_scope("rec_mean") as scope:
        recognition_mean = layers.fully_connected(next_layer, latent_dim, activation_fn=None, scope=scope)
    with tf.variable_scope("rec_log_variance") as scope:
        recognition_log_variance = layers.fully_connected(next_layer, latent_dim, activation_fn=None, scope=scope)

    with tf.variable_scope("rec_sample"):
        standard_normal_sample = tf.random_normal([input_size, latent_dim])
        recognition_sample = recognition_mean + standard_normal_sample * tf.sqrt(tf.exp(recognition_log_variance))

    next_layer = recognition_sample
    for i in range(len(gen_hidden_units)):
        with tf.variable_scope("generative_" + str(i+1)) as scope:
            next_layer = layers.fully_connected(
                next_layer, gen_hidden_units[i], activation_fn=activation, scope=scope
            )

    with tf.variable_scope("gen_mean") as scope:
        generative_mean = layers.fully_connected(next_layer, input_dim, activation_fn=None, scope=scope)

    with tf.variable_scope("gen_sample"):
        standard_normal_sample2 = tf.random_normal([input_size, input_dim])
        generative_sample = generative_mean + standard_normal_sample2 * likelihood_std
        reconstruction = tf.nn.sigmoid(
            generative_mean
        )

    return generative_mean, recognition_mean, recognition_log_variance, recognition_sample

def encoder(inputs, is_train, rec_hidden_units, latent_dim, activation=tf.nn.softplus):

    input_size = tf.shape(inputs)[0]

    next_layer = inputs

    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        normalizer_fn=slim.batch_norm,
        activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
        normalizer_params={"is_training": is_train}):

        for i in range(len(rec_hidden_units)):
            next_layer  = slim.conv2d(next_layer , rec_hidden_units[i], [3, 3], scope="encode_conv%d" % (i*2+1))
            next_layer  = slim.conv2d(next_layer , rec_hidden_units[i], [3, 3], stride=2, scope="encode_conv%d" % (i*2+2))
        next_layer = slim.flatten(next_layer)


        with tf.variable_scope("rec_mean") as scope:
            #recognition_mean = layers.fully_connected(next_layer, latent_dim, activation_fn=None, scope=scope)
            recognition_mean = slim.fully_connected(next_layer, latent_dim, activation_fn=None, scope=scope)
        with tf.variable_scope("rec_log_variance") as scope:
            #recognition_log_variance = layers.fully_connected(next_layer, latent_dim, activation_fn=None, scope=scope)
            recognition_log_variance = slim.fully_connected(next_layer, latent_dim, activation_fn=None, scope=scope)

    return recognition_mean, recognition_log_variance


def decoder(inputs, is_train, input_dim, gen_hidden_units, latent_dim, activation=tf.nn.softplus):
    likelihood_std = 0.3

    with slim.arg_scope(
        [slim.conv2d_transpose, slim.fully_connected],
        normalizer_fn=slim.batch_norm,
        activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
        normalizer_params={"is_training": is_train}):

        next_layer = slim.fully_connected(inputs, input_dim[1] * input_dim[2] * input_dim[3])
        next_layer =tf.reshape(next_layer, [-1, input_dim[1] , input_dim[2], input_dim[3]])

        for i in range(len(gen_hidden_units)):
            next_layer  = slim.conv2d_transpose(next_layer , gen_hidden_units[i], [3, 3], scope="decode_conv%d" % (i*2+1))
            next_layer  = slim.conv2d_transpose(next_layer , gen_hidden_units[i], [3, 3], stride=2, scope="decode_conv%d" % (i*2+2))


    with tf.variable_scope("gen_mean") as scope:
        generative_mean = slim.conv2d_transpose(next_layer, 1, [3, 3], activation_fn=None, padding='same', scope=scope)


    with tf.variable_scope("gen_sample"):
        #standard_normal_sample2 = tf.random_normal([input_size, input_dim])
        standard_normal_sample2 = tf.random_normal(tf.shape(generative_mean))
        generative_sample = generative_mean #+ standard_normal_sample2 * likelihood_std
        reconstruction = tf.nn.sigmoid(
            generative_sample
            ) #+ tf.square(standard_normal_sample2 * likelihood_std)

    return reconstruction
