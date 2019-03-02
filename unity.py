import tensorflow as tf
import tensorflow.contrib.slim as slim
import traceback
import os
import sys

def colored_hook(home_dir):
  """Colorizes python's error message.
  Args:
    home_dir: directory where code resides (to highlight your own files).
  Returns:
    The traceback hook.
  """

  def hook(type_, value, tb):
    def colorize(text, color, own=0):
      """Returns colorized text."""
      endcolor = "\x1b[0m"
      codes = {
          "green": "\x1b[0;32m",
          "green_own": "\x1b[1;32;40m",
          "red": "\x1b[0;31m",
          "red_own": "\x1b[1;31m",
          "yellow": "\x1b[0;33m",
          "yellow_own": "\x1b[1;33m",
          "black": "\x1b[0;90m",
          "black_own": "\x1b[1;90m",
          "cyan": "\033[1;36m",
      }
      return codes[color + ("_own" if own else "")] + text + endcolor

    for filename, line_num, func, text in traceback.extract_tb(tb):
      basename = os.path.basename(filename)
      own = (home_dir in filename) or ("/" not in filename)

      print(colorize("\"" + basename + '"', "green", own) + " in " + func)
      print("%s:  %s" % (
          colorize("%5d" % line_num, "red", own),
          colorize(text, "yellow", own)))
      print("  %s" % colorize(filename, "black", own))

    print(colorize("%s: %s" % (type_.__name__, value), "cyan"))
  return hook
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
        generative_mean = slim.conv2d_transpose(next_layer, input_dim[0], [3, 3],activation_fn=None, padding='same', scope=scope)


    with tf.variable_scope("gen_sample"):
        #standard_normal_sample2 = tf.random_normal([input_size, input_dim])
        standard_normal_sample2 = tf.random_normal(tf.shape(generative_mean))
        generative_sample = generative_mean #+ standard_normal_sample2 * likelihood_std
        reconstruction = tf.nn.sigmoid(
            generative_sample
            ) #+ tf.square(standard_normal_sample2 * likelihood_std)

    return generative_sample#reconstruction
