#!coding:utf-8
"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current


def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net

def transition_layer(net, layers, growth, scope='transition'):
    """
    We use 1×1 convolution followed by 2×2 average pooling as transition layers between two contiguous dense blocks.
    :param net:
    :param layers:
    :param growth:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope):
        net = slim.conv2d(net, 4 * growth, [1, 1], padding='SAME', scope='conv1x1')
        net = slim.avg_pool2d(net, [2, 2], stride=2, padding='VALID', scope='avgPool_1a_2x2')
    return net

def densenet(images, num_classes=1001, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates a variant of the densenet model.  juest  DenseNet-BC.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 24
    compression_rate = 0.5
    L = 4  # In our experiments on ImageNet, we use a DenseNet-BC structure with 4 dense blocks on 224x224 input images

    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:
            pass
            ##########################
            # Put your code here.
            ##########################
            # a convolution with 16 (or twice the growth rate for DenseNet-BC) output channels is performed on the
            # input images. For convolutional layers with kernel size 3x3, each side of the inputs is zero-padded
            # by one pixel to keep the feature-map size fixed
            net = slim.conv2d(images, 2 * growth, [7, 7], stride=2, padding='SAME', scope=scope + '_conv3x3')  # (32, 224, 224, 3) ===> (32, 112, 112, 48)
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='maxPool_3x3')                 # (32, 112, 112, 48) ===> (32, 56, 56, 48)
            end_points['input'] = net
            # block 1 (-1, 56, 56, 48) ==> (-1, 56, 56, 48)
            for layer in range(L - 1):
                scope_name = 'block{}'.format(layer)
                net = block(net, layer, growth, scope_name)
                end_points[scope_name] = net
                # transition_layer (-1, w, h, 48) ==> (-1, w, h, 96)
                scope_name = 'transition{}'.format(layer)
                net = transition_layer(net, layer, growth, scope_name)
                end_points[scope_name] = net

            # # block 2
            # net = block(net, 1, growth)
            # # transition_layer (-1, 28, 28, 48) ==> (-1, 14, 14, 48)
            # net = transition_layer(net, 2, growth)
            # # block 3
            # net = block(net, 1, growth)
            # # transition_layer  (-1, 14, 14, 48) ==> (-1, 7, 7, 48)
            # net = transition_layer(net, 3, growth)
            # block 4
            net = block(net, L, growth)

            # Final pooling and prediction
            # At the end of the last dense block, a global average pooling is performed and then a softmax classifier is attached.
            with tf.variable_scope('Logits'):
                net = tf.reduce_mean(net, [1, 2], keep_dims=False, name='global_pool')
                logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                  scope='Logits')
                end_points['Logits'] = logits
                end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')

    print('==='*20)
    print(logits)
    print(end_points['Predictions'])

    return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224
