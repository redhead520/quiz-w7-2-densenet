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
    """
    Dense Block
    稠密连接：每个卷积层的输入都是前面所有层的输出,即与前面所有层相连.
    layers个卷积层的dense block,有L层（L=layers+1,1为input）, 就会有　L(L+1)/2个连接
    这种连接方式使得特征和梯度的传递更加有效，网络也就更加容易训练
    :param net: 输入数据
    :param layers: 这个　Dense block 有几个卷积层
    :param growth: 由于Dense Block中的每个卷积层的输入都是前面所有层的输出在feature-maps的维度相加，导致每层的输入数据的feature-maps的数量几何递增，
    而growth就是为了控制每层的feature-maps的数量，使每个卷积层的输出的feature-maps数量都为growth,
    :param scope: 命名空间
    :return:　Dense Block输出结果
    """
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net

def transition_layer(net, num_outputs, scope='transition'):
    """
    Transition Layer 过渡层(稠密链接)
    We use 1×1 convolution followed by 2×2 average pooling as transition layers between two contiguous dense blocks.
    对上一层输出，下一层输入数据的width,height,depth３个维度的特征进一步提取,
    使width,height降低一半, depth降为depth*compression_rate
    :param net:输入数据
    :param num_outputs:输出结果的feature-maps数量
    :param scope:命名空间
    :return:　返回输出结果
    """
    with tf.variable_scope(scope):
        # net = slim.conv2d(net, num_outputs, [1, 1], padding='SAME', scope='conv1x1')
        net = bn_act_conv_drp(net, num_outputs, [1, 1], scope='conv1x1')
        net = slim.avg_pool2d(net, [2, 2], stride=2, padding='VALID', scope='avgPool_1a_2x2')
    return net

def GAP(net, keep_dims=False, scope='Global_Average_Pooling'):
    """
    Global Average Pooling 全局平均池化
    gap (-1, w, h, d) ==> (-1, 1, 1, d)
    :param net:输入数据
    :param keep_dims:　是否保留输出结果中axis=[1,2]的维度．（axis=[0,1,2,3]）
    :param scope:命名空间
    :return:　返回输出结果
    """
    # net = tf.reduce_mean(net, [1, 2], keep_dims=keep_dims, name=scope)  # tf.reduce_mean相当于下面俩行
    net = slim.avg_pool2d(net, net.shape[1:3], padding='valid', scope=scope)
    net = net if keep_dims else tf.squeeze(net, [1, 2])
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
    # For DenseNet-BC, the networks with configurations
    growth = 24
    compression_rate = 0.5
    # In our experiments on ImageNet, we use a DenseNet-BC structure with 4 dense blocks on 224x224 input images
    block_numbers = 4               # the number of dense blocks
    block_layers = [6, 12, 24, 16]  # DenseNet-121 params: the layers of dense blocks


    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:

            ##########################
            # Put your code here.this is a DenseNet-BC structure, quiz just like ImageNet DataSet
            # In our experiments on ImageNet, we use a DenseNet-BC structure with 4 dense blocks on 224×224 input images
            ##########################
            # The initial convolution layer comprises 2k convolutions of size 7×7 with stride 2;
            # (batch_size, 224, 224, 3) ===> (-1, 112, 112, 48)
            net = slim.conv2d(images, 2 * growth, [7, 7], stride=2, padding='SAME', scope=scope + '_conv3x3')
            # (-1, 112, 112, 48) ===> (-1, 56, 56, 48)
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME', scope='maxPool_3x3')
            end_points['input'] = net

            # block 1 　==> block L - 1
            for index, layers in enumerate(block_layers[:-1]):
                scope_name = 'block{}x{}'.format(index, layers)
                net = block(net, layers, growth, scope_name)

                # transition_layer
                scope_name = 'transition{}'.format(index)
                net = transition_layer(net, reduce_dim(net), scope_name)

            # last block
            scope_name = 'block{}'.format(block_numbers)
            net = block(net, block_layers[-1], growth, scope_name)

            # Final pooling and prediction
            # At the end of the last dense block, a global average pooling is performed and then a softmax classifier is attached.
            with tf.variable_scope('Logits'):
                # GAP
                net = GAP(net, keep_dims=False, scope='Global_Average_Pooling')
                stddev = tf.sqrt(1.0/int(net.shape[-1]))
                logits = slim.fully_connected(net,
                                              num_classes,
                                              activation_fn=None,
                                              weights_initializer=trunc_normal(stddev),
                                              scope='Logits')
                end_points['Logits'] = logits
                end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')

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
