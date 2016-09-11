import tensorflow as tf

slim = tf.contrib.slim

def model(images, num_classes=11):
    """Defines the digit model.
    
    Args:
        images: A `tf.Tensor` with rank 4.
        num_classes: A scalar denoting the number of classes.
    Returns:
        logits: A `tf.Tensor` with the logits.
    """
    
    with slim.arg_scope([slim.layers.conv2d], normalizer_fn=slim.batch_norm):
        net = slim.layers.conv2d(images, 128, 5, scope='conv1')
        net = slim.layers.max_pool2d(net, 2, scope='pool1')
        net = slim.layers.conv2d(net, 64, 5, scope='conv2')
        net = slim.layers.max_pool2d(net, 2, scope='pool2')
        net = slim.layers.flatten(net, scope='flatten')
        net = slim.layers.dropout(net, scope='dropout')
        net = slim.layers.fully_connected(net, 1024, scope='fc3')

    return slim.layers.fully_connected(net, num_classes, activation_fn=None, scope='logits')

