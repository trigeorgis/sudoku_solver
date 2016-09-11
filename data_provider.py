import tensorflow as tf
from pathlib import Path


def read_and_decode(filename_queue):
    """Method that reads and decodes an example.
    
    Args:
        filename_queue : The files queue.
    
    Returns:
        image: The loaded image as a `tf.Tensor`.
        label: The label of the image as a `tf.Tensor`.
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    
    image = tf.reshape(image, (height, width, 1))
    
    image = tf.to_float(image)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)
    
    return image, label


def get_data(paths, batch_size, is_training=False, num_classes=11, image_size=28):
    """Retrieves the data from the tfrecords in batches.
    
    Args:
        paths: a list with the paths to the tfrecords
            where the data reside.
        batch_size: the size of the batches.
        is_training: whether in training mode or not.
    Return:
        images: a `tf.Tensor` with dimensions [batch_size, 28, 28, 1].
        labels: a `tf.Tensor` with dimensions [batch_size, 11].
    """

    # Create filename queue and read
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            paths, num_epochs=None, shuffle=True)

        # Even when reading in multiple threads, share the filename queue.
        image, label = read_and_decode(filename_queue)
        image /= 255.
        
        if is_training:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.resize_images(image, image_size + 5, image_size + 5)
            image = tf.random_crop(image, (image_size, image_size, 1), seed=42)
            image.set_shape((image_size, image_size, 1))
        else:
            image = tf.image.resize_images(image, image_size, image_size)

        label = tf.one_hot(label, num_classes, dtype=tf.int32)

        images, sparse_labels = tf.train.shuffle_batch(
                [image, label], batch_size=batch_size, num_threads=4,
                capacity=100000, min_after_dequeue=batch_size*4)

    return images, sparse_labels


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_data(images, filename):
    """Method that saves the provided images and labels to
    tfrecords file using the tensorflow record writer.
    
    Args:
        images: A list with `Image` items to serialize.
        filename : A `str` with the filename to use.
    """
    # Save data
    filename = str(Path('data') / (filename + '.tfrecords'))

    writer = tf.python_io.TFRecordWriter(filename)

    for im in images:
        image_raw = im.pixels[0].tostring()
        height, width = im.shape
        label = int(im.path.stem.split('_')[0])
        if label == 0:
            label = 10

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_raw)}))
        
        writer.write(example.SerializeToString())

    writer.close()
