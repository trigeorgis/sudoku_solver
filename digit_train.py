import tensorflow as tf
import data_provider


from tensorflow.python.platform import tf_logging as logging
from digit_model import model

slim = tf.contrib.slim

# Initialize the argparse wrapper
flags = tf.app.flags
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_string('log_dir', 'ckpt', 'The logging directory.')

# Learning rate decay parameters
flags.DEFINE_float('initial_learning_rate', 0.001,
                   'Initial value of learning rate decay.')
flags.DEFINE_float('decay_steps', 10000, 'Learning rate decay steps.')
flags.DEFINE_float('decay_rate', 0.9, 'Learning rate decay rate.')

FLAGS = tf.app.flags.FLAGS

def train():
    record_names = ["sudoku_train"] * 50 + ["mnist_train"]
    record_paths = ["data/{}.tfrecords".format(x) for x in record_names]

    # Load data in batches.
    images, labels = data_provider.get_data(
        record_paths, batch_size=FLAGS.batch_size, is_training=True)
    
    # Define network
    with slim.arg_scope([slim.layers.dropout, slim.batch_norm],
                        is_training=True):
        predictions = model(images)
    
    # Display images to tensorboard
    tf.image_summary('images', images, max_images=5)

    # Define loss function
    slim.losses.softmax_cross_entropy(predictions, labels)
    total_loss = slim.losses.get_total_loss()
    
    tf.scalar_summary('loss', total_loss)

    # Create learning rate decay
    global_step = slim.get_or_create_global_step()
    
    learning_rate = tf.train.exponential_decay(
        FLAGS.initial_learning_rate,
        global_step=global_step,
        decay_steps=FLAGS.decay_steps,
        decay_rate=FLAGS.decay_rate)

    # Optimizer to use.
    optimizer = tf.train.AdamOptimizer(learning_rate)
    
    # Create training operation
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    logging.set_verbosity(1)
    
    # Start training
    slim.learning.train(train_op, FLAGS.log_dir,
                        save_summaries_secs=20,
                        save_interval_secs=60,
                        log_every_n_steps=100)

if __name__ == '__main__':
    train()