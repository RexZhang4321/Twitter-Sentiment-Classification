import tensorflow as tf
import numpy as np
import os
import time
import datetime
import cnn_preprocessing
from cnn_model import CNN_Model
from tensorflow.contrib import learn

dataset = '2-points'

tf.flags.DEFINE_string("train_data_file", "../data/semeval/train.tsv", "Data source for the 3 point training data.")
tf.flags.DEFINE_string("train_data_file_2point", "../data/training.1600000.processed.noemoticon.csv", "Data source for the 2 point train data.")
tf.flags.DEFINE_string("test_data_file_2point", "../data/testdata.manual.2009.06.14.csv", "Data source for the 2 point test data.")
tf.flags.DEFINE_string("test_data_file", "../data/semeval/test.tsv", "Data source for the 3 point test data.")

# Model Hyperparameters
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 300, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.5, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("learning_rate", 3e-3, "Learning Rate (default: 1e-3)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if dataset == '2-points':
    sequence_length = 30
    num_classes = 2
elif dataset == '3-points':
    sequence_length = 24
    num_classes = 3
embedding_size = 300

print("Loading data...")
if dataset == '3-points':
    x_train, y_train = cnn_preprocessing.load_data_and_labels(FLAGS.train_data_file)
    x_dev, y_dev = cnn_preprocessing.load_data_and_labels(FLAGS.test_data_file)
elif dataset == '2-points':
    x_train, y_train = cnn_preprocessing.load_data_and_labels(FLAGS.train_data_file_2point)
    x_dev, y_dev = cnn_preprocessing.load_data_and_labels(FLAGS.test_data_file_2point)
    # x, y = cnn_preprocessing.load_data_and_labels(FLAGS.data_file_2point)
    # dev_sample_index = -1 * int(0.1 * float(len(y)))
    # x_train, x_dev = x[:dev_sample_index], x[dev_sample_index:]
    x_dev = cnn_preprocessing.convert2vec(x_dev, sequence_length, cnn_preprocessing.model)
    # y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = CNN_Model(
            sequence_length,
            num_classes,
            embedding_size,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables(), max_to_keep = 0)


        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy, precision, recall, f1 = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.precision, cnn.recall, cnn.f1],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}, precision {:g}, recall {:g}, f1 {:g},".format(time_str, step, loss, accuracy, precision, recall, f1))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = cnn_preprocessing.batch_iter(
            (x_train, y_train), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            if dataset == '3-points':
                x_batch = batch[0]
            elif dataset == '2-points':
                x_batch = cnn_preprocessing.convert2vec(np.array(batch[0]), sequence_length, cnn_preprocessing.model)
            y_batch = batch[1]
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))