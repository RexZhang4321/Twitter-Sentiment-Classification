import tensorflow as tf


class CNN_Model(object):
    def __init__(
            self, sequence_length, num_classes,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)

        self.embedded_chars_expanded = tf.expand_dims(self.input_x, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                # h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                h = tf.tanh(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout1"):
            self.h_drop1 = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        # Final (unnormalized) scores and predictions
        # with tf.name_scope("fc1-relu"):
        #     W1 = tf.get_variable(
        #         "W1",
        #         shape=[num_filters_total, 500],
        #         initializer=tf.contrib.layers.xavier_initializer())
        #     b1 = tf.Variable(tf.constant(0.01, shape=[500]), name="b1")
        #     l2_loss += tf.nn.l2_loss(W1)
        #     l2_loss += tf.nn.l2_loss(b1)
        #     self.fc1 = tf.nn.sigmoid(tf.nn.xw_plus_b(self.h_pool_flat, W1, b1), name="fc1-relu")
        #
        # with tf.name_scope("dropout1"):
        #     self.h_drop1 = tf.nn.dropout(self.fc1, self.dropout_keep_prob)
        #
        # with tf.name_scope("fc2-relu"):
        #     W2 = tf.get_variable(
        #         "W2",
        #         shape=[500, 300],
        #         initializer=tf.contrib.layers.xavier_initializer())
        #     b2 = tf.Variable(tf.constant(0.01, shape=[300]), name="b2")
        #     l2_loss += tf.nn.l2_loss(W2)
        #     l2_loss += tf.nn.l2_loss(b2)
        #     self.fc2 = tf.nn.tanh(tf.nn.xw_plus_b(self.h_drop1, W2, b2), name="fc12-relu")
        #
        # with tf.name_scope("dropout2"):
        #     self.h_drop2 = tf.nn.dropout(self.fc2, self.dropout_keep_prob)

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.01, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop1, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
