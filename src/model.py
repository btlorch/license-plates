from ops import conv2d, weights_variable_xavier, bias_variable, weights_variable_truncated_normal
from writer import BufferedWriter, DATA_IMAGES, DATA_CHAR_LABELS, DATA_CHAR_PROBABILITIES
from utils import setup_custom_logger
import tensorflow as tf
import numpy as np
import h5py
import time
import os


log = setup_custom_logger("LicensePlatesCNN")


# Some string constants
CONV0_WEIGHTS = "conv0_weights"
CONV0_BIAS = "conv0_bias"
CONV1_WEIGHTS = "conv1_weights"
CONV1_BIAS = "conv1_bias"
CONV2_WEIGHTS = "conv2_weights"
CONV2_BIAS = "conv2_bias"
CONV3_WEIGHTS = "conv3_weights"
CONV3_BIAS = "conv3_bias"
CONV4_WEIGHTS = "conv4_weights"
CONV4_BIAS = "conv4_bias"
CONV5_WEIGHTS = "conv5_weights"
CONV5_BIAS = "conv5_bias"
CONV6_WEIGHTS = "conv6_weights"
CONV6_BIAS = "conv6_bias"
CONV7_WEIGHTS = "conv7_weights"
CONV7_BIAS = "conv7_bias"
FC0_WEIGHTS = "fc0_weights"
FC0_BIAS = "fc0_bias"
FC1_WEIGHTS = "fc1_weights"
FC1_BIAS = "fc1_bias"
FC_CHAR0_WEIGHTS = "fc_char0_weights"
FC_CHAR0_BIAS = "fc_char0_bias"
FC_CHAR1_WEIGHTS = "fc_char1_weights"
FC_CHAR1_BIAS = "fc_char1_bias"
FC_CHAR2_WEIGHTS = "fc_char2_weights"
FC_CHAR2_BIAS = "fc_char2_bias"
FC_CHAR3_WEIGHTS = "fc_char3_weights"
FC_CHAR3_BIAS = "fc_char3_bias"
FC_CHAR4_WEIGHTS = "fc_char4_weights"
FC_CHAR4_BIAS = "fc_char4_bias"
FC_CHAR5_WEIGHTS = "fc_char5_weights"
FC_CHAR5_BIAS = "fc_char5_bias"
FC_CHAR6_WEIGHTS = "fc_char6_weights"
FC_CHAR6_BIAS = "fc_char6_bias"


class LicensePlatesCNN(object):
    def __init__(self,
                 sess,
                 checkpoint_dir,
                 summary_dir,
                 input_channels=3,
                 num_distinct_chars=36,
                 auto_set_up_model=True,
                 convert_input=None):
        """
        Set up CNN member variables.
        Automatically sets up the computational graph if not explicitly disabled.
        :param sess: TensorFlow session
        :param checkpoint_dir: directory where to store the latest checkpoint to or to read the latest checkpoint from
        :param summary_dir: directory where to store summaries for visualization with TensorBoard
        :param input_channels: number of input channels provided by the data set: 1 for grayscale, 3 for color.
        :param num_distinct_chars: number of distinct characters which the network is supposed to distinguish
        :param auto_set_up_model: flag whether to build the graph right now. If False, _build_model() will need to be invoked manually.
        """

        # Constants
        self._model_name = "LicensePlatesCNN"
        self._training_batch_size = 32
        self._num_steps_to_show_loss = 100
        self._num_steps_to_check = 1000
        self._initial_patience = 100
        self._max_length = 7
        self._report_accuracy_top_k = 1

        # Copy constructor arguments
        self._sess = sess
        self._checkpoint_dir = checkpoint_dir
        self._summary_dir = summary_dir
        self._input_channels = input_channels
        self._num_distinct_chars = num_distinct_chars
        self._convert_input = convert_input

        # The following variables are set during training
        self._eval_summary_writer = None

        # The following variables will be set by build_model()
        self._images = None
        self._char_labels = None
        self._drop_rate = None
        self._output = None
        self._output_logits = None
        self._char_top_k_accuracies_samplewise = None
        self._char_top_k_accuracies_mean = None
        self._weight_vars = dict()
        self._weight_var_placeholders = dict()
        self._weight_vars_previous_eval = dict()
        self._saver = None

        if auto_set_up_model:
            self._build_model()

    def _build_model(self):
        # Input 100x50 images
        self._images = tf.placeholder(tf.float32, [None, 50, 100, self._input_channels], name="images")
        # Placeholder for output labels of size batch_size x num_characters x num_distinct_characters (including null chararacter)
        self._char_labels = tf.placeholder(tf.float32, [None, self._max_length, self._num_distinct_chars + 1], name="char_labels")
        # Dropout probability
        self._drop_rate = tf.placeholder(tf.float32, name="drop_rate")
        # Set up computational graph
        self._output, self._output_logits = self.model(self._images)

        # Set up placeholder and tensor for top-k accuracy
        self._set_up_eval_vars_and_ops()

        # Saving only relative paths is particularly useful when we copy a saved model to another machine
        self._saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)

    def _set_up_eval_vars_and_ops(self):
        """
        Set up tensor and placeholder that are used in each evaluation step.
        This avoid recreation of these tensors in every evaluation step.
        """

        # tf.nn.in_top_k cannot work with 3-D matrices, but with ndarrays of shape [batch_size, num_classes] only
        # Since we know the maximum length of the license numbers in advance, we can set up one tf.nn.in_top_k method for each character and stack the results
        char_top_k_accuracies = []
        for i in range(self._max_length):
            with tf.variable_scope("char_{:d}".format(i)):
                class_ids_vector = tf.argmax(self._char_labels[:, i, :], axis=1)
                top_k_accuracy = tf.nn.in_top_k(predictions=self._output[:, i, :], targets=class_ids_vector, k=self._report_accuracy_top_k)
                char_top_k_accuracies.append(top_k_accuracy)

        # Stack per-character results and average over all characters for each sample
        char_top_k_accuracies = tf.reduce_mean(tf.cast(tf.stack(char_top_k_accuracies, axis=1), tf.float32), axis=1)

        # Per character top k accuracy for each sample
        self._char_top_k_accuracies_samplewise = char_top_k_accuracies
        self._char_top_k_accuracies_mean = tf.reduce_mean(char_top_k_accuracies)

    def model(self, input):
        """
        Bakes the CNN architecture into a computational graph
        :param input: Tensor to contain the input image. May be a `tf.Variable` or `tf.placeholder`.
        :return: softmax outputs and output logits before softmax activation as 2-tuple
        """

        # (50, 100, 1) -> (50, 100, 64)
        with tf.variable_scope("conv0"):
            conv0_weights = weights_variable_xavier([3, 3, self._input_channels, 64], name=CONV0_WEIGHTS)
            conv0_bias = bias_variable([64], value=0.1, name=CONV0_BIAS)
            conv0_z = conv2d(input, conv0_weights) + conv0_bias
            conv0_a = tf.nn.relu(conv0_z)

        # (50, 100, 64) -> (50, 100, 64)
        with tf.variable_scope("conv1"):
            conv1_weights = weights_variable_xavier([3, 3, 64, 64], name=CONV1_WEIGHTS)
            conv1_bias = bias_variable([64], value=0.1, name=CONV1_BIAS)
            conv1_z = conv2d(conv0_a, conv1_weights) + conv1_bias
            conv1_a = tf.nn.relu(conv1_z)

        # (50, 100, 64) -> (25, 50, 64)
        # TODO does variable_scope make sense if pooling layers don't even have variables?
        with tf.variable_scope("pool0"):
            pool0 = tf.layers.max_pooling2d(conv1_a, pool_size=[2, 2], strides=2, padding="same")

        # (25, 50, 64) -> (25, 50, 128)
        with tf.variable_scope("conv2"):
            conv2_weights = weights_variable_xavier([3, 3, 64, 128], name=CONV2_WEIGHTS)
            conv2_bias = bias_variable([128], value=0.1, name=CONV2_BIAS)
            conv2_z = conv2d(pool0, conv2_weights) + conv2_bias
            conv2_a = tf.nn.relu(conv2_z)

        # (25, 50, 128) -> (25, 50, 128)
        with tf.variable_scope("conv3"):
            conv3_weights = weights_variable_xavier([3, 3, 128, 128], name=CONV3_WEIGHTS)
            conv3_bias = bias_variable([128], value=0.1, name=CONV3_BIAS)
            conv3_z = conv2d(conv2_a, conv3_weights) + conv3_bias
            conv3_a = tf.nn.relu(conv3_z)

        # (25, 50, 128) -> (25, 50, 128)
        with tf.variable_scope("pool1"):
            pool1 = tf.layers.max_pooling2d(conv3_a, pool_size=[2, 2], strides=1, padding="same")

        # (25, 50, 128) -> (25, 50, 256)
        with tf.variable_scope("conv4"):
            conv4_weights = weights_variable_xavier([3, 3, 128, 256], name=CONV4_WEIGHTS)
            conv4_bias = bias_variable([256], value=0.1, name=CONV4_BIAS)
            conv4_z = conv2d(pool1, conv4_weights) + conv4_bias
            conv4_a = tf.nn.relu(conv4_z)

        # (25, 50, 256) -> (25, 50, 256)
        with tf.variable_scope("conv5"):
            conv5_weights = weights_variable_xavier([3, 3, 256, 256], name=CONV5_WEIGHTS)
            conv5_bias = bias_variable([256], value=0.1, name=CONV5_BIAS)
            conv5_z = conv2d(conv4_a, conv5_weights) + conv5_bias
            conv5_a = tf.nn.relu(conv5_z)

        # (25, 50, 256) -> (13, 25, 256)
        with tf.variable_scope("pool2"):
            pool2 = tf.layers.max_pooling2d(conv5_a, pool_size=[2, 2], strides=2, padding="same")

        # (13, 25, 256) -> (13, 25, 512)
        with tf.variable_scope("conv6"):
            conv6_weights = weights_variable_xavier([3, 3, 256, 512], name=CONV6_WEIGHTS)
            conv6_bias = bias_variable([512], value=0.1, name=CONV6_BIAS)
            conv6_z = conv2d(pool2, conv6_weights) + conv6_bias
            conv6_a = tf.nn.relu(conv6_z)

        # (13, 25, 512) -> (13, 25, 512)
        with tf.variable_scope("pool3"):
            pool3 = tf.layers.max_pooling2d(conv6_a, pool_size=[2, 2], strides=1, padding="same")

        # (13, 25, 512) -> (13, 25, 512)
        with tf.variable_scope("conv7"):
            conv7_weights = weights_variable_xavier([3, 3, 512, 512], name=CONV7_WEIGHTS)
            conv7_bias = bias_variable([512], value=0.1, name=CONV7_BIAS)
            conv7_z = conv2d(pool3, conv7_weights) + conv7_bias
            conv7_a = tf.nn.relu(conv7_z)

        # (13, 25, 512) -> (7, 13, 512)
        with tf.variable_scope("pool4"):
            pool4 = tf.layers.max_pooling2d(conv7_a, pool_size=[2, 2], strides=2, padding="same")

        flatten = tf.reshape(pool4, [-1, 7 * 13 * 512])

        with tf.variable_scope("fc0"):
            fc0_weights = weights_variable_truncated_normal([7 * 13 * 512, 1024], stddev=0.005, name=FC0_WEIGHTS)
            fc0_bias = bias_variable([1024], value=0.1, name=FC0_BIAS)
            fc0_z = tf.matmul(flatten, fc0_weights) + fc0_bias
            fc0_a = tf.nn.relu(fc0_z)
            dropout_0 = tf.layers.dropout(fc0_a, rate=self._drop_rate)

        with tf.variable_scope("fc1"):
            fc1_weights = weights_variable_truncated_normal([1024, 2048], stddev=0.005, name=FC1_WEIGHTS)
            fc1_bias = bias_variable([2048], value=0.1, name=FC1_BIAS)
            fc1_z = tf.matmul(dropout_0, fc1_weights) + fc1_bias
            fc1_a = tf.nn.relu(fc1_z)
            dropout_1 = tf.layers.dropout(fc1_a, rate=self._drop_rate)

        # Output layers
        with tf.variable_scope("char0"):
            char0_weights = weights_variable_xavier([2048, self._num_distinct_chars + 1], name=FC_CHAR0_WEIGHTS)
            char0_bias = bias_variable([self._num_distinct_chars + 1], name=FC_CHAR0_BIAS)
            char0_logits = tf.matmul(dropout_1, char0_weights) + char0_bias
            char0_out = tf.nn.softmax(char0_logits)

        with tf.variable_scope("char1"):
            char1_weights = weights_variable_xavier([2048, self._num_distinct_chars + 1], name=FC_CHAR1_WEIGHTS)
            char1_bias = bias_variable([self._num_distinct_chars + 1], name=FC_CHAR1_BIAS)
            char1_logits = tf.matmul(dropout_1, char1_weights) + char1_bias
            char1_out = tf.nn.softmax(char1_logits)

        with tf.variable_scope("char2"):
            char2_weights = weights_variable_xavier([2048, self._num_distinct_chars + 1], name=FC_CHAR2_WEIGHTS)
            char2_bias = bias_variable([self._num_distinct_chars + 1], name=FC_CHAR2_BIAS)
            char2_logits = tf.matmul(dropout_1, char2_weights) + char2_bias
            char2_out = tf.nn.softmax(char2_logits)

        with tf.variable_scope("char3"):
            char3_weights = weights_variable_xavier([2048, self._num_distinct_chars + 1], name=FC_CHAR3_WEIGHTS)
            char3_bias = bias_variable([self._num_distinct_chars + 1], name=FC_CHAR3_BIAS)
            char3_logits = tf.matmul(dropout_1, char3_weights) + char3_bias
            char3_out = tf.nn.softmax(char3_logits)

        with tf.variable_scope("char4"):
            char4_weights = weights_variable_xavier([2048, self._num_distinct_chars + 1], name=FC_CHAR4_WEIGHTS)
            char4_bias = bias_variable([self._num_distinct_chars + 1], name=FC_CHAR4_BIAS)
            char4_logits = tf.matmul(dropout_1, char4_weights) + char4_bias
            char4_out = tf.nn.softmax(char4_logits)

        with tf.variable_scope("char5"):
            char5_weights = weights_variable_xavier([2048, self._num_distinct_chars + 1], name=FC_CHAR5_WEIGHTS)
            char5_bias = bias_variable([self._num_distinct_chars + 1], name=FC_CHAR5_BIAS)
            char5_logits = tf.matmul(dropout_1, char5_weights) + char5_bias
            char5_out = tf.nn.softmax(char5_logits)

        with tf.variable_scope("char6"):
            char6_weights = weights_variable_xavier([2048, self._num_distinct_chars + 1], name=FC_CHAR6_WEIGHTS)
            char6_bias = bias_variable([self._num_distinct_chars + 1], name=FC_CHAR6_BIAS)
            char6_logits = tf.matmul(dropout_1, char6_weights) + char6_bias
            char6_out = tf.nn.softmax(char6_logits)

        # Keep track of weight variables
        self._weight_vars[CONV0_WEIGHTS] = conv0_weights
        self._weight_vars[CONV1_WEIGHTS] = conv1_weights
        self._weight_vars[CONV2_WEIGHTS] = conv2_weights
        self._weight_vars[CONV3_WEIGHTS] = conv3_weights
        self._weight_vars[CONV4_WEIGHTS] = conv4_weights
        self._weight_vars[CONV5_WEIGHTS] = conv5_weights
        self._weight_vars[CONV6_WEIGHTS] = conv6_weights
        self._weight_vars[CONV7_WEIGHTS] = conv7_weights
        self._weight_vars[FC0_WEIGHTS] = fc0_weights
        self._weight_vars[FC1_WEIGHTS] = fc1_weights

        self._weight_vars[CONV0_BIAS] = conv0_bias
        self._weight_vars[CONV1_BIAS] = conv1_bias
        self._weight_vars[CONV2_BIAS] = conv2_bias
        self._weight_vars[CONV3_BIAS] = conv3_bias
        self._weight_vars[CONV4_BIAS] = conv4_bias
        self._weight_vars[CONV5_BIAS] = conv5_bias
        self._weight_vars[CONV6_BIAS] = conv6_bias
        self._weight_vars[CONV7_BIAS] = conv7_bias
        self._weight_vars[FC0_BIAS] = fc0_bias
        self._weight_vars[FC1_BIAS] = fc1_bias

        self._weight_vars[FC_CHAR0_WEIGHTS] = char0_weights
        self._weight_vars[FC_CHAR1_WEIGHTS] = char1_weights
        self._weight_vars[FC_CHAR2_WEIGHTS] = char2_weights
        self._weight_vars[FC_CHAR3_WEIGHTS] = char3_weights
        self._weight_vars[FC_CHAR4_WEIGHTS] = char4_weights
        self._weight_vars[FC_CHAR5_WEIGHTS] = char5_weights
        self._weight_vars[FC_CHAR6_WEIGHTS] = char6_weights

        self._weight_vars[FC_CHAR0_BIAS] = char0_bias
        self._weight_vars[FC_CHAR1_BIAS] = char1_bias
        self._weight_vars[FC_CHAR2_BIAS] = char2_bias
        self._weight_vars[FC_CHAR3_BIAS] = char3_bias
        self._weight_vars[FC_CHAR4_BIAS] = char4_bias
        self._weight_vars[FC_CHAR5_BIAS] = char5_bias
        self._weight_vars[FC_CHAR6_BIAS] = char6_bias

        # Combine output and output logits
        outputs = tf.stack([char0_out, char1_out, char2_out, char3_out, char4_out, char5_out, char6_out], axis=1)
        logits = tf.stack([char0_logits, char1_logits, char2_logits, char3_logits, char4_logits, char5_logits, char6_logits], axis=1)

        return outputs, logits

    def load(self):
        log.debug("Attempting to read checkpoint from {}".format(self._checkpoint_dir))

        checkpoint = tf.train.get_checkpoint_state(self._checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            self._saver.restore(self._sess, checkpoint.model_checkpoint_path)
            log.info("Successfully restored checkpoint")
            return True

        log.info("Failed to restore checkpoint")
        return False

    def store(self, step):
        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)

        path = self._saver.save(self._sess, os.path.join(self._checkpoint_dir, self._model_name), global_step=step)
        log.info("Stored model at step {}".format(step))
        return path

    def variable_summaries(self, var_name):
        """
        Attach a lot of summaries to a Tensor (for TensorBoard visualization).
        :param var_name: name of variable to monitor
        :return:
        """
        var_variable = self._weight_vars[var_name]

        with tf.name_scope("{}_summary".format(var_name)):
            mean = tf.reduce_mean(var_variable)
            mean_summary = tf.summary.scalar("mean", mean)
            with tf.name_scope("stddev"):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var_variable - mean)))
            stddev_summary = tf.summary.scalar("stddev", stddev)
            max_summary = tf.summary.scalar("max", tf.reduce_max(var_variable))
            min_summary = tf.summary.scalar("min", tf.reduce_min(var_variable))
            histogram_summary = tf.summary.histogram("histogram", var_variable)

            return [mean_summary, stddev_summary, max_summary, min_summary, histogram_summary]

    def train(self, path_to_training_set, path_to_validation_set):
        if not os.path.exists(path_to_training_set):
            raise ValueError("Training set could not be found at {}".format(path_to_training_set))
        if not os.path.exists(path_to_validation_set):
            raise ValueError("Validation set could not be found at {}".format(path_to_validation_set))

        # Load the data set
        f = h5py.File(path_to_training_set, "r")
        images = f[DATA_IMAGES]
        char_labels = f[DATA_CHAR_LABELS]
        # Leave open on purpose

        # Tensors cannot exceed 2GB, thus use indirect addressing
        num_total_imgs = len(images)
        data_indices = np.arange(num_total_imgs)
        dataset = tf.data.Dataset.from_tensor_slices(data_indices)
        num_batches = int(np.ceil(num_total_imgs / self._training_batch_size))

        # Divide into batches
        batched_dataset = dataset.batch(self._training_batch_size)
        iterator = batched_dataset.make_initializable_iterator()
        next_element_op = iterator.get_next()

        # Fetch the output units
        char_logits = self._output_logits
        with tf.name_scope("loss"):
            char0_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._char_labels[:, 0], logits=char_logits[:, 0, :]))
            char1_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._char_labels[:, 1], logits=char_logits[:, 1, :]))
            char2_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._char_labels[:, 2], logits=char_logits[:, 2, :]))
            char3_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._char_labels[:, 3], logits=char_logits[:, 3, :]))
            char4_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._char_labels[:, 4], logits=char_logits[:, 4, :]))
            char5_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._char_labels[:, 5], logits=char_logits[:, 5, :]))
            char6_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._char_labels[:, 6], logits=char_logits[:, 6, :]))
            loss = char0_cross_entropy + char1_cross_entropy + char2_cross_entropy + char3_cross_entropy + char4_cross_entropy + char5_cross_entropy + char6_cross_entropy

        # Set up optimizer
        with tf.name_scope("optimizer"):
            global_step = tf.Variable(0, name="global_step", trainable=False)
            learning_rate = tf.train.exponential_decay(1e-2, global_step=global_step, decay_steps=num_batches, decay_rate=0.9, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=global_step)

        # Summary
        loss_summary = tf.summary.scalar("loss", loss)
        learning_rate_summary = tf.summary.scalar("learning_rate", learning_rate)

        # Show statistics for each of our weight variables on TensorBoard
        training_summary_tensors = []
        for var_name in self._weight_vars.keys():
            summary_tensors = self.variable_summaries(var_name)
            training_summary_tensors.extend(summary_tensors)
        summary_op = tf.summary.merge([loss_summary, learning_rate_summary] + training_summary_tensors)

        # Prepare paths for summaries
        summary_dir_name = time.strftime("%Y_%m_%d_%H_%M_%S") + "-" + self._model_name
        training_summary_dir = os.path.join(self._summary_dir, summary_dir_name, "training")
        validation_summary_dir = os.path.join(self._summary_dir, summary_dir_name, "validation")
        # Create two different summary writers to give statistics on training and validation images
        training_summary_writer = tf.summary.FileWriter(training_summary_dir, graph=self._sess.graph)
        # Set up evaluation summary writer without graph to avoid overlap with training graph
        self._eval_summary_writer = tf.summary.FileWriter(validation_summary_dir)

        # Start the training
        self._sess.run(tf.global_variables_initializer())

        # Restore model checkpoint
        if self.load():
            log.info("Restored model")
        else:
            log.info("Initializing new model")

        for var_name, var_variable in self._weight_vars.items():
            # Fill previous eval weights with initialized weight values
            self._weight_vars_previous_eval[var_name] = var_variable.eval()

        patience = self._initial_patience
        best_validation_accuracy = 0.0
        epoch = 0

        self._sess.run(iterator.initializer)
        while patience > 0:
            try:
                # Fetch batch
                batch_data_indices = self._sess.run(next_element_op)
                # Fancy indexing only works with lists of boolean masks, not ndarrays
                batch_data_indices = batch_data_indices.tolist()
                images_batch = images[batch_data_indices]
                char_labels_batch = char_labels[batch_data_indices]

                # Run the training op
                _, loss_val, global_step_val = self._sess.run([train_op, loss, global_step],
                                                              feed_dict={self._images: images_batch,
                                                                         self._char_labels: char_labels_batch,
                                                                         self._drop_rate: 0.5})

                # Display loss every now and then
                if global_step_val % self._num_steps_to_show_loss == 0:
                    log.info("Epoch: {:d}, global step {:d}, loss = {:3.3f}".format(epoch, global_step_val, loss_val))

                # Write summary accuracy on validation after in predefined intervals
                if global_step_val % self._num_steps_to_check == 0:
                    feed_dict = {self._images: images_batch,
                                 self._char_labels: char_labels_batch,
                                 self._drop_rate: 0.0}
                    for var_name, var_variable in self._weight_var_placeholders.items():
                        feed_dict[var_variable] = self._weight_vars_previous_eval[var_name]

                    summary_val = self._sess.run(summary_op, feed_dict=feed_dict)
                    # Write summary
                    training_summary_writer.add_summary(summary_val, global_step=global_step_val)

                    # Evaluate accuracy on validation set
                    store_results_path = os.path.join(validation_summary_dir, "validation_set_results.h5")
                    validation_accuracy = self.evaluate(path_to_validation_set, global_step_val, store_results_path=store_results_path)

                    # Stop training if the  accuracy on the validation set hasn't changed for `patience` steps any more
                    if validation_accuracy > best_validation_accuracy:
                        self.store(global_step_val)
                        # Reset patience
                        patience = self._initial_patience
                        best_validation_accuracy = validation_accuracy
                    else:
                        patience -= 1

                    log.info("Patience: {:d}".format(patience))
            except tf.errors.OutOfRangeError:
                # End of batch reached
                log.debug("Reached end of batch in epoch in epoch {:d}. Reinitializing batch iterator".format(epoch))
                epoch += 1
                self._sess.run(iterator.initializer)

        f.close()
        log.info("Finished training")

    def evaluate(self, path_to_dataset, global_step=0, batch_size=128, store_results_path=None):
        # Load the data set
        f = h5py.File(path_to_dataset, "r")
        images = f[DATA_IMAGES]
        char_labels = f[DATA_CHAR_LABELS]

        # Use indirect addressing for images
        num_total_imgs = len(images)
        data_indices = np.arange(num_total_imgs)
        dataset = tf.data.Dataset.from_tensor_slices(data_indices)
        num_batches = int(np.ceil(num_total_imgs / self._training_batch_size))

        # Divide data set into batches
        batched_dataset = dataset.batch(batch_size)
        iterator = batched_dataset.make_initializable_iterator()
        next_element_op = iterator.get_next()

        # Set up writer to store predictions
        if store_results_path is not None:
            writer = BufferedWriter(store_results_path)

        # Initialize batch iterator
        self._sess.run(iterator.initializer)
        top_k_accuracies = []

        # Loop over batches
        while True:
            try:
                # Fetch batch
                batch_data_indices = self._sess.run(next_element_op)
                # Fancy indexing only works with lists or boolean masks, not ndarrays
                batch_data_indices = batch_data_indices.tolist()
                images_batch = images[batch_data_indices]
                char_labels_batch = char_labels[batch_data_indices]

                feed_dict = {self._images: images_batch,
                             self._char_labels: char_labels_batch,
                             self._drop_rate: 0.0}

                batch_char_probabilities, batch_top_k_accuracies = self._sess.run([self._output, self._char_top_k_accuracies_samplewise], feed_dict=feed_dict)
                top_k_accuracies.append(batch_top_k_accuracies)

                if store_results_path is not None:
                    # Copy all data sets from input to output file
                    batch = {}
                    for key in list(f.keys()):
                        # Determine if data set is list or array
                        if len(f[key].shape) == 1:
                            # We're dealing with a list
                            data = list(f[key][batch_data_indices])
                            batch[key] = data
                        else:
                            # We're dealing with an array
                            data = np.array(f[key][batch_data_indices])
                            batch[key] = data
                    batch[DATA_CHAR_PROBABILITIES] = batch_char_probabilities
                    writer.write(batch)

            except tf.errors.OutOfRangeError:
                top_k_accuracy = np.mean(np.concatenate(top_k_accuracies))
                summary = tf.Summary()
                summary.value.add(tag="accuracy", simple_value=top_k_accuracy)

                # Write evaluation summary if evaluation summary writer was set up (which it is during training)
                if self._eval_summary_writer:
                    self._eval_summary_writer.add_summary(summary, global_step=global_step)

                break

        if store_results_path is not None:
            writer.flush()
            log.info("Dumped evaluated probabilities to {}".format(store_results_path))

        log.debug("Finished evaluation")

        # Eventually close data set
        f.close()

        return top_k_accuracy

    def inference(self, img_batch):
        # Get output chars
        feed_dict = {self._images: img_batch, self._drop_rate: 0.0}
        char_probabilities = self._sess.run(self._output, feed_dict=feed_dict)

        return char_probabilities
