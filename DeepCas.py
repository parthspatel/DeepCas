from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import os


class ConvBlock():

    def __init__(self, input_tensor, conv_param, pool_param, inner_activation, last_activation, use_batch_norm, is_training, trainable, padding='same'):
        self.input_tensor = input_tensor
        self.conv_param = conv_param
        self.pool_param = pool_param
        self.inner_activation = inner_activation
        self.last_activation = last_activation
        self.use_batch_norm = use_batch_norm
        self.is_training = is_training
        self.trainable = trainable
        self.padding = padding

    def build(self):
        for layer_index in range(len(self.conv_param)):

            conv_layer = self.conv_param[layer_index]
            pool_layer = self.pool_param[layer_index]

            if layer_index is 0:
                network = self.input_tensor

            activation = self.inner_activation
            use_batch_norm = self.use_batch_norm and True
            if layer_index is len(self.conv_param) - 1:
                activation = self.last_activation
                use_batch_norm = False

            network = tf.layers.conv2d(inputs=network,
                                       filters=conv_layer['filters'],
                                       kernel_size=conv_layer['kernel_size'],
                                       strides=conv_layer['strides'],
                                       padding=self.padding,
                                       activation=tf.nn.relu,
                                       use_bias=True,
                                       kernel_initializer=tf.keras.initializers.he_uniform(),
                                       bias_initializer=tf.zeros_initializer(),
                                       trainable=True)

            if use_batch_norm:
                network = tf.layers.batch_normalization(inputs=network,
                                                        training=self.is_training,
                                                        trainable=self.trainable,
                                                        scale=True)

            if activation is not None:
                network = activation(network)

            if pool_layer is not None:
                network = tf.layers.max_pooling2d(inputs=network,
                                                  pool_size=pool_layer['pool_size'],
                                                  strides=pool_layer['strides'],
                                                  padding='same')
        return network


class DeepCas():

    def __init__(self, sess, data_shape, batch_size, epochs, learning_rate, conv_parameters, max_pool_parameters, save_directory):
        self.sess = sess
        self.data_shape = data_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.conv_parameters = conv_parameters
        self.max_pool_parameters = max_pool_parameters
        self.save_directory = save_directory

        self.initModel()

    def input_data(self, data, labels):
        assert type(data).__module__ == np.__name__
        assert type(labels).__module__ == np.__name__

        self.data = data
        self.labels = labels

    def initModel(self):

        self.x = tf.placeholder(tf.float32,
                                [None,
                                 self.data_shape[0],
                                 self.data_shape[1],
                                 self.data_shape[2]])

        self.y = tf.placeholder(tf.float32, [None, 1])
        self.is_training = tf.placeholder(dtype=tf.bool, shape=None)

        net = self.x
        print('Input Tensor: {}'.format(str(list(net.get_shape())).rjust(10, ' ')))
        layer_index = 1
        for conv_param, pool_param in zip(self.conv_parameters, self.max_pool_parameters):
            net = ConvBlock(input_tensor=net,
                            conv_param=conv_param,
                            pool_param=pool_param,
                            inner_activation=tf.nn.relu,
                            last_activation=tf.nn.relu,
                            use_batch_norm=True,
                            is_training=self.is_training,
                            trainable=True).build()
            print('Layer {}: {}'.format(str(layer_index).rjust(3, ' '),
                                        str(list(net.get_shape())).rjust(10, ' ')))
            layer_index += 1

        net = tf.layers.flatten(net)
        net = tf.layers.dense(inputs=net,
                              units=160,  # tune
                              activation=tf.nn.relu,
                              use_bias=True,
                              kernel_initializer=tf.keras.initializers.he_uniform(),
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=None,
                              bias_regularizer=None,
                              trainable=True,
                              name='fc_1')
        print('Fully Connected 1: {}'.format(str(list(net.get_shape())).rjust(10, ' ')))
        net = tf.layers.batch_normalization(inputs=net,
                                            training=self.is_training,
                                            trainable=True,
                                            name='bn_fc_1',
                                            scale=True)

        net = tf.nn.relu(net)
        net = tf.layers.dense(inputs=net,
                              units=160,  # tune
                              activation=tf.nn.relu,
                              use_bias=True,
                              kernel_initializer=tf.keras.initializers.he_uniform(),
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=None,
                              bias_regularizer=None,
                              trainable=True,
                              name='fc_2')
        print('Fully Connected 2: {}'.format(str(list(net.get_shape())).rjust(10, ' ')))
        net = tf.layers.batch_normalization(inputs=net,
                                            training=self.is_training,
                                            trainable=True,
                                            name='bn_fc_2',
                                            scale=True)

        net = tf.nn.relu(net)
        net = tf.layers.dense(inputs=net,
                              units=1,
                              activation=None,
                              use_bias=True,
                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                              bias_initializer=tf.zeros_initializer(),
                              kernel_regularizer=None,
                              bias_regularizer=None,
                              trainable=True,
                              name='fc_3')
        print('Fully Connected 3: {}'.format(str(list(net.get_shape())).rjust(10, ' ')))

        # Results
        # ---------------------------------------------_------------------------
        # Loss Calculation
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.y,
                                                    logits=net)

        # Accuracy Calculation
        self.predicted_indicies = tf.argmax(input=net,
                                            axis=0)
        self.real_indices = tf.argmax(input=self.y,
                                      axis=0)
        self.accuracy = tf.cast(tf.equal(self.predicted_indicies, self.real_indices),
                                dtype=tf.float32)
        self.accuracy = tf.reduce_mean(self.accuracy)

        # Variable to dump accuracy
        self.accuracy_output = tf.placeholder(dtype=tf.float32, shape=None)

        # TensorBoard Summary
        self.loss_summary = tf.summary.scalar(name='Loss',
                                              tensor=self.loss)
        self.accuracy_summary = tf.summary.scalar(name='Accuracy',
                                                  tensor=self.accuracy)
        self.merged_summaries = tf.summary.merge(inputs=[self.loss_summary, self.accuracy_summary])

        self.accuracy_output_summary = tf.summary.scalar(name='Val Accuracy',
                                                         tensor=self.accuracy_output)

        self.val_summary = tf.summary.scalar(name='Val Accuracy',
                                             tensor=self.accuracy_output)

    def train_init(self):

        model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss,
                                                               var_list=model_variables)
        self.sess.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))

    def train(self, tensorboard_directory):

        tf.logging.set_verbosity(tf.logging.INFO)

        # init_op = tf.global_variables_initializer()

        self.train_init()

        # TensorBoard & Saver Init
        if not os.path.exists(tensorboard_directory):
            os.makedirs(tensorboard_directory)
        train_writer, val_writer = [tf.summary.FileWriter(os.path.join(tensorboard_directory, phase),
                                                          self.sess.graph) for phase in ['train', 'val']]
        save_file = os.path.join(self.save_directory, 'model')

        # self.sess.run(init_op)
        num_batches = int(len(self.labels) / self.batch_size)

        batch_loss, batch_accuracy = 0, 0
        for epoch in range(1, self.epochs+1):
            step = 1
            for _ in range(num_batches):
                batch_x, batch_y = self.next_batch(self.batch_size, self.data, self.labels)
                batch_y = batch_y[:, None]

                # print('Batch x: {}'.format(str(list(batch_x.shape)).rjust(10, ' ')))
                # print('Batch y: {}'.format(str(list(batch_y.shape)).rjust(10, ' ')))

                loss, summary, _, = self.sess.run([self.loss, self.merged_summaries, self.optimizer],
                                                  feed_dict={self.is_training: True,
                                                             self.x: batch_x,
                                                             self.y: batch_y})
                if step % 100 is 0:
                    # Output Loss to Terminal, Summary to TensorBoard
                    print("Step: {} Loss: {}".format(step, loss))
                    # train_writer.add_summary(summary, step)

                step += 1

            # TODO: Test Accuracy in multiple batches, none of this 1 batch crap
            epoch_x, epoch_y = epoch_x, epoch_y = self.next_batch(self.batch_size, self.data, self.labels)
            epoch_y = epoch_y[:, None]
            loss, accuracy = self.sess.run([self.loss, self.accuracy],
                                           feed_dict={self.is_training: True,
                                                      self.x: batch_x,
                                                      self.y: batch_y})
            batch_loss += loss
            batch_accuracy += accuracy
            if epoch % 100 is 0:
                print("Avgs:\tEpoch: {}\tLoss: {}\tAccuracy {}".format(
                    str(epoch).rjust(100000), str(batch_loss/100).rjust(100000), str(batch_accuracy/100).rjust(100000)))
            # val_summary = self.sess.run(self.val_summary,
            #                             feed_dict={self.accuracy_output: accuracy})
            # val_writer.add_summary(val_summary, step)

    def next_batch(self, batch_size, data, labels):

        idx = np.arange(0, len(labels))
        np.random.shuffle(idx)

        idx = idx[:batch_size]
        data_shuffled = [data[i] for i in idx]
        labels_shuffled = [labels[i] for i in idx]

        return np.asarray(data_shuffled), np.asarray(labels_shuffled)
