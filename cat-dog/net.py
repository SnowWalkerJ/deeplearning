import tensorflow as tf
import tf_learn.models
import tf_learn.models.dnn
import tf_learn.layers


class Model(tf_learn.models.dnn.DNN):
    def build_net(self):
        self.placeholders = {
            'keep_prob': {
                'train': 0.7,
                'evaluate': 1.0,
            },
            'lr': 0.001
        }
        keep_prob = self.register_placeholder('keep_prob', shape=None, dtype=tf.float32)
        lr = self.register_placeholder('lr', shape=None, dtype=tf.float32)

        self.input_tensor = tf.placeholder(tf.int8, [None, 300, 300, 3], name="input")
        with tf.name_scope('normalization'):
            net = (tf.cast(self.input_tensor, tf.float32) - 128.0) / 128.0
        with tf.name_scope("conv1"):
            net1 = tf_learn.layers.conv2d(net, depth=8, filter_size=1, strides=1, activation='relu')
            net1 = tf_learn.layers.conv2d(net1, depth=16, filter_size=3, strides=1, activation='relu')
            tf.histogram_summary('conv1.weight', net1.W)
            net1 = tf.nn.local_response_normalization(net1)
            net1 = tf.concat(3, [net, net1])
            net1 = tf.nn.max_pool(net1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        with tf.name_scope("conv2"):
            net2 = tf_learn.layers.conv2d(net1, depth=24, filter_size=1, strides=1, activation='relu')
            net2 = tf_learn.layers.conv2d(net2, depth=32, filter_size=3, strides=1, activation='relu')
            tf.histogram_summary('conv2.weight', net2.W)
            net2 = tf.nn.local_response_normalization(net2)
            net2 = tf.concat(3, [net1, net2])
            net2 = tf.nn.max_pool(net2, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        with tf.name_scope("conv3"):
            net3 = tf_learn.layers.conv2d(net2, depth=64, filter_size=1, strides=1, activation='relu')
            net3 = tf_learn.layers.conv2d(net3, depth=64, filter_size=3, strides=1, activation='relu')
            tf.histogram_summary('conv3.weight', net3.W)
            net3 = tf.nn.local_response_normalization(net3)
            net3 = tf.concat(3, [net2, net3])
            net3 = tf.nn.max_pool(net3, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        with tf.name_scope("conv4"):
            net4 = tf_learn.layers.conv2d(net3, depth=128, filter_size=1, strides=1, activation='relu')
            net4 = tf_learn.layers.conv2d(net4, depth=128, filter_size=3, strides=1, activation='relu')
            tf.histogram_summary('conv4.weight', net4.W)
            net4 = tf.nn.local_response_normalization(net4)
            net4 = tf.concat(3, [net3, net4])
            net4 = tf.nn.max_pool(net4, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        with tf.name_scope("conv5"):
            net5 = tf_learn.layers.conv2d(net4, depth=256, filter_size=1, strides=1, activation='relu')
            net5 = tf_learn.layers.conv2d(net5, depth=256, filter_size=3, strides=1, activation='relu')
            tf.histogram_summary('conv5.weight', net5.W)
            net5 = tf.nn.local_response_normalization(net5)
            net5 = tf.concat(3, [net5, net4])
            net5 = tf.nn.max_pool(net5, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

        flattened = tf_learn.layers.flatten(net5, name='flatten')
        dropped0 = tf.nn.dropout(flattened, keep_prob)
        fc1 = tf_learn.layers.fully_connection(dropped0, 1024, activation='tanh', name='fc1')
        dropped1 = tf.nn.dropout(fc1, keep_prob, name='dropout1')
        fc2 = tf_learn.layers.fully_connection(dropped1, 1024 * 2, activation='tanh', name='fc2')
        dropped2 = tf.nn.dropout(fc2, keep_prob, name='dropout2')
        self.output_tensor = tf_learn.layers.fully_connection(dropped2, 2, activation='linear', name='output_tensor')
        
        self.target_tensor = tf.placeholder(tf.int32, [None], name='target_tensor')
        with tf.name_scope('one_hot'):
            self.one_hot_labels = tf.one_hot(self.target_tensor, 2, name='one_hot_labels')
        with tf.name_scope('loss'):
            # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output_tensor, self.one_hot_labels, name='cross_entropy'))
            self.loss = tf.reduce_mean(-tf.reduce_sum(tf.log(tf.nn.softmax(self.output_tensor)) * self.one_hot_labels, reduction_indices=[1]))
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
        with tf.name_scope('accuracy'):
            acc = tf.reduce_mean(tf.cast(tf.equal(self.target_tensor,
                                                  tf.cast(tf.argmax(self.output_tensor, 1), tf.int32)),
                                         tf.float32),
                                 name='accuracy')
        self.evaluation_dict = {
            'loss': self.loss,
            'acc': acc,
        }
        tf.scalar_summary('accuracy', acc)
        tf.scalar_summary('loss', self.loss)
        self.summary = tf.merge_all_summaries()

    def on_train_finish_epoch(self):
        if self.epoch % 5 == 0:
            self.placeholders['lr'] *= 0.7
        self.run_summary(self.epoch + 1)

    def on_before_train(self):
	    self.run_summary(0)



