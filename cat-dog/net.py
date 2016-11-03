import tensorflow as tf
import tf_learn.models
import tf_learn.models.dnn
import tf_learn.layers


class Model(tf_learn.models.dnn.DNN):

    def build_net(self):
        self.placeholders = {
            'keep_prob': {
                'train': 0.5,
                'evaluate': 1.0,
            },
        }
        self.global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(0.005, self.global_step, 1500, 0.96, staircase=True)
        tf.scalar_summary("learning rate", lr)
        keep_prob = self.register_placeholder("keep_prob", None, tf.float32)

        self.input_tensor = tf.placeholder(tf.int8, [None, 300, 300, 3], name="input")
        with tf.name_scope('normalize'):
            net = (tf.to_float(self.input_tensor) - 128) / 128.0
        with tf.name_scope("conv1"):
            net1 = tf_learn.layers.conv2d(net, depth=16, filter_size=3, strides=1, activation='relu')
            tf.histogram_summary("conv1", net1.W)
            net1 = tf_learn.layers.conv2d(net1, depth=16, filter_size=1, strides=1, activation='relu')
            net1 = tf.nn.local_response_normalization(net1)
            net1 = tf.concat(3, [net, net1])
            net1 = tf.nn.max_pool(net1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        with tf.name_scope("conv2"):
            net2 = tf_learn.layers.conv2d(net1, depth=32, filter_size=3, strides=1, activation='relu')
            tf.histogram_summary("conv2", net2.W)
            net2 = tf_learn.layers.conv2d(net2, depth=32, filter_size=1, strides=1, activation='relu')
            net2 = tf.nn.local_response_normalization(net2)
            net2 = tf.concat(3, [net1, net2])
            net2 = tf.nn.max_pool(net2, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        with tf.name_scope("conv3"):
            net3 = tf_learn.layers.conv2d(net2, depth=64, filter_size=3, strides=1, activation='relu')
            tf.histogram_summary("conv3", net3.W)
            net3 = tf_learn.layers.conv2d(net3, depth=64, filter_size=1, strides=1, activation='relu')
            net3 = tf.nn.local_response_normalization(net3)
            net3 = tf.concat(3, [net2, net3])
            net3 = tf.nn.max_pool(net3, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        with tf.name_scope("conv4"):
            net4 = tf_learn.layers.conv2d(net3, depth=128, filter_size=3, strides=1, activation='relu')
            tf.histogram_summary("conv4", net4.W)
            net4 = tf_learn.layers.conv2d(net4, depth=128, filter_size=1, strides=1, activation='relu')
            net4 = tf.nn.local_response_normalization(net4)
            net4 = tf.concat(3, [net3, net4])
            net4 = tf.nn.max_pool(net4, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

        net = tf_learn.layers.flatten(net4)
        net = tf_learn.layers.fully_connection(net, 512, 'tanh')
        net = tf.dropout(net, keep_prob=keep_prob)
        net = tf_learn.layers.fully_connection(net, 2048, 'tanh')
        net = tf.droupout(net, keep_prob)
        self.output_tensor = tf_learn.layers.fully_connection(net, 2, activation='linear')

        self.target_tensor = tf.placeholder(tf.uint8, [None])
        with tf.name_scope("one_hot"):
            one_hot_label = tf.one_hot(self.target_tensor, 2, dtype=tf.float32)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output_tensor, one_hot_label), name="loss")
            acc = tf.reduce_mean(tf.to_float(tf.equal(self.target_tensor, tf.argmax(self.output_tensor))))
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

        tf.scalar_summary('accuracy', acc)
        tf.scalar_summary('loss', self.loss)
        self.summary = tf.merge_all_summaries()

    def on_train_finish_epoch(self):
        if self.epoch % 4:
            self.placeholders['lr'] *= 0.9
        step = self.sess.run(tf.assign_add(self.global_step, 1))
        self.run_summary(step)

    def on_before_train(self):
        self.run_summary(0)



