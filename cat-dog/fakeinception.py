import tensorflow as tf
import tf_learn.models
import tf_learn.models.dnn
import tf_learn.layers


class Model(tf_learn.models.dnn.DNN):
    @staticmethod
    def inception(input_tensor, name):
        with tf.name_scope(name):
            base_depth = input_tensor.get_shape().as_list()[-1]
            small = tf_learn.layers.conv2d(input_tensor, depth=base_depth, filter_size=1, activation='relu', strides=[1, 2, 2, 1])
            small_pool = tf_learn.layers.conv2d(input_tensor, depth=base_depth, filter_size=1, activation='relu', strides=1)
            small_pool = tf.nn.max_pool(small_pool, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            middle = tf_learn.layers.conv2d(input_tensor, depth=base_depth, filter_size=1, activation='relu', strides=1)
            middle = tf_learn.layers.conv2d(middle, depth=base_depth, filter_size=3, strides=[1, 2, 2, 1], activation='relu')
            stacked = tf.concat(3, [small, small_pool, middle])
            stacked = tf.nn.local_response_normalization(stacked)
        return stacked

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

        net = self.inception(net, 'layer1')
        net = self.inception(net, 'layer2')
        net = self.inception(net, 'layer3')
        net = self.inception(net, 'layer4')
        net = self.inception(net, 'layer5')

        flattened = tf_learn.layers.flatten(net, name='flatten')
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



