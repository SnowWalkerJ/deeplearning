import tensorflow as tf
import tf_learn.models
import tf_learn.models.dnn
import tf_learn.layers


class Model(tf_learn.models.dnn.DNN):
    def build_net(self):
        self.placeholders = {
            'keep_prob': {
                'train': 0.8,
                'evaluate': 1.0,
            },
            'lr': 0.001
        }
        keep_prob = self.register_placeholder('keep_prob', shape=None, dtype=tf.float32)
        lr = self.register_placeholder('lr', shape=None, dtype=tf.float32)

        self.input_tensor = tf.placeholder(tf.int8, [None, 300, 300, 3], name="input")
        with tf.name_scope('normalization'):
            cast_float = (tf.cast(self.input_tensor, tf.float32) - 128.0) / 128.0
        with tf.name_scope('layer1'):
            conv1 = tf_learn.layers.conv2d(cast_float, depth=4, filter_size=3, strides=1, activation='relu', name='3x3x4')
            conv11 = tf_learn.layers.conv2d(conv1, depth=8, filter_size=1, strides=1, activation='relu', name='1x1x8')
            lrn1 = tf.nn.local_response_normalization(conv11, name='local_response_normalization1')
            stack1 = tf.concat(3, [cast_float, lrn1], name='stack1')

        with tf.name_scope('layer1.5'):
            conv15 = tf_learn.layers.conv2d(stack1, depth=12, filter_size=3, strides=1, activation='relu', name='3x3x12')
            conv115 = tf_learn.layers.conv2d(conv15, depth=16, filter_size=1, strides=1, activation='relu', name='1x1x16')
            lrn15 = tf.nn.local_response_normalization(conv115, name='local_response_normalization15')
            pool1 = tf.nn.max_pool(lrn15, [1, 3, 3, 1], [1, 3, 3, 1], padding='SAME')

        with tf.name_scope('layer2'):
            conv2 = tf_learn.layers.conv2d(pool1, depth=16, filter_size=3, strides=1, activation='relu', name='3x3x16')
            conv21 = tf_learn.layers.conv2d(conv2, depth=32, filter_size=1, strides=1, activation='relu', name='1x1x32')
            lrn2 = tf.nn.local_response_normalization(conv21, name='local_response_normalization2')
            stack2 = tf.concat(3, [pool1, lrn2], name='stack2')

        with tf.name_scope('layer2.5'):
            # conv25 = tf_learn.layers.conv2d(stack2, depth=48, filter_size=3, strides=1, activation='relu', name='3x3x48')
            conv251 = tf_learn.layers.conv2d(stack2, depth=64, filter_size=1, strides=1, activation='relu', name='1x1x64')
            lrn25 = tf.nn.local_response_normalization(conv251)
            pool2 = tf.nn.max_pool(lrn25, [1, 3, 3, 1], [1, 3, 3, 1], padding='SAME')

        with tf.name_scope('layer3'):
            conv3 = tf_learn.layers.conv2d(pool2, depth=64, filter_size=3, strides=1, activation='relu', name='3x3x64')
            lrn3 = tf.nn.local_response_normalization(conv3, name='local_response_normalization3')
            conv31 = tf_learn.layers.conv2d(lrn3, depth=128, filter_size=1, strides=1, activation='relu', name='1x1x128')
            stack3 = tf.concat(3, [pool2, conv31], name='stack3')
            pool3 = tf.nn.max_pool(stack3, [1, 3, 3, 1], [1, 3, 3, 1], padding='SAME')
        
        with tf.name_scope('layer4'):
            conv4 = tf_learn.layers.conv2d(pool3, depth=256, filter_size=3, strides=1, activation='relu', name='3x3x256')
            lrn4 = tf.nn.local_response_normalization(conv4, name='local_response_normalization4')
            conv41 = tf_learn.layers.conv2d(lrn4, depth=512, filter_size=1, strides=1, activation='relu', name='1x1x512')
            stack4 = tf.concat(3, [pool3, conv41], name='stack4')
            pool4 = tf.nn.max_pool(stack4, [1, 3, 3, 1], [1, 3, 3, 1], padding='SAME')
        flattened = tf_learn.layers.flatten(pool4, name='flatten')
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
        tf.histogram_summary('fc1_weight', fc1.W)
        tf.histogram_summary('fc2_weight', fc2.W)
        tf.histogram_summary('conv1_weight', conv1.W)
        tf.histogram_summary('conv2_weight', conv2.W)
        tf.histogram_summary('conv3_weight', conv3.W)
        tf.histogram_summary('conv4.weight', conv4.W)
        self.summary = tf.merge_all_summaries()

    def on_train_finish_epoch(self):
        if self.epoch % 5 == 0:
            self.placeholders['lr'] *= 0.7
        self.run_summary(self.epoch + 1)

    def on_before_train(self):
	    self.run_summary(0)



