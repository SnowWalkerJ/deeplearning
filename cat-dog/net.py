import tensorflow as tf
import tf_learn as learn
import tf_learn.models, tf_learn.models.dnn, tf_learn.layers


class Model(tf_learn.models.dnn.DNN):
    def build_net(self):
        self.input_tensor = tf.placeholder(tf.int8, [None, 300, 300, 3], name="input")
        cast_float = (tf.cast(self.input_tensor, tf.float32) - 128.0) / 128
        conv1 = tf_learn.layers.conv2d(cast_float, depth=16, filter_size=3, strides=1, activation='relu')
        pool1 = tf.nn.max_pool(conv1, [1, 3, 3, 1], [1, 3, 3, 1], padding='SAME')
        conv2 = tf_learn.layers.conv2d(pool1, depth=32, filter_size=3, strides=1, activation='relu')
        pool2 = tf.nn.max_pool(conv2, [1, 3, 3, 1], [1, 3, 3, 1], padding='SAME')
        conv3 = tf_learn.layers.conv2d(pool2, depth=64, filter_size=3, strides=1, activation='relu')
        pool3 = tf.nn.max_pool(conv3, [1, 3, 3, 1], [1, 3, 3, 1], padding='SAME')
        conv4 = tf_learn.layers.conv2d(pool3, depth=128, filter_size=3, strides=1, activation='relu')
        pool4 = tf.nn.max_pool(conv4, [1, 3, 3, 1], [1, 3, 3, 1], padding='SAME')
        lrn = tf.nn.local_response_normalization(pool4, name='local_response_normalization')
        flattened = tf_learn.layers.flatten(lrn, name='flatten')
        fc1 = tf_learn.layers.fully_connection(flattened, 1024, activation='tanh', name='fc1')
        keep_prob = self.register_placeholder('keep_prob', shape=None, dtype=tf.float32)
        dropped1 = tf.nn.dropout(fc1, keep_prob, name='dropout1')
        fc2 = tf_learn.layers.fully_connection(dropped1, 4096, activation='tanh', name='fc2')
        self.output_tensor = tf_learn.layers.fully_connection(fc2, 2, activation='linear', name='output_tensor')
        self.target_tensor = tf.placeholder(tf.int32, [None], name='target_tensor')
        self.one_hot_labels = tf.one_hot(self.target_tensor, 2, name='one_hot_labels')
        self.loss = tf.nn.softmax_cross_entropy_with_logits(self.output_tensor, self.one_hot_labels, name='cross_entropy')
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        acc = tf.reduce_mean(tf.cast(tf.equal(self.target_tensor, tf.cast(tf.argmax(self.output_tensor, 1), tf.int32)), tf.float32), name='accuracy')
        self.evaluation_dict = {
            'loss': self.loss,
            'acc': acc,
        }
        tf.scalar_summary('accuracy', acc)
        tf.scalar_summary('loss', self.loss)
        self.summary = tf.merge_all_summaries()



