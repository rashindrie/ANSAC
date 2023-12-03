from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow.compat.v1 as tf


if __name__ == "__main__":
    DOWNLOADED_TF_CHECKPOINT_WEIGHTS_PATH = sys.argv[1]

    # ### Create VGG model
    vgg = _BuildGraph()

    # ### Extract tf model files
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        labels = tf.squeeze(vgg.pred_up)
        writer = tf.summary.FileWriter('./tf_model/graphs', sess.graph)
        writer.close()

        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, DOWNLOADED_TF_CHECKPOINT_WEIGHTS_PATH + 'model_epoch99.75_.ckpt-75000')
        save_path = saver.save(sess, "./tf_model/crowdsourcing_fcn8vgg16.ckpt")
        print("Model saved in file: %s" % save_path)
