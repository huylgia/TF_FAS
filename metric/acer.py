# customize metrics
from keras.metrics import Metric
import tensorflow as tf

class ACER(Metric):
    def __init__(self, name='acer', **kwargs):
        super(ACER, self).__init__(name=name, **kwargs)
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
        self.n = self.add_weight(name='n', initializer='zeros')
        self.p = self.add_weight(name='p', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_pred should be similarity scores
        if y_pred.shape[-1] == 1:
            y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
        else:
            y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)

        # y_true should be labels indicating whether the pairs of face images match or not
        pos = tf.math.equal(y_true, 1)
        neg = tf.math.equal(y_true, 0)

        fn = tf.equal(tf.gather(y_true, tf.where(pos)[:, 0]), tf.gather(y_pred, tf.where(pos)[:, 0]))
        fn = tf.reduce_sum(1 - tf.cast(fn, tf.float32))

        fp = tf.equal(tf.gather(y_true, tf.where(neg)[:, 0]), tf.gather(y_pred, tf.where(neg)[:, 0]))
        fp = tf.reduce_sum(1 - tf.cast(fp, tf.float32))        
        
        n = tf.reduce_sum(tf.cast(neg, tf.float32))
        p = tf.reduce_sum(tf.cast(pos, tf.float32))

        self.fp.assign_add(fp)
        self.fn.assign_add(fn)
        self.n.assign_add(n)
        self.p.assign_add(p)

    def result(self):
        apcer = self.fp / self.n
        bpcer = self.fn / self.p
        acer = (apcer + bpcer) / 2

        return eval(self.name)
        
    def reset_state(self):
        self.fp.assign(0)
        self.fn.assign(0)
        self.n.assign(0)
        self.p.assign(0)