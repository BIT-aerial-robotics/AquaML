import tensorflow as tf
import numpy as np

a = np.array([1, 2, 3, 4, 5, 6]).reshape((-1, 2))
b = tf.reduce_sum(a, axis=1, keepdims=True)

print(b.numpy())