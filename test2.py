import tensorflow as tf

import tensorflow_probability as tfp
import numpy as np

pi = tf.constant(np.pi, dtype=tf.float32)


def Normal_prob(x):
    prob = 1 / tf.sqrt(2 * pi) * tf.exp(-0.5 * x ** 2)

    return prob


norm = tfp.distributions.Normal(0, 1)


std = tf.Variable(0.5, dtype=tf.float32, trainable=True, name='std')

with tf.GradientTape() as tape:
    tape.watch(std)

    prob = norm.prob(0.5/std)

grad = tape.gradient(prob, std)

print(grad)

with tf.GradientTape() as tape:
    tape.watch(std)

    prob = Normal_prob(0.5/std)

grad = tape.gradient(prob, std)

print(grad)
