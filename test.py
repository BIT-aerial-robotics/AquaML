import tensorflow as tf

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

optimizer.learning_rate = 0.01

print(optimizer.learning_rate)