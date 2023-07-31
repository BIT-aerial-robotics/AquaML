import tensorflow as tf
import numpy as np


# CNN+LSTM 测试 多模态输入
class CNNLSTM(tf.keras.Model):
    def __init__(self):
        super(CNNLSTM, self).__init__()

        # 创建encoder

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=8,
                    kernel_size=7,
                    input_shape=(64, 64, 1),
                    activation='relu',
                ),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='pool1'),
                tf.keras.layers.Conv2D(filters=16, kernel_size=4, activation='relu', name="conv2", ),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="pool2"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, name="IMG_layer1"),
                tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Dense(64, name="IMG_layer2"),
                tf.keras.layers.LeakyReLU(0.2),
            ]
        )

        # self.reshape_layer = tf.keras.layers.Reshape((1, 64))

        self.lstm = tf.keras.layers.LSTM(256, return_sequences=True, return_state=True, name="LSTM_layer")

        self.dense = tf.keras.layers.Dense(128, name="dense_layer")

        self.dense2 = tf.keras.layers.Dense(64, name="dense_layer2")

        self.action_layer = tf.keras.layers.Dense(4, name="action_layer")
    @tf.function
    def call(self, img, obs, hidden1, hidden2, mask=None):
        img = self.encoder(img)
        # img = self.reshape_layer(img)
        # img = tf.expand_dims(img, axis=1)

        img = tf.reshape(img, (obs.shape[0], obs.shape[1], 64))

        x = tf.concat([img, obs], axis=-1)

        x, hidden1, hidden2 = self.lstm(x, initial_state=[hidden1, hidden2], mask=mask)

        x = self.dense(x)
        x = self.dense2(x)
        action = self.action_layer(x)

        return action, hidden1, hidden2


pad_sequences = tf.keras.utils.pad_sequences

img = tf.random.normal((16, 64, 64, 1))

obs = tf.random.normal((2, 8, 6))

hidden1 = tf.random.normal((2, 256))
hidden2 = tf.random.normal((2, 256))

model = CNNLSTM()

action, hidden1, hidden2 = model(img, obs, hidden1, hidden2)

target = tf.random.normal((2, 8, 4))

seq_obs1 = tf.random.normal((8, 6))
seq_obs2 = tf.random.normal((6, 6))
seq_obs3 = tf.random.normal((4, 6))

seq_obs = [seq_obs1, seq_obs2, seq_obs3]

pad_seq_obs = pad_sequences(seq_obs, padding='post', dtype='float32')

seq_img1 = tf.random.normal((8, 64, 64, 1))
seq_img2 = tf.random.normal((6, 64, 64, 1))
seq_img3 = tf.random.normal((4, 64, 64, 1))

seq_img = [seq_img1, seq_img2, seq_img3]

pad_seq_img_ = pad_sequences(seq_img, padding='post', dtype='float32')
pad_seq_img = tf.reshape(pad_seq_img_, (-1, 64, 64, 1))

seq_bool_mask1 = tf.ones((8, 1), dtype=tf.bool)
seq_bool_mask2 = tf.ones((6, 1), dtype=tf.bool)
seq_bool_mask3 = tf.ones((4, 1), dtype=tf.bool)

seq_bool_mask = [seq_bool_mask1, seq_bool_mask2, seq_bool_mask3]

pad_seq_bool_mask = pad_sequences(seq_bool_mask, padding='post', dtype='bool')
pad_seq_bool_mask = tf.squeeze(pad_seq_bool_mask, axis=-1)

seq_hidden1 = tf.random.normal((3, 256))
seq_hidden2 = tf.random.normal((3, 256))

target = tf.random.normal((3, 8, 4))

with tf.GradientTape() as tape:
    action, hidden1, hidden2 = model(pad_seq_img, pad_seq_obs, seq_hidden1, seq_hidden2, mask=pad_seq_bool_mask)

    value = tf.square(action - target)
    mask_value = tf.boolean_mask(value, pad_seq_bool_mask)
    loss = tf.reduce_mean(tf.square(mask_value))

grads = tape.gradient(loss, model.trainable_variables)

print(grads)
