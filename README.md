# AquaML

## Installation

```bash
conda create -n AquaML python=3.8
conda activate AquaML
conda install tensorflow-gpu
pip install tensorflow-probability==0.12.2
pip install mpi4py
pip install gym
pip install keras-core
```

## Usage

### Create a self defined model

You do not need to concentrate on how the model will be used in the framework. You just need to define a model class which inherits from `Model` and implement the `call` method. Some algorithms may need to implement special functions. 


Now we give an example of a simple model. The model is used in reinforcement learning proximal policy gradient. The input of the model is image with shape (64,64,1) and a tensor with shape (6,). And we will use CNN to extract features from the image and concatenate the features with the tensor. Then we will use a LSTM layer to extract the temporal features. Finally, we will use a dense layer to output the action.

The input of `call` must contain `mask=None` even though you do not use it. 

```python
import tensorflow as tf

class Actor_net(tf.keras.Model):
    def __init__(self, ):
        super(Actor_net, self).__init__()

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
                tf.keras.layers.LeakyReLU(0.15),
                tf.keras.layers.Dense(64, name="IMG_layer2"),
                tf.keras.layers.LeakyReLU(0.15),
            ])

        self.lstm = tf.keras.layers.LSTM(256, return_sequences=True, return_state=True, name="LSTM_layer")

        self.dense = tf.keras.layers.Dense(128, name="dense_layer")

        self.leaky_relu1 = tf.keras.layers.LeakyReLU(0.2)

        self.dense2 = tf.keras.layers.Dense(64, name="dense_layer2")

        self.leaky_relu2 = tf.keras.layers.LeakyReLU(0.2)

        self.action_layer = tf.keras.layers.Dense(4, name="action_layer")

        # config input and output
        self.output_info = {'action': (4,), 'hidden1': (256,), 'hidden2': (256,)}
        self.input_name = ('img', 'actor_obs', 'hidden1', 'hidden2')

        # config optimizer
        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 3e-4,
                     'epsilon': 1e-5,
                     'clipnorm': 0.5,
                     },
        }

        # config special flag, it is used for special network, such as rnn
        self.rnn_flag = True

    def call(self, img, actor_obs, hidden1, hidden2, mask=None):
        
        # conv usully can not support sequence input, so we need to reshape the input
        img = tf.reshape(img, (-1, width, height, channel)) 
        encode = self.encoder(img)

        # When use rnn, if encode can't output as (batch_size, time_step, 64), use this reshape
        encode = tf.reshape(encode, (actor_obs.shape[0], actor_obs.shape[1], 64))

        x = tf.concat([encode, actor_obs], axis=-1)

        x, hidden1, hidden2 = self.lstm(x, initial_state=[hidden1, hidden2], mask=mask)

        x = self.dense(x)
        x = self.leaky_relu1(x)
        x = self.dense2(x)
        x = self.leaky_relu2(x)
        action = self.action_layer(x)

        return action, hidden1, hidden2
```

In all algorithms, you need to specify the output and input of the model. The output and input are specified by `output_info` and `input_name`. The `output_info` is a dictionary, the key is the name of the output, and the value is the shape of the output. The `input_name` is a tuple, the elements of the tuple are the name of the input. The order of the elements in the tuple is the same as the order of the input in the `call` method.


### Parrallel training

If you want to train your model in parallel, you just need to add the following code to your code.

```python
from mpi4py import MPI
from AquaML.Tool import allocate_gpu
comm = MPI.COMM_WORLD  # get group communicator
allocate_gpu(comm,1) # use GPU 1
rank = comm.Get_rank()
```

Then add `comm` to the algorithm runner initialization. 

At last, you can run your code with mpi. For example, if you want to use 4 gpus to train your model, you can run the following command.

```bash
mpirun -np 4 python your_code.py
```

**Note**: parallel training will be changed in the future.