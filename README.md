# AquaML

## Features

1. Support reinforcement learning, generative learning algorithm.
2. Support reinforcement learning training with recurrent neural networks.
3. Support multi-thread sampling and parameters tuning.
4. Support high performance computer(HPC)
5.  Data communication has almost zero lat ency when running on a single machine.

## Install

## Tutorials

### Train Pendulum-v0 with soft-actor-critic(SAC)

This tutorial is to show how to use AquaML to control pendulum-v0(https://gym.openai.com/envs/Pendulum-v0/). The environment is a continuous action space environment. The action is a 1-dim vector. The observation is a 3-dim vector.

All the codes are available in Tutorial/tutorial1.py.

#### Create neural network model for reinforcement learning

AquaML just supports 'expert'  TF model style, you can learn more in  https://tensorflow.google.cn/overview. But in AquaML, the reinforcement learning model must inherit from  ``RLBaseModel``.

##### 1. Actor

Before creating please do the following things:

```python
import tensorflow as tf
from AquaML.BaseClass import RLBaseModel  # Base class for reinforcement network
```

 Then we can create the model.

```python
class Actor_net(RLBaseModel):
    def __init__(self):
        super(Actor_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='tanh')
```

Point out learning rate. In ``AuqaML``, each model can have its own learning rate

```python
		self._learning_rate = 2e-4
```

Our framework can fusion muti type data, please specify the input data name

```python
		self._input_name = ('obs',)
```

Actor net is special than others, its out may be different. Thus you should specify 

#### Create new reinforcement algorithm

#### Create reinforcement learning environment



## Change logs

#### v1.1 

1. unify ```MPIRuner``` API.
2. Add ``com`` package, it contains all base class.
3. ``save_data`` and ``load_data`` are created for supervised learning and expert learning. 
4. Gradually convert our framework to next generation like HPC-v0.1.
5. The following algos just use ``DataCollector``(support all type of algo) instead of ``DataManeger``.
6. Add plot tools for rosbag, paper.

#### v2.0

1. split optimize thread and worker thread.
2. add soft actor critic.



## Requirement

seaborn<=0.9
