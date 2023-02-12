# AquaML

## Features

1. Support reinforcement learning, generative learning algorithm.
2. Support reinforcement learning training with recurrent neural networks.
2. Support RNN for reinforcement learning and provide two basic forms.
3. Support multi-thread sampling and parameters tuning.
4. Support high performance computer(HPC)
5. Data communication has almost zero lat ency when running on a single machine.

## Install

## Tutorials

### Train Pendulum-v0 with soft-actor-critic(SAC)

This tutorial is to show how to use AquaML to control pendulum-v0(https://gym.openai.com/envs/Pendulum-v0/). The
environment is a continuous action space environment. The action is a 1-dim vector. The observation is a 3-dim vector.

All the codes are available in Tutorial/tutorial1.py.

#### Create neural network model for reinforcement learning

AquaML just supports 'expert' TF model style, you can learn more in  https://tensorflow.google.cn/overview. But in
AquaML, the reinforcement learning model must inherit from  ``RLBaseModel``.

##### 1. Actor

Before creating please do the following things:

```python
import tensorflow as tf
```

Then we can create the model.

```python
class Actor_net(tf.keras.Model):
    def __init__(self):
        super(Actor_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='tanh')
```

Point out learning rate. In ``AuqaML``, each model can have its own learning rate

```python
    self.learning_rate = 2e-4
```

Our framework can fusion muti type data, please specify the input data name

```python
    self.input_name = ('obs',)
```

Actor net is special than others, its out may be different. Thus you should specify .

```python
    self.output_info = {'action': (1,)}
```

Then specify the optimizer of your neural network.

``_name`` contains name of data, _info also contains shape.

```python
        self.optimizer = 'Adam'
```

Then declaim ``call`` function:

```python
    def call(self, obs):
    x = self.dense1(obs)
    x = self.dense2(x)
    x = self.dense3(x)

    # the output of actor network must be a tuple
    # and the order of output must be the same as the order of output name

    return (x,)
```

For actor, the return must be a tuple.

``reset`` function is also needed in ``AquaML``. If your neural network dose not contain RNN, just write pass like:

```python
    def reset(self):
    # This model does not contain RNN, so this function is not necessary,
    # just pass

    # If the model contains RNN, you should reset the state of RNN
    pass
```

When using RNN, ``reset`` can reset the hidden.

Our frame work support adaptive ``log_std`` . So the output of the ``Actor_net`` also contains ``log_std``.

##### 2. Q value network

Creating Q value network is similar to creating actor. However, the ``call`` 's return of Q is tf tensor not tuple. And
in Q, the ``output_info`` can not be specified.

##### 3. Environment

If you use Gym environment, the you can:

```python
from AquaML.Tool import GymEnvWrapper  # Gym environment wrapper

env = GymEnvWrapper('Pendulum-v1')
```

##### 4. Define algorithm parameters

This tutorial is about SAC, so :

```python
from AquaML.rlalgo.Parameters import SAC2_parameter

sac_parameter = SAC2_parameter(
    epoch_length=200,
    n_epochs=10,
    batch_size=32,
    discount=0.99,
    alpha=0.2,
    tau=0.005,
    buffer_size=100000,
    mini_buffer_size=1000,
    update_interval=50,
)

model_class_dict = {
    'actor': Actor_net,
    'qf1': Q_net,
    'qf2': Q_net,
}
```

##### 5. Create task starter

```python
from AquaML.starter.RLTaskStarter import RLTaskStarter  # RL task starter

starter = RLTaskStarter(
    env=env,
    model_class_dict=model_class_dict,
    algo=SAC2,
    algo_hyperparameter=sac_parameter,
)
```

##### 6. Run task

```python
starter.run()
```

##### 7. Run by MPI

You can change the following codes to run parallelly.

###### Configure gpu

```python
import sys

sys.path.append('..')

from AquaML.Tool import allocate_gpu
from mpi4py import MPI

# get group communicator
comm = MPI.COMM_WORLD
allocate_gpu(comm)
```

Notice: This block must add at the head of python script.

###### Revise hyper parameters

```python
sac_parameter = SAC2_parameter(
    episode_length=200,
    n_epochs=200,
    batch_size=256,
    discount=0.99,
    tau=0.005,
    buffer_size=100000,
    mini_buffer_size=5000,
    update_interval=1000,
    display_interval=1,
    calculate_episodes=5,
    alpha_learning_rate=3e-3,
    update_times=100,
)
```

###### add MPI.comm to rl starter

```python
starter = RLTaskStarter(
    env=env,
    model_class_dict=model_class_dict,
    algo=SAC2,
    algo_hyperparameter=sac_parameter,
    mpi_comm=comm,
    name='SAC'
)

```

After those steps, you can run by the following command in terminal:

```bash
mpirun -n 6 python Tutorial1.py
```



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
