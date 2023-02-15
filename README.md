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

``reset`` function is also needed in ``AquaML``. 

```python
    def reset(self):
    	pass
```

Our frame work support adaptive ``log_std`` . So the output of the ``Actor_net`` also contains ``log_std``.

For POMDP version:

```python
class Actor_net(tf.keras.Model):

    def __init__(self):
        super(Actor_net, self).__init__()

        self.lstm = tf.keras.layers.LSTM(32, input_shape=(2,), return_sequences=False, return_state=True)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.action_layer = tf.keras.layers.Dense(1, activation='tanh')
        self.log_std = tf.keras.layers.Dense(1)

        self.learning_rate = 2e-4

        self.output_info = {'action': (1,), 'log_std': (1,), 'hidden1': (32,), 'hidden2': (32,)}

        self.input_name = ('pos', 'hidden1', 'hidden2')

        self.optimizer = 'Adam'

    # @tf.function
    def call(self, vel, hidden1, hidden2):
        hidden_states = (hidden1, hidden2)
        vel = tf.expand_dims(vel, axis=1)
        whole_seq, last_seq, hidden_state = self.lstm(vel, hidden_states)
        x = self.dense1(whole_seq)
        x = self.dense2(x)
        action = self.action_layer(x)
        log_std = self.log_std(x)

        return (action, log_std, last_seq, hidden_state)

    def reset(self):
        pass

```



##### 2. Q value network

Creating Q value network is similar to creating actor. However, the ``call`` 's return of Q is tf tensor not tuple. And
in Q, the ``output_info`` can not be specified.

##### 3. Environment

###### Gym environment

If you use Gym environment, the you can wrap the environment by the following steps:

###### 1. Inherit from ``AquaML.BaseClass.RLBaseEnv``

```python
from AquaML.BaseClass import RLBaseEnv
```

###### 2. Specify information of environment

```python
import gym
from AquaML.DataType import DataInfo

class PendulumWrapper(RLBaseEnv):
	    def __init__(self, env_name: str):
        super().__init__()
        # TODO: update in the future
        self.env = gym.make(env_name)
        self.env_name = env_name
        
        # our frame work support POMDP env
        self._obs_info = DataInfo(
            names=('obs',),
            shapes=((3,)),
            dtypes=np.float32
        )
```

If you want change the environment into POMDP, then you can:

```python
       # If you want specify different observation you can
        self._obs_info = DataInfo(
            names=('obs','pos'),
            shapes=((3,),(2,)),
            dtypes=np.float32
        )
```

###### 3. Implement ``reset``

```python
    def reset(self):
        observation = self.env.reset()
        observation = observation.reshape(1, -1)

        # observation = tf.convert_to_tensor(observation, dtype=tf.float32)

        obs = {'obs': observation}

        obs = self.initial_obs(obs)

        return obs
```

 For POMDP version:

```python
    def reset(self):
        observation = self.env.reset()
        observation = observation.reshape(1, -1)

        # observation = tf.convert_to_tensor(observation, dtype=tf.float32)

        # obs = {'obs': observation}
        obs = {'obs': observation, 'pos': observation[:, :2]}

        obs = self.initial_obs(obs)

        return obs
```

###### 4. Implement ``step``

```python
    def step(self, action_dict):
        action = action_dict['action']
        action *= 2
        observation, reward, done, info = self.env.step(action)
        observation = observation.reshape(1, -1)

        obs = {'obs': observation}

        obs = self.check_obs(obs, action_dict)

        
        reward = {'total_reward': reward}

        return obs, reward, done, info
```

For POMDP version:

```python
    def step(self, action_dict):
        action = action_dict['action']
        action *= 2
        observation, reward, done, info = self.env.step(action)
        observation = observation.reshape(1, -1)

        obs = {'obs': observation, 'pos': observation[:, :2]}

        obs = self.check_obs(obs, action_dict)

        # obs = {'obs': observation}
        reward = {'total_reward': reward}

        return obs, reward, done, info
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
