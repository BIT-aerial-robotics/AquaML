import sys

sys.path.append('..')
# from AquaML.Tool import allocate_gpu
# from mpi4py import MPI
#
# #
# # #
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# allocate_gpu(comm, 0)

from AquaML.rlalgo.AqauRL import AquaRL, LoadFlag
from AquaML.rlalgo.AgentParameters import PPOAgentParameter
from AquaML.rlalgo.PPOAgent import PPOAgent
import numpy as np
import gym
from AquaML.DataType import DataInfo
from AquaML.core.RLToolKit import RLBaseEnv
from AquaML.core.RLToolKit import RLVectorEnv
import tensorflow as tf


class FEnv(RLBaseEnv):

    def __init__(self):
        super().__init__()

        self.s_1 = 60
        self.s_2 = 100

        self._obs_info = DataInfo(
            names=('obs', ),
            shapes=((2,), ),
            dtypes=np.float32
        )

        self._reward_info = ['total_reward', ]

    def reset(self):
        self.s_1 = np.squeeze(np.random.randint(35, 65))

        self.s_2 = np.squeeze(np.random.randint(80, 100))

        obs = {
            'obs': np.array([self.s_1, self.s_2]).reshape((1,2))
        }

        return obs, True

    def step(self, action):

        action = action['action']

        if isinstance(action, tf.Tensor):
            action = action.numpy()

        action_ = np.reshape(action, (2,))

        action_ = np.clip(action_, 0, 1)*100

        # action_ = (action_ + 1) /2  * 60

        self.s_1 = self.stochastic_internal_dynamic_model(x=self.s_1, h=action_[0])

        self.s_2 = self.stochastic_internal_dynamic_model(x=self.s_2, h=action_[1])

        reward = self.s_1 + self.s_2 - 1000 * (abs(self.s_1 - self.s_2))

        reward = {
            'total_reward':reward
        }

        obs = {
            'obs': np.array([self.s_1, self.s_2]).reshape((1,2))
        }

        return obs, reward, False, None

    def close(self):
        pass

    def stochastic_internal_dynamic_model(slef, x, h):
        alpha = 0.8
        r = 1
        i = 7
        w_n = 15
        w_s = 280
        a = (1 - alpha) * ((x + w_n) * (1 + r) + w_n)
        b = (1 - alpha) * (w_s + (x - h) * (1 + i))
        g = (1 - alpha) * (h * (1 + i) - w_s) / ((1 + i) * (1 - alpha) - 1)
        if a > b:
            x_next = (1 - alpha) * ((x + w_n) * (1 + r) + w_n)
        elif x > h:
            x_next = (1 - alpha) * (w_s + (x - h) * (1 + r))
        else:
            x_next = (1 - alpha) * (w_s + (x - h) * (1 + i))
        # print(g)
        x_1 = (1 - alpha) * w_n * (2 + r) / (1 - (1 - alpha) * (1 + r))
        # print(x_1)
        x_2 = (1 - alpha) * (w_s - h * (1 + r)) / (1 - (1 - alpha) * (1 + r))
        # print(x_2)
        if a > b:
            s = np.random.uniform(-10, 10, size=1)
            x_next = x_next + s
        else:
            s = np.random.uniform(-15, 15, size=1)
            x_next = x_next + s

        return x_next



class Actor_net(tf.keras.Model):

    def __init__(self):
        super(Actor_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.action_layer = tf.keras.layers.Dense(2, activation=None)
        # self.log_std = tf.keras.layers.Dense(1)

        # self.learning_rate = 2e-5

        self.output_info = {'action': (2,), }

        self.input_name = ('obs',)

        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 3e-4,
                     # 'epsilon': 1e-5,
                     # 'clipnorm': 0.5,
                     },
        }

    @tf.function
    def call(self, obs, mask=None):
        x = self.dense1(obs)
        x = self.dense2(x)
        action = self.action_layer(x)
        # log_std = self.log_std(x)

        return (action,)

    def reset(self):
        pass


class Critic_net(tf.keras.Model):
    def __init__(self):
        super(Critic_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(32, activation='relu',
                                            kernel_initializer=tf.keras.initializers.orthogonal())
        self.dense2 = tf.keras.layers.Dense(32, activation='relu',
                                            kernel_initializer=tf.keras.initializers.orthogonal())
        self.dense3 = tf.keras.layers.Dense(1, activation=None, kernel_initializer=tf.keras.initializers.orthogonal())

        # self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=0.3,
        #     decay_steps=10,
        #     decay_rate=0.95,
        #
        # )

        self.output_info = {'value': (1,)}

        self.input_name = ('obs',)

        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 3e-4,
                     # 'epsilon': 1e-5,
                     # 'clipnorm': 0.5,
                     }
        }

    @tf.function
    def call(self, obs):
        x = self.dense1(obs)
        x = self.dense2(x)
        value = self.dense3(x)

        return value

    def reset(self):
        pass



vec_env = RLVectorEnv(FEnv, 20, normalize_obs=False, )


parameters = PPOAgentParameter(
    rollout_steps=100,
    epochs=10000,
    batch_size=1000,
    update_times=4,
    max_steps=100,
    update_actor_times=1,
    update_critic_times=2,
    eval_episodes=5,
    eval_interval=100000,
    eval_episode_length=200,
    entropy_coef=0.0,
    batch_advantage_normalization=False,
    clip_ratio=0.2,


    checkpoint_interval=20,
    log_std_init_value=1.0,
    train_all=True,
    min_steps=5,
    target_kl=0.01,
    lamda=0.95,
    gamma=0.99,
)

agent_info_dict = {
    'actor': Actor_net,
    'critic': Critic_net,
    'agent_params': parameters,
}

load_flag = LoadFlag(
    actor=True,
    critic=False,
    state_normalizer=True,
    reward_normalizer=False
)

rl = AquaRL(
    env=vec_env,
    agent=PPOAgent,
    agent_info_dict=agent_info_dict,
    name='FFenv',
    reward_norm=True,
    state_norm=False,
    decay_lr=False,
    snyc_norm_per=100,
    distributed_norm=False,
    reset_norm_per=100,
    # check_point_path='cache',
    # load_flag=load_flag,
)

rl.run()
