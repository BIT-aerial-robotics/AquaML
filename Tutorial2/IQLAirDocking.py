import sys

sys.path.append('..')

from AquaML.rlalgo.AqauRL import AquaRL, LoadFlag
from AquaML.rlalgo.AgentParameters import IQLAgentParameters
from AquaML.rlalgo.IQLAgent import IQLAgent
import numpy as np
import gym
from AquaML.DataType import DataInfo
from AquaML.core.RLToolKit import RLBaseEnv
from AquaML.core.RLToolKit import RLVectorEnv
import tensorflow as tf
from env_air_sb3.AMP_sample import AMP
from env_air_sb3.env_params import rew_coeff_sou


class Actor_net(tf.keras.Model):

    def __init__(self):
        super(Actor_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.action_layer = tf.keras.layers.Dense(8)  #TODO 输出范围是
        # self.log_std = tf.keras.layers.Dense(1)

        # self.learning_rate = 2e-5

        self.output_info = {'action': (8,), }

        self.input_name = ('obs',)

        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 3e-4,
                     # 'epsilon': 1e-54
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

        self.dense1 = tf.keras.layers.Dense(512, activation='relu',
                                            kernel_initializer=tf.keras.initializers.orthogonal())
        self.dense2 = tf.keras.layers.Dense(256, activation='relu',
                                            kernel_initializer=tf.keras.initializers.orthogonal())
        self.dense3 = tf.keras.layers.Dense(1, activation=None, kernel_initializer=tf.keras.initializers.orthogonal())

        self.dense4 = tf.keras.layers.Dense(512, activation='relu',
                                            kernel_initializer=tf.keras.initializers.orthogonal())
        self.dense5 = tf.keras.layers.Dense(256, activation='relu',
                                            kernel_initializer=tf.keras.initializers.orthogonal())
        self.dense6 = tf.keras.layers.Dense(1, activation=None, kernel_initializer=tf.keras.initializers.orthogonal())

        # self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=0.3,
        #     decay_steps=10,
        #     decay_rate=0.95,
        #
        # )

        self.output_info = {'value': (2,)}

        self.input_name = ('obs', 'action',)

        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 3e-4,
                     # 'epsilon': 1e-5,
                     # 'clipnorm': 0.5,
                     }
        }

    @tf.function
    def call(self, obs, action):
        new_obs = tf.concat([obs, action], axis=1)

        q1 = self.dense1(new_obs)
        q1 = self.dense2(q1)
        q1 = self.dense3(q1)

        q2 = self.dense4(new_obs)
        q2 = self.dense5(q2)
        q2 = self.dense6(q2)

        return q1, q2

    def reset(self):
        pass


class S_Value_net(tf.keras.Model):

    def __init__(self):
        super(S_Value_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(512, activation='relu',
                                            kernel_initializer=tf.keras.initializers.orthogonal())
        self.dense2 = tf.keras.layers.Dense(256, activation='relu',
                                            kernel_initializer=tf.keras.initializers.orthogonal())
        self.dense3 = tf.keras.layers.Dense(1, activation=None, kernel_initializer=tf.keras.initializers.orthogonal())
        # self.log_std = tf.keras.layers.Dense(1)

        # self.learning_rate = 2e-5

        self.output_info = {'s_value': (1,), }

        self.input_name = ('obs',)

        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 3e-4,
                     # 'epsilon': 1e-54
                     # 'clipnorm': 0.5,
                     },
        }

    @tf.function
    def call(self, obs, mask=None):
        v = self.dense1(obs)
        v = self.dense2(v)
        v = self.dense3(v)
        # log_std = self.log_std(x)

        return v

    def reset(self):
        pass


class AirD(RLBaseEnv):
    def __init__(self, env_name="AirDocking"):
        super().__init__()
        # TODO: update in the future
        self.step_s = 0
        self.env = AMP(rew_coeff=rew_coeff_sou, sense_noise='default')
        self.env_name = env_name

        # our frame work support POMDP env
        self._obs_info = DataInfo(
            names=('obs', 'step',),
            shapes=((42,), (1,)),
            dtypes=np.float32
        )

        self._reward_info = ['total_reward', 'indicate_1']

    def reset(self):
        observation = self.env.reset()
        observation = observation[0].reshape(1, -1)

        self.step_s = 0
        # observation = observation.

        # observation = tf.convert_to_tensor(observation, dtype=tf.float32)

        obs = {'obs': observation, 'step': self.step_s}

        obs = self.initial_obs(obs)

        return obs, True  # 2.0.1 new version

    def step(self, action_dict):
        self.step_s += 1
        action = action_dict['action']
        if isinstance(action, tf.Tensor):
            action = action.numpy()
        # action *= 2
        observation, reward, done, tru, info = self.env.step(action)
        observation = observation.reshape(1, -1)

        indicate_1 = reward
        #
        if reward <= -100:
            reward = -1
            done = True

        obs = {'obs': observation, 'step': self.step_s}

        obs = self.check_obs(obs, action_dict)

        reward = {'total_reward': reward, 'indicate_1': indicate_1}

        # if self.id == 0:
        #     print('reward', reward)

        return obs, reward, done, info

    def close(self):
        self.env.close()


env = AirD()  # need environment provide obs_info and reward_info

parameters = IQLAgentParameters(
    epochs=1000,
    batch_size=1024,
    tau=0.005,
    noise_clip_range=0.5,
    sigma=0.2,
    # policy_noise=0.2,
    # action_clip_range=1,
    update_times=1,
    delay_update=2,
    # normalize_reward=True,
    # normalize=True,
    checkpoint_interval=25,
)

agent_info_dict = {
    'actor': Actor_net,
    'q_critic': Critic_net,
    's_value': S_Value_net,
    'agent_params': parameters,
    'expert_dataset_path': '/home/ming/aaa/AquaML-2.2.0/dataset/Joint200',
}

offline_rl = AquaRL(
    env=env,
    agent=IQLAgent,
    agent_info_dict=agent_info_dict,
    # eval_env=eval_env,
    # comm=comm,
    name='debug_iqlaird_test1',
    # reward_norm=True,
    # state_norm=False,
    # decay_lr=False,
    # snyc_norm_per=10,
    # check_point_path='cache',
    # load_flag=load_flag,
)

offline_rl.run_offline()