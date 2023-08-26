import sys

sys.path.append('..')

from AquaML.rlalgo.AqauRL import AquaRL, LoadFlag
from AquaML.rlalgo.AgentParameters import TD3AgentParameters
from AquaML.rlalgo.TD3Agent import TD3Agent
import numpy as np
import gym
from AquaML.DataType import DataInfo
from AquaML.core.RLToolKit import RLBaseEnv
from AquaML.core.RLToolKit import RLVectorEnv
import tensorflow as tf


class Actor_net(tf.keras.Model):

    def __init__(self):
        super(Actor_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.action_layer = tf.keras.layers.Dense(1)
        # self.log_std = tf.keras.layers.Dense(1)

        # self.learning_rate = 2e-5

        self.output_info = {'action': (1,), }

        self.input_name = ('obs',)

        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 1e-3,
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

        self.dense1 = tf.keras.layers.Dense(64, activation='relu',
                                            kernel_initializer=tf.keras.initializers.orthogonal())
        self.dense2 = tf.keras.layers.Dense(64, activation='relu',
                                            kernel_initializer=tf.keras.initializers.orthogonal())
        self.dense3 = tf.keras.layers.Dense(1, activation=None, kernel_initializer=tf.keras.initializers.orthogonal())

        # self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=0.3,
        #     decay_steps=10,
        #     decay_rate=0.95,
        #
        # )

        self.output_info = {'value': (1,)}

        self.input_name = ('obs', 'action',)

        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 1e-3,
                     # 'epsilon': 1e-5,
                     # 'clipnorm': 0.5,
                     }
        }

    @tf.function
    def call(self, obs, action):
        new_obs = tf.concat([obs, action], axis=-1)
        x = self.dense1(new_obs)
        x = self.dense2(x)
        value = self.dense3(x)

        return value

    def reset(self):
        pass


class PendulumWrapper(RLBaseEnv):
    def __init__(self, env_name="Pendulum-v1"):
        super().__init__()
        # TODO: update in the future
        self.step_s = 0
        self.env = gym.make(env_name)
        self.env_name = env_name

        # our frame work support POMDP env
        self._obs_info = DataInfo(
            names=('obs', 'step',),
            shapes=((3,), (1,)),
            dtypes=np.float32
        )

        self._reward_info = ['total_reward', ]

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
        action *= 2
        observation, reward, done, tru, info = self.env.step(action)
        observation = observation.reshape(1, -1)

        obs = {'obs': observation, 'step': self.step_s}

        obs = self.check_obs(obs, action_dict)

        reward = {'total_reward': reward}

        # if self.id == 0:
        #     print('reward', reward)

        return obs, reward, done, info

    def close(self):
        self.env.close()


env = PendulumWrapper()  # need environment provide obs_info and reward_info
eval_env = RLVectorEnv(PendulumWrapper,20)

parameters = TD3AgentParameters(
    epochs=10000000,
    max_steps=200,
    rollout_steps=1,
    batch_size=128,
    update_times=1,
    eval_interval=400,
    eval_episodes=1,
    eval_episode_length=200,
    learning_starts=600
)

agent_info_dict = {
    'actor': Actor_net,
    'q_critic': Critic_net,
    'agent_params': parameters,
    # 'expert_dataset_path': 'ExpertPendulum',
}

rl = AquaRL(
    env=env,
    agent=TD3Agent,
    agent_info_dict=agent_info_dict,
    eval_env=eval_env,
    # comm=comm,
    name='debug1',
    # reward_norm=True,
    # state_norm=False,
    # decay_lr=False,
    # snyc_norm_per=10,
    # check_point_path='cache',
    # load_flag=load_flag,
)

rl.run_off_policy()
