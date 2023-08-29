import sys

sys.path.append('..')

from AquaML.rlalgo.AqauRL import AquaRL, LoadFlag
from AquaML.rlalgo.AgentParameters import AMPAgentParameter
from AquaML.rlalgo.AMPAgent import AMPAgent
import numpy as np
import gym
from AquaML.DataType import DataInfo
from AquaML.core.RLToolKit import RLBaseEnv
from AquaML.core.RLToolKit import RLVectorEnv
import tensorflow as tf


class Actor_net(tf.keras.Model):

    def __init__(self):
        super(Actor_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.action_layer = tf.keras.layers.Dense(4)
        # self.log_std = tf.keras.layers.Dense(1)

        # self.learning_rate = 2e-5

        self.output_info = {'action': (4,), }

        self.input_name = ('obs',)

        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 4e-6,
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
        self.dense2 = tf.keras.layers.Dense(512, activation='relu',
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
            'args': {'learning_rate': 1e-4,
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


class Discriminator_net(tf.keras.Model):
    def __init__(self):
        super(Discriminator_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(512, activation='relu',
                                            kernel_initializer=tf.keras.initializers.orthogonal())
        self.dense2 = tf.keras.layers.Dense(512, activation='relu',
                                            kernel_initializer=tf.keras.initializers.orthogonal())
        self.dense3 = tf.keras.layers.Dense(1, activation=None, kernel_initializer=tf.keras.initializers.orthogonal())

        # self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=0.3,
        #     decay_steps=10,
        #     decay_rate=0.95,
        #
        # )

        self.output_info = {'value': (1,)}

        self.input_name = ('obs', 'next_obs',)

        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 1e-5,
                     # 'epsilon': 1e-5,
                     # 'clipnorm': 0.5,
                     }
        }

    @tf.function
    def call(self, obs, next_obs):
        input = tf.concat([obs, next_obs], axis=-1)
        x = self.dense1(input)
        x = self.dense2(x)
        value = self.dense3(x)

        return value

    def reset(self):
        pass


class BipedalWalker(RLBaseEnv):
    def __init__(self, env_name="BipedalWalker-v3"):
        super().__init__()
        # TODO: update in the future
        self.step_s = 0
        self.env = gym.make(env_name, hardcore=True)
        self.env_name = env_name

        # our frame work support POMDP env
        self._obs_info = DataInfo(
            names=('obs', 'step',),
            shapes=((24,), (1,)),
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


vec_env = RLVectorEnv(BipedalWalker, 4, normalize_obs=False, )

# load_flag = LoadFlag(
#     actor=True
# )
parameters = AMPAgentParameter(
    rollout_steps=1000,
    epochs=5000,
    batch_size=256,

    # AMP parameters
    k_batch_size=256,
    update_discriminator_times=15,
    discriminator_replay_buffer_size=int(1e5),
    gp_coef=25.0,
    task_rew_coef=0.5,
    style_rew_coef=0.5,

    update_times=1,
    max_steps=1000,
    update_actor_times=1,
    update_critic_times=1,
    eval_episodes=5,
    eval_interval=10000,
    eval_episode_length=200,
    entropy_coef=0.005,
    batch_advantage_normalization=False,
    checkpoint_interval=20,
    log_std_init_value=-0.5,
    train_all=False,
    min_steps=24,
    target_kl=0.01,
    lamda=0.95,
    gamma=0.99,
    summary_style='step',
    summary_steps=1000,
)


agent_info_dict = {
    'actor': Actor_net,
    'critic': Critic_net,
    'agent_params': parameters,
    'discriminator': Discriminator_net,
    'expert_dataset_path': 'ExpertBipedalWalker',
}


load_flag = LoadFlag(
    actor=True,
    critic=False,
    state_normalizer=False,
    reward_normalizer=False
)

rl = AquaRL(
    env=vec_env,
    agent=AMPAgent,
    agent_info_dict=agent_info_dict,
    # eval_env=eval_env,
    # comm=comm,
    name='AMPB',
    reward_norm=False,
    state_norm=False,
    decay_lr=False,
    # snyc_norm_per=10,
    check_point_path='TD3BC',
    load_flag=load_flag,
)

rl.run()