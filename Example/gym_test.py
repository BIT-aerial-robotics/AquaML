'''
用于测试gym环境，测试算法能否运行
'''

import sys

sys.path.append('/Users/yangtao/Documents/code.nosync/EvolutionRL')

from AquaML.framework.EvolutionML import EvolutionML
from AquaML.param.DataInfo import DataInfo
from AquaML.RobotAPI.GymWrapper import GymWrapper
from AquaML.policy.RealWorldPolicy import DeterminateRealWorldPolicy
from AquaML.policy.FixNNPolicy import FixNNPolicy
from AquaML.core.Tool import generate_gym_env_info
from AquaML.communicator.DebugCommunicator import DebugCommunicator
from AquaML.communicator.MPICommunicator import MPICommunicator
from AquaML.tf.OfflineRL.TD3BC import TD3BC, TD3BCParams
from AquaML.tf.PolicyCandidate.PEX import PEX, PEXParams
import tensorflow as tf

gym_env_name = 'BipedalWalker-v3'
task_name = 'gym_test'

# 配置环境信息
info_element_dict, rl_state_names, rl_action_name, rl_reward_names = generate_gym_env_info(gym_env_name)
rl_action_name = rl_action_name[0]

data_info = DataInfo(
    scope_name='gym_env',
)

for info_name, info_element in info_element_dict.items():
    data_info.add_element(**info_element)

data_info.set_rl_state(rl_state_names)
data_info.set_rl_action2(rl_action_name)
data_info.set_rl_rewards(rl_reward_names)

# 创建通信器

communicator = MPICommunicator(wait_time_out = 1000,  # 等待超时时间
                               check_time_interval = 2,
                               log=True)
# communicator = DebugCommunicator()

#################### 配置updater ####################

# 创建动作网络
class Actor_net(tf.keras.Model):

    def __init__(self):
        super(Actor_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.action_layer = tf.keras.layers.Dense(4, activation='tanh')
        # self.log_std = tf.keras.layers.Dense(1)

        # self.learning_rate = 2e-5

        self.output_info = {'action': (4,), }  # 目前不支持LSTM

        self.input_names = ('env_obs',)

        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 1e-4,
                     # 'epsilon': 1e-5,
                     # 'clipnorm': 0.5,
                     },
        }

    # @tf.function
    def call(self, obs, mask=None):
        x = self.dense1(obs)
        x = self.dense2(x)
        action = self.action_layer(x)
        # log_std = self.log_std(x)

        # return (action,)
        return action

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

        # self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=0.3,
        #     decay_steps=10,
        #     decay_rate=0.95,
        #
        # )

        self.output_info = {'value': (1,)}

        self.input_names = ('env_obs', 'action',)

        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 1e-4,
                     # 'epsilon': 1e-5,
                     # 'clipnorm': 0.5,
                     }
        }

    @tf.function
    def call(self, obs, action):
        new_obs = tf.concat([obs, action], axis=1)
        x = self.dense1(new_obs)
        x = self.dense2(x)
        value = self.dense3(x)

        return value

    def reset(self):
        pass


hyper_params = TD3BCParams(
    epoch=1000,
    update_times=20,
    batch_size=2,
    update_start=2,
    model_save_interval=100,
)

td3bc_param = {
    'actor': Actor_net,
    'q_critic': Critic_net,
    'hyper_params': hyper_params,

}

policy_updater = {
    'TD3BC': {
        'updater': TD3BC,
        'param': td3bc_param,
    }
}
#################### 配置real policy ####################

# 配置fixed policy

real_policies = {}

real_policies['fix_policy'] = {
    'policy': FixNNPolicy,
    'param': {
        'name_scope': 'fix_policy',
        'candidate_action_id': 0,
        'keras_model': Actor_net,
        'weight_path': 'dataset/TD3BC_Bipedal_560/actor.h5',
    }
}

real_policies['update_policy'] = {
    'policy': DeterminateRealWorldPolicy,
    'param': {
        'name_scope': 'TD3BC',
        'candidate_action_id': 1,
        'keras_model': Actor_net,
        'switch_times': 5,
        'weight_path': 'dataset/TD3BC_Bipedal_560/actor.h5',
    }
}

#################### 创建robot api ####################

robot_api = {
    'gym_env': {
        'api': GymWrapper,
        'param': {
            'name': 'gym_env',
            'gym_env_name': gym_env_name,
            # 'gym_env_param': {'render_mode': 'human'}
        }
    }
}

#################### 创建policy candidate ####################
pex_param = PEXParams(
    inv_temperature=1,
    algo='TD3BC'
)
policy_selector = {
    'PEX': {
        'selector': PEX,
        'param': {
            'critic1': Critic_net,
            'critic2': Critic_net,
            'weight_path': 'dataset/TD3BC_Bipedal_560',
            'hyper_params': pex_param,
        }
    }
}

# 创建evolution ml

evolution_ml = EvolutionML(
    task_name=task_name,
    env_info=data_info,
    real_policies=real_policies,
    policy_updater=policy_updater,
    robot_api=robot_api,
    policy_selector=policy_selector,
    capacity=2,
    communicator=communicator,
    process_config_path='C:\\Users\\29184\\Documents\GitHub\\EvolutionRL\\AquaML\\config\\process\\EvolutionML.yaml'
)

evolution_ml.run()
