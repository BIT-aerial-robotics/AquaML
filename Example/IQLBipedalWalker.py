import sys

sys.path.append('C:\RL\EvolutionRL')

from AquaML.framework.EvolutionML import EvolutionML
from AquaML.param.DataInfo import DataInfo
from AquaML.RobotAPI.GymWrapper import GymWrapper
from AquaML.policy.RealWorldPolicy import DeterminateRealWorldPolicy
from AquaML.policy.FixNNPolicy import FixNNPolicy
from AquaML.core.Tool import generate_gym_env_info
from AquaML.communicator.DebugCommunicator import DebugCommunicator
from AquaML.communicator.MPICommunicator import MPICommunicator
from AquaML.tf.OfflineRL.IQL import IQL, IQLParams
from AquaML.tf.PolicyCandidate.PEX import PEX, PEXParams
import tensorflow as tf

gym_env_name = 'BipedalWalker-v3'
task_name = 'IQL_BipedalWalker'

# 配置环境信息
info_element_dict, rl_state_names, rl_action_name, rl_reward_names = generate_gym_env_info(gym_env_name)
rl_action_name = rl_action_name[0]

data_info = DataInfo(
    scope_name='IQL_BipedalWalker',
)

for info_name, info_element in info_element_dict.items():
    data_info.add_element(**info_element)

data_info.set_rl_state(rl_state_names)
data_info.set_rl_action2(rl_action_name)
data_info.set_rl_rewards(rl_reward_names)

# 创建通信器

communicator = MPICommunicator(wait_time_out = 1000,  # 等待超时时间
                               check_time_interval = 0.000001,
                               detailed_log=False)
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
            'args': {'learning_rate': 2e-5,
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

        self.input_names = ('env_obs', 'action',)

        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 2e-5,
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


class State_value_net(tf.keras.Model):

    def __init__(self):
        super(State_value_net, self).__init__()

        self.dense1 = tf.keras.layers.Dense(512, activation='relu',
                                            kernel_initializer=tf.keras.initializers.orthogonal())
        self.dense2 = tf.keras.layers.Dense(256, activation='relu',
                                            kernel_initializer=tf.keras.initializers.orthogonal())
        self.dense3 = tf.keras.layers.Dense(1, activation=None, kernel_initializer=tf.keras.initializers.orthogonal())
        # self.log_std = tf.keras.layers.Dense(1)

        # self.learning_rate = 2e-5

        self.output_info = {'state_value': (1,), }

        self.input_names = ('env_obs',)

        self.optimizer_info = {
            'type': 'Adam',
            'args': {'learning_rate': 2e-5,
                     # 'epsilon': 1e-5,
                     # 'clipnorm': 0.5,
                     },
        }

    @tf.function
    def call(self, obs):
        x = self.dense1(obs)
        x = self.dense2(x)
        x = self.dense3(x)
        # log_std = self.log_std(x)

        return x

    def reset(self):
        pass


hyper_params = IQLParams(
    epoch=1000,  # TODO
    update_times=10,
    batch_size=512,
    update_start=1000,
    model_save_interval=20,
)

iql_param = {
    'actor': Actor_net,
    'q_critic': Critic_net,
    'state_value': State_value_net,
    'hyper_params': hyper_params,
    'weight_path': 'dataset/IQL_Bipedal_150000',
}

policy_updater = {
    'IQL': {
        'updater': IQL,
        'param': iql_param,
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
        'weight_path': 'dataset/IQL_Bipedal_150000/actor.h5',
    }
}

real_policies['update_policy'] = {
    'policy': DeterminateRealWorldPolicy,
    'param': {
        'name_scope': 'IQL',
        'candidate_action_id': 1,
        'keras_model': Actor_net,
        'switch_times': 5,
        'weight_path': 'dataset/IQL_Bipedal_150000/actor.h5',
    }
}

#################### 创建robot api ####################

robot_api = {
    'gym_env': {
        'api': GymWrapper,
        'param': {
            'name': 'gym_env',
            'gym_env_name': gym_env_name,
            'max_step': 1000,
            'gym_env_param': {
                'render_mode': 'human',
                'hardcore': True
            }
        }
    }
}

#################### 创建policy candidate ####################
pex_param = PEXParams(
    inv_temperature=1,
    algo='IQL'
)
policy_selector = {
    'PEX': {
        'selector': PEX,
        'param': {
            'critic1': Critic_net,
            'critic2': Critic_net,
            'initial_weight_path': 'dataset/IQL_Bipedal_150000',
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
    capacity=102400,
    communicator=communicator,
    offline_dataset_path='dataset/ExpertBipedalWalker',
    process_config_path='C:\RL\EvolutionRL/AquaML/config/process/EvolutionML.yaml'
)

evolution_ml.run()