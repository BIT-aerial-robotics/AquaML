import sys
# sys.path.append('/Users/yangtao/Documents/code.nosync/EvolutionRL')
sys.path.append('C:/Users/29184/Documents/GitHub/EvolutionRL')
import AquaML
from AquaML.tf.OnlineRL import PPOAlgo, PPOParam
from AquaML.framework import RL
from AquaML.Tool import GymnasiumMaker
import tensorflow as tf
import numpy as np




# 环境的参数
env_args = {
    'env_name': 'Pendulum-v1',
    'env_args': {}
}

param = PPOParam(
    rollout_steps=200,
    epoch=200,
    batch_size=1000,
    env_num=20,
    envs_args=env_args,
    summary_steps=200,
)

AquaML.init(
    hyper_params=param,
    root_path='test',
    memory_path='test',
    wandb_project='test'
)

######################
# 定义模型
# 当前推荐使用tf.keras.Model作为模型的基类
######################

class Actor(tf.keras.Model):
    def __init__(self):
        # AquaML.ModelBase.__init__(self)
        tf.keras.Model.__init__(self)
        
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.action_layer = tf.keras.layers.Dense(1)
        
        self.learning_rate = 3e-4
        self.optimizer_type = 'Adam'
        self.output_info = AquaML.DataInfo(
            names=('action',),
            shapes=((1,),),
            dtypes=(np.float32,)
        )
        self.input_names = ('obs',)
        self.optimizer_other_args = {
                        'epsilon': 1e-5,
                     'clipnorm': 0.5,}
    @tf.function
    def call(self, obs):
        x = self.dense1(obs)
        x = self.dense2(x)
        action = self.action_layer(x)
        return (action,)

class Critic(tf.keras.Model):
    def __init__(self):
        # AquaML.ModelBase.__init__(self)
        tf.keras.Model.__init__(self)
        
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.value_layer = tf.keras.layers.Dense(1)
        
        self.learning_rate = 2e-4
        self.output_info = AquaML.DataInfo(
            names=('value',),
            shapes=((1,),),
            dtypes=(np.float32,)
        )
        self.input_names = ('obs',)
        self.optimizer_type = 'Adam'
        self.optimizer_other_args = {
                    #     'epsilon': 1e-5,
                    #  'clipnorm': 0.5,
                                    }
    @tf.function
    def call(self, obs):
        x = self.dense1(obs)
        x = self.dense2(x)
        value = self.value_layer(x)
        return (value,)

model_dict = {
    'actor': Actor,
    'critic': Critic
}

rl = RL(
    env_class=GymnasiumMaker,
    algo=PPOAlgo,
    hyper_params=param,
    model_dict=model_dict,
)

rl.run()

