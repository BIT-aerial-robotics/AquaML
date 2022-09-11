import sys

sys.path.append('../')
from AquaML.args.RLArgs import PPGHyperParam, EnvArgs, TaskArgs, TrainArgs
from AquaML.rlalgo.PPG import PhasicPolicyGradient as PPG
from AquaML.Tool.neural import mlp
from AquaML.Tool.RLTaskRunner import TaskRunner
from AquaML.Tool.GymEnvWrapper import GymEnvWrapper
from AquaML.policy.GaussianPolicy import GaussianPolicy
import gym
import os
import tensorflow as tf
import ModelExample

# class Model(tf.keras.Model):
#     def __init__(self):
#         super(Model).__init__()
#
#         self.model = tf.keras.Sequential([
#             tf.keras.layers.Dense(shape=(3,),units=)
#         ])


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))

env = gym.make("Pendulum-v1")
observation_dims = env.observation_space.shape[0]
action_dims = env.action_space.shape[0]

env_args = EnvArgs(
    max_steps=200,
    total_steps=4000,
    worker_num=1,
    episode_clip=50
)

algo_param = PPGHyperParam(
    epochs=120,
    batch_size=20,
    update_times=4,
    update_actor_times=4,
    update_critic_times=4,
    c1=0.02,
    c2=0.8,
)
training_args = TrainArgs(
    actor_is_batch_timesteps=True
)

# task_args = TaskArgs(
#     algo_param=algo_param,
#     obs_info={'obs': (observation_dims,)},
#     actor_inputs_info=list({'obs'}),
#     actor_outputs_info={'action': (action_dims,), 'prob': (action_dims,)},
#     critic_inputs_info=list({'obs'}),
#     reward_info=list({'total_reward'}),
#     distribution_info={'is_distribution': True},
#     env_args=env_args,
#     training_args=training_args,
# )

task_args = TaskArgs(
    algo_param=algo_param,
    obs_info={'obs': (observation_dims,), 'pos': (2,)},
    actor_inputs_info=list({'pos'}),
    actor_outputs_info={'action': (action_dims,), 'prob': (action_dims,), 'joint_loss': (1,)},
    critic_inputs_info=list({'obs'}),
    reward_info=list({'total_reward'}),
    distribution_info={'is_distribution': True},
    env_args=env_args,
    training_args=training_args,
)

critic = mlp(
    state_dims=observation_dims,
    output_dims=1,
    hidden_size=(64, 64),
    name='value'
)

# actor = mlp(
#     state_dims=observation_dims,
#     output_dims=1,
#     hidden_size=(32, 32),
#     name='actor',
#     output_activation='tanh'
# )

actor = ModelExample.LSTMActorValue1(2)
actor.build((1, 1, 2))
# actor(np.array([1, 1]))
actor_policy = GaussianPolicy(actor, name='actor', reset_flag=True)

task_runner = TaskRunner(
    task_args=task_args,
    actor_policy=actor_policy,
    critic=critic,
    algo=PPG,
    work_space='PPG_debug',
    env=GymEnvWrapper('Pendulum-v1'),
)

if __name__ == '__main__':
    task_runner.run()
