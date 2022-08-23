import sys

sys.path.append('../')
from AquaML.args.RLArgs import PPOHyperParam, EnvArgs, TaskArgs, TrainArgs
from AquaML.rlalgo.PPO import ProximalPolicyOptimization as PPO
from AquaML.Tool.neural import mlp
from AquaML.Tool.MPIRLTaskRunner import TaskRunner
from AquaML.Tool.GymEnvWrapper import GymEnvWrapper
from AquaML.policy.GaussianPolicy import GaussianPolicy

import gym
import os
import tensorflow as tf
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for k in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[k], True)
            print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
    else:
        print("Not enough GPU hardware devices available")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# class Model(tf.keras.Model):
#     def __init__(self):
#         super(Model).__init__()
#
#         self.model = tf.keras.Sequential([
#             tf.keras.layers.Dense(shape=(3,),units=)
#         ])


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    total_steps=2000,
    worker_num=size-1
)

algo_param = PPOHyperParam(
    epochs=100,
    batch_size=64,
    update_times=4
)

training_args = TrainArgs()

task_args = TaskArgs(
    algo_param=algo_param,
    obs_info={'obs': (observation_dims,)},
    actor_inputs_info=list({'obs'}),
    actor_outputs_info={'action': (action_dims,), 'prob': (action_dims,)},
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

actor = mlp(
    state_dims=observation_dims,
    output_dims=1,
    hidden_size=(32, 32),
    name='actor',
    output_activation='tanh'
)

actor_policy = GaussianPolicy(actor, name='actor')

task_runner = TaskRunner(
    task_args=task_args,
    actor_policy=actor_policy,
    critic=critic,
    algo=PPO,
    work_space='test2',
    env=GymEnvWrapper('Pendulum-v1'),
    comm=comm
)

task_runner.run()
