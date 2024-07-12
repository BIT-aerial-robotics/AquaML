import gym
import isaacgym
import isaacgymenvs

from numpy import inf
from AquaML.worker import RLEnvBase
import numpy as np
import torch
import os
from isaacgym import gymutil

from aerial_gym.envs import *
from aerial_gym.utils import task_registry

class RecordEpisodeStatisticsTorch(gym.Wrapper):
      def __init__(self, env, device):
            super().__init__(env)
            self.num_envs = getattr(env, "num_envs", 1)
            self.device = device
            self.episode_returns = None
            self.episode_lengths = None

      def reset(self, **kwargs):
            observations = super().reset(**kwargs)
            self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self.returned_episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            self.returned_episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            return observations
      

      # pdb.set_trace()
      def step(self, action):
            observations, privileged_observations, rewards, dones, infos = super().step(action)
            
            self.episode_returns += rewards
            self.episode_lengths += 1
            self.returned_episode_returns[:] = self.episode_returns
            self.returned_episode_lengths[:] = self.episode_lengths
            self.episode_returns *= 1 - dones
            self.episode_lengths *= 1 - dones
            infos["r"] = self.returned_episode_returns
            infos["l"] = self.returned_episode_lengths
            return (
                  observations,
                  rewards,
                  dones,
                  infos,
            )


class AerialGymMaker(RLEnvBase):
      def __init__(self, env_num: int, env_name:str):
            """
            AerialGymMaker的构造函数。

            Args:
                env_name (str): 环境的名称。
                env_args (dict): 环境的参数。
            """
            
            super(AerialGymMaker,self).__init__()
            
            # default_args = {
            #       'task': 'model',
            #       'seed' : 1,
            #       'experiment_name' : 'hovering',
            #       'checkpoint' : None,
            #       'headless' : False,
            #       'horovod' : False,
            #       'rl_device': 'cuda:0',
            #       'sim_device' : 'cuda:0',
            #       'num_envs' : 4096,
            #       "seed" : 1,
            #       'play' : False,
            #       'torch-deterministic-off' : False,
            #       'track' : False,
            #       'multi_gpu':False,
            #       'virtual_screen_capture':False,
            #       'force_render':False,
            #       'graphics_device_id':0
            # }
            
            def get_args():
                  custom_parameters = [
                        {"name": "--task", "type": str, "default": "F450", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
                        {"name": "--experiment_name", "type": str, "default": os.path.basename(__file__).rstrip(".py"), "help": "Name of the experiment to run or load. Overrides config file if provided."},
                        {"name": "--checkpoint", "type": str, "default": None, "help": "Saved model checkpoint number."},        
                        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
                        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
                        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
                        {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create. Overrides config file if provided."},
                        {"name": "--seed", "type": int, "default": 1, "help": "Random seed. Overrides config file if provided."},
                        {"name": "--play", "required": False, "help": "only run network", "action": 'store_true'},

                        {"name": "--torch-deterministic-off", "action": "store_true", "default": False, "help": "if toggled, `torch.backends.cudnn.deterministic=False`"},

                        {"name": "--track", "action": "store_true", "default": False,"help": "if toggled, this experiment will be tracked with Weights and Biases"},
                        {"name": "--wandb-project-name", "type":str, "default": "cleanRL", "help": "the wandb's project name"},
                        {"name": "--wandb-entity", "type":str, "default": None, "help": "the entity (team) of wandb's project"},

                        # Algorithm specific arguments
                        {"name": "--total-timesteps", "type":int, "default": 30000000000,
                              "help": "total timesteps of the experiments"},
                        {"name": "--learning-rate", "type":float, "default": 0.0026,
                              "help": "the learning rate of the optimizer"},
                        {"name": "--num-steps", "type":int, "default": 64,
                              "help": "the number of steps to run in each environment per policy rollout"},
                        {"name": "--anneal-lr", "action": "store_true", "default": False,
                              "help": "Toggle learning rate annealing for policy and value networks"},
                        {"name": "--gamma", "type":float, "default": 0.99,
                              "help": "the discount factor gamma"},
                        {"name": "--gae-lambda", "type":float, "default": 0.97,
                              "help": "the lambda for the general advantage estimation"},
                        {"name": "--num-minibatches", "type":int, "default": 2,
                              "help": "the number of mini-batches"},
                        {"name": "--update-epochs", "type":int, "default": 4,
                              "help": "the K epochs to update the policy"},
                        {"name": "--norm-adv-off", "action": "store_true", "default": False,
                              "help": "Toggles advantages normalization"},
                        {"name": "--clip-coef", "type":float, "default": 0.2,
                              "help": "the surrogate clipping coefficient"},
                        {"name": "--clip-vloss", "action": "store_true", "default": False,
                              "help": "Toggles whether or not to use a clipped loss for the value function, as per the paper."},
                        {"name": "--ent-coef", "type":float, "default": 0.0,
                              "help": "coefficient of the entropy"},
                        {"name": "--vf-coef", "type":float, "default": 2,
                              "help": "coefficient of the value function"},
                        {"name": "--max-grad-norm", "type":float, "default": 1,
                              "help": "the maximum norm for the gradient clipping"},
                        {"name": "--target-kl", "type":float, "default": None,
                              "help": "the target KL divergence threshold"},
                        ]

                  # parse arguments
                  args = gymutil.parse_arguments(
                        description="RL Policy",
                        custom_parameters=custom_parameters)
                  
                  args.batch_size = int(args.num_envs * args.num_steps)
                  args.minibatch_size = int(args.batch_size // args.num_minibatches)

                  args.torch_deterministic = not args.torch_deterministic_off
                  args.norm_adv = not args.norm_adv_off

                  # name allignment
                  args.sim_device_id = args.compute_device_id
                  args.sim_device = args.sim_device_type
                  if args.sim_device=='cuda':
                        args.sim_device += f":{args.sim_device_id}"
                  return args
                  
            args = get_args()
            
      
      # self.envs_args = env_args
      # self._env = isaacgymenvs.make(**default_args)
            
            envs , self._env_cfg= task_registry.make_env(name="quad", args=args) #task_registry.make -->return env, env_cfg
            self._env = RecordEpisodeStatisticsTorch(envs, args.rl_device)
            
            self._obs_info.add_info(
                        name='obs',
                        shape=13,
                        dtype=float,
                  )
                  
            self._rewards = ('reward',)
            
      def reset(self):
            """
            重置环境。

            returns:
                  observation(dict): 环境的观察值。
                  info：环境的信息。
            """

            obs = self._env.reset()
            obs = {"obs" : obs[0]}
            return obs, None
            
      def step(self, action):
            """
            执行动作。

            Args:
                  action (dict): 动作。
                      
            Returns:
                  tuple: observation, reward, terminated, truncated, info
            """

            obs, reward, done, info = self._env.step(action['action'])
            reward = torch.unsqueeze(torch.tensor(reward), axis=1)
            terminated = torch.unsqueeze(torch.tensor(done), axis=1)
            truncated = torch.zeros_like(done)
                  
            reward_dict = {
                  'reward': reward
            }
            obs = {"obs" : obs}      
            return obs, obs, reward_dict, terminated, truncated, info
            
      def close(self):
            """
            关闭环境。
            """
            self._env.close()
                  
      def auto_step(self, action_dict: dict, max_step: int = np.inf):
            return self.step(action_dict)
    
      def auto_reset(self):
            return self.reset()
      
      