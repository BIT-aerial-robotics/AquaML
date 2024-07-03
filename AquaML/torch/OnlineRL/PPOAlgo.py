import torch
from AquaML.torch.TorchAlgoBase import TorchRLAlgoBase
from AquaML.param.ParamBase import RLParmBase
from AquaML import logger, settings, data_module, communicator
import numpy as np
from AquaML.torch.Dataset import RLDataSet
import math

class PPOParam(RLParmBase):
    def __init__(self, 
                 rollout_steps: int, 
                 epoch: int,
                 batch_size: int,
                 ent_coef: float=0.0,
                 clip_ratio: float=0.2,
                 update_times: int=4,
                 gamma: float=0.99,
                 summary_steps: int=1000,
                 env_num: int=1,
                 max_step: int=np.inf,
                 envs_args: dict={},
                 lamda: float=0.95,
                 log_std: float = -0.0,
                 reward_norm: bool = True,
                 target_kl: float = 0.01,
                 checkpoints_store_interval: int = 5,
                 independent_model: bool = True,
                 ):
        """
        PPO算法的超参数。

        Args:
            rollout_steps (int): 每次rollout的步数。
            epoch (int): 训练的epoch数。
            batch_size (int): 每次训练的batch大小,推荐为总样本数目的1/4。
            update_times (int, optional): 一次采集的数据更新的次数. 默认为4.
            gamma (float, optional): 折扣因子. 默认为0.99.
            env_num (int, optional): 每个进程环境的数量. 默认为1.
            max_step (int, optional): 最大步数. 默认为np.inf.
            envs_args (dict, optional): 环境的参数. 默认为{}.
            lamda (float, optional): GAE的lamda参数. 默认为0.95.
            log_std (float, optional): 高斯分布的标准差. 默认为-0.0.
            independent_model (bool, optional): 确认每个进程中是否有独立的模型。当为True时，每个进程中都有独立的模型。当为False时，所有进程共享一个模型. 默认为True.
        """
        super().__init__(
            rollout_steps=rollout_steps,
            epoch=epoch,
            gamma=gamma,
            env_num=env_num,
            max_step=max_step,
            envs_args=envs_args,
            independent_model=independent_model,
            summary_steps=summary_steps,
            algo_name='PPO',
            checkpoints_store_interval=checkpoints_store_interval,
        )
        
        self.log_std = log_std
        self.update_times = update_times
        self.lamda = lamda
        self.batch_size = batch_size
        self.clip_ratio = clip_ratio
        self.ent_coef = ent_coef
        self.reward_norm = reward_norm
        self.target_kl = target_kl
        
class PPOAlgo(TorchRLAlgoBase):
    
    def __init__(self, hyper_params, model_dict):
        """
        PPO算法。

        Args:
            hyper_params (PPOParam): PPO算法的超参数。
            model_dict (dict): 模型字典。
        """
        super().__init__(hyper_params, model_dict)
        
        self._algo_name = 'PPO'
        
        self.actor = model_dict['actor']().to(settings.device)
        self.critic = model_dict['critic']().to(settings.device)
        
        
        ##############################
        # 1. 初始化数据集
        ##############################
        
         # 模型接口
        self._model_dict['actor'] = self.actor
        self._model_dict['critic'] = self.critic
        
        # 创建额外的参数
        action_shape = self.actor.output_info.last_shape_dict['action']
        self._torch_log_std = torch.nn.Parameter(torch.ones(action_shape,dtype=torch.float32,device=settings.device) * hyper_params.log_std).to(settings.device)
        # self._torch_log_std.detach()
        self._torch_log_std.requires_grad = True
        
        
        # 创建优化器
        self.actor_optimizer, self.actor_optimizer_step_fn = self.create_optimizer(self.actor,other_params=self._torch_log_std)
        self.critic_optimizer, self.critic_optimizer_step_fn = self.create_optimizer(self.critic)
        
        ##############################
        # 2. 创建高斯分布
        ##############################
        mu = torch.zeros(action_shape, dtype=torch.float32).to(settings.device)
        sigma = torch.ones(action_shape, dtype=torch.float32).to(settings.device)
        self._dist = torch.distributions.Normal(mu, sigma)
        
        ##############################
        # PPO算法需要的额外数据
        ##############################
        self._action_info.add_info(
            name='log_prob',
            shape=action_shape,
            dtype=np.float32,
        )
        
        self._fix_log = math.log(2 * math.pi)
        self._torch_fix_log = torch.tensor(self._fix_log, dtype=torch.float32, device=settings.device)
        
        self.reward_mu = None
        self.reward_s = None
        self.reward_std = None
        
        self.epoch = 0
        
    def get_action(self, state):
        
        input_data = []
        
        for name in self.actor.input_names:
            data = state[name]
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).to(settings.device)
            input_data.append(data)
        
        # self.actor.eval()
            
        with torch.no_grad():
            actor_out_ = self.actor(*input_data)
        
            actor_out = dict(zip(self.actor.output_info.names, actor_out_))
        
            mu = actor_out_[0]
        
            noise = self._dist.sample((self._action_size,))
            log_prob = self._dist.log_prob(noise)
        
            action = mu + noise * torch.exp(self._torch_log_std)
        
        actor_out['action'] = action
        actor_out['log_prob'] = log_prob
        
        return actor_out
    
    def train_critic(self,
                     critic_inputs:tuple,
                     target:torch.Tensor,
                     ):
        
        critic_out = self.critic(*critic_inputs)[0]
        critic_loss = torch.nn.functional.mse_loss(critic_out, target)
        # critic_loss.backward()
        self.critic_optimizer_step_fn(critic_loss)
        
        dic = {
            'critic_loss':critic_loss.detach(),
        }
        
        return dic
    
    def train_actor(self,
                    actor_inputs:tuple,
                    advantage:torch.Tensor,
                    old_log_prob:torch.Tensor,
                    action:torch.Tensor,
                    clip_ratio:float,
                    ent_coef:float,
                    ):
        
        old_log_prob = torch.sum(old_log_prob,dim=-1,keepdim=True)
        
        ##############################
        # 1.计算actor_loss
        ##############################
        mu = self.actor(*actor_inputs)[0]
        std = torch.exp(self._torch_log_std)
        
        noise = (action - mu) / std
        log_prob = self._dist.log_prob(noise)
        log_prob = torch.sum(log_prob,dim=-1,keepdim=True)
        
        importance_ratio = torch.exp(log_prob - old_log_prob)
        
        ##############################
        # 2.计算surrogate loss
        ##############################
        surrogate_loss1 = importance_ratio * advantage
        surrogate_loss2 = torch.clamp(importance_ratio, 1-clip_ratio, 1+clip_ratio) * advantage
        surrogate_loss = torch.min(surrogate_loss1, surrogate_loss2).mean()
        
        ##############################
        # 3.计算entropy loss
        ##############################
        entropy = 0.5 * (1.0 + self._torch_fix_log) + self._torch_log_std
        entropy_loss = torch.sum(entropy)
        
        actor_loss = -surrogate_loss - ent_coef * entropy_loss
        
        # actor_loss.backward()
        self.actor_optimizer_step_fn(actor_loss)
        
        dic = {
            'actor_loss':actor_loss.detach(),
            'surrogate_loss':surrogate_loss.detach(),
            'entropy_loss':entropy_loss.detach(),
        }
        
        return dic, importance_ratio
    
    def train(self, data_dict: dict):
        
        data_set = RLDataSet(
            data_dict=data_dict,
            env_nums=settings.env_num,
            rollout_steps=self._hyper_params.rollout_steps,
            default_type='tensor',
            default_device=settings.device,
        )
        
        
        
        ##############################
        #  1. 计算GAE
        ##############################
        
        # 获取critic的输入
        critic_inputs = data_set.get_corresponding_data(
            names=self.critic.input_names,
        )
        
        next_citic_inputs = data_set.get_corresponding_data(
            names=self.critic.input_names,
            prefix='next_',
        )
        
        # 计算value和next_value
        # TODO:当前可能会出现内存爆炸。
        with torch.no_grad():
            value = self.critic(*critic_inputs)[0]
            next_value = self.critic(*next_citic_inputs)[0]
        
               # 获取reward和mask
        rewards = data_dict['reward']
        
        # 处理reward
        if self._hyper_params.reward_norm:
            if self.reward_mu is None:
                self.reward_mu = torch.mean(rewards)
                self.reward_std = rewards.std()
                self.reward_s = self.reward_std.pow(2)
            else:
                # new_std = rewards.std()
                new_mean = rewards.mean()
                old_mean = self.reward_mu.clone()
                self.reward_mu = (self.reward_mu * self.epoch  + new_mean) / (self.epoch + 1)
                self.reward_s = self.reward_s + (new_mean - old_mean) * (new_mean - self.reward_mu) 
                self.reward_std = torch.sqrt(self.reward_s / (self.epoch + 1))
                
            rewards = (rewards - self.reward_mu) / self.reward_std
            self.epoch += 1
        # masks = data_dict['mask']
        terminateds = data_dict['terminal']
        # truncated = data_dict['truncated']

        done = 1 - terminateds
             
        gae = torch.zeros_like(rewards).to(settings.device)
        n_steps_target = torch.zeros_like(rewards).to(settings.device)
        cumulated_advantage = torch.zeros_like(rewards[:, 0]).to(settings.device)
        # np.zeros_like(rewards[:, 0])
        # for env_num in range(settings.env_num):
        # self.actor.train()
        # self.critic.train()
        for step in range(self.hyper_params.rollout_steps):
            reversed_step = self.hyper_params.rollout_steps - step - 1
            
            
            delta = rewards[:, reversed_step] + self.hyper_params.gamma * next_value[:, reversed_step] * done[:, reversed_step] - value[:, reversed_step]
            cumulated_advantage = self.hyper_params.gamma * self.hyper_params.lamda * cumulated_advantage * done[:, reversed_step] + delta
            gae[:, reversed_step] = cumulated_advantage
            n_steps_target[:, reversed_step] = gae[:, reversed_step] + value[:, reversed_step]
                
        data_set.add_data('advantage', gae)
        data_set.add_data('target', n_steps_target)
        
        ##############################
        # 2. 将数据shape转换为(steps, dims)
        ##############################
        # data_set.concat(axis=0)
        data_set.reshape(shape=(gae.shape[0]*gae.shape[1], -1))

        # distance1 = data_set['obs'][1:] - data_set['next_obs'][:-1]
        
        ##############################
        # 3. 训练actor和critic
        ##############################
        self.loss_tracker.reset()
        
        early_stop = False
        
        for _ in range(self.hyper_params.epoch):
            if early_stop:
                break
            for bath_data in data_set.get_batch(self.hyper_params.batch_size):
                
                critic_inputs = bath_data.get_corresponding_data(
                    names=self.critic.input_names,
                )
                
                actor_inputs = bath_data.get_corresponding_data(
                    names=self.actor.input_names,
                )
                
                advantage = bath_data['advantage']
                old_log_prob = bath_data['log_prob']
                action = bath_data['action']
                
                critic_loss = self.train_critic(
                    critic_inputs=critic_inputs,
                    target=bath_data['target'],
                )
                
                actor_loss,ratio= self.train_actor(
                    actor_inputs=actor_inputs,
                    advantage=advantage,
                    old_log_prob=old_log_prob,
                    action=action,
                    clip_ratio=self.hyper_params.clip_ratio,
                    ent_coef=self.hyper_params.ent_coef,
                )
                
                self.loss_tracker.add_data(critic_loss)
                self.loss_tracker.add_data(actor_loss)
                
                # early stopping
                log_ratio = torch.log(ratio).detach()
                approx_kl = (log_ratio.exp() - 1 - log_ratio).mean()
                self.loss_tracker.add_data({'approx_kl':approx_kl.detach()})
                
                if self._hyper_params.target_kl is not None:
                    if approx_kl > 1.5 * self._hyper_params.target_kl:
                        # logger.info(f'Early stopping at epoch {_} due to reaching max kl')
                        early_stop = True
                        break
                
                
                
        
        # self.actor.eval()
        # self.critic.eval()
                
        return self.loss_tracker