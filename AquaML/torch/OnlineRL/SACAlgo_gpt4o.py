import torch
import torch.nn as nn
import torch.nn.functional as F
from AquaML.torch.TorchAlgoBase import TorchRLAlgoBase
from AquaML.param.ParamBase import RLParmBase
from AquaML import settings
from AquaML.torch.Dataset import RLDataSet
import numpy as np

class SACParam(RLParmBase):
    def __init__(self,
                 rollout_steps: int,
                 epoch: int,
                 batch_size: int,
                 ent_coef: float=0.2,
                 gamma: float=0.99,
                 tau: float=0.005,
                 lr: float=3e-4,
                 alpha: float=0.2,
                 reward_norm: bool=True,
                 target_update_interval: int=1,
                 env_num: int=1,
                 max_step: int=np.inf,
                 envs_args: dict={},
                 summary_steps: int=1000,
                 checkpoints_store_interval: int=5,
                 independent_model: bool=True,
                 ):
        super().__init__(
            rollout_steps=rollout_steps,
            epoch=epoch,
            gamma=gamma,
            env_num=env_num,
            max_step=max_step,
            envs_args=envs_args,
            independent_model=independent_model,
            summary_steps=summary_steps,
            algo_name='SAC',
            checkpoints_store_interval=checkpoints_store_interval,
        )
        self.ent_coef = ent_coef
        self.batch_size = batch_size
        self.lr = lr
        self.tau = tau
        self.alpha = alpha
        self.reward_norm = reward_norm
        self.target_update_interval = target_update_interval

class SACAlgo(TorchRLAlgoBase):
    def __init__(self, hyper_params, model_dict):
        super().__init__(hyper_params, model_dict)
        
        self._algo_name = 'SAC'
        
        self.actor = model_dict['actor']().to(settings.device)
        self.critic_1 = model_dict['critic_1']().to(settings.device)
        self.critic_2 = model_dict['critic_2']().to(settings.device)
        self.target_critic_1 = model_dict['critic_1']().to(settings.device)
        self.target_critic_2 = model_dict['critic_2']().to(settings.device)
        
        self._model_dict['actor'] = self.actor
        self._model_dict['critic_1'] = self.critic_1
        self._model_dict['critic_2'] = self.critic_2
        self._model_dict['target_critic_1'] = self.target_critic_1
        self._model_dict['target_critic_2'] = self.target_critic_2
        
        self.actor_optimizer, self.actor_optimizer_step_fn = self.create_optimizer(self.actor)
        self.critic_1_optimizer, self.critic_1_optimizer_step_fn = self.create_optimizer(self.critic_1)
        self.critic_2_optimizer, self.critic_2_optimizer_step_fn = self.create_optimizer(self.critic_2)
        
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
        self._action_info.add_info(
            name='log_prob',
            shape=self.actor.output_info.last_shape_dict['log_prob'],
            dtype=np.float32,
        )
        
        self.reward_mu = None
        self.reward_s = None
        self.reward_std = None
        self.epoch = 0

    def _train_action(self, state):
        input_data = []
        for name in self.actor.input_names:
            data = state[name]
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).to(settings.device)
            input_data.append(data)
        
        with torch.no_grad():
            actor_out_ = self.actor(*input_data)
            actor_out = dict(zip(self.actor.output_info.names, actor_out_))
            action = actor_out_['action']
            log_prob = actor_out_['log_prob']
        
        actor_out['action'] = action
        actor_out['log_prob'] = log_prob
        
        return actor_out, actor_out_['mu']

    def train_critic(self, critic_inputs, target):
        critic_1_out = self.critic_1(*critic_inputs)[0]
        critic_2_out = self.critic_2(*critic_inputs)[0]
        
        critic_1_loss = F.mse_loss(critic_1_out, target)
        critic_2_loss = F.mse_loss(critic_2_out, target)
        
        self.critic_1_optimizer_step_fn(critic_1_loss)
        self.critic_2_optimizer_step_fn(critic_2_loss)
        
        return {'critic_1_loss': critic_1_loss.detach(), 'critic_2_loss': critic_2_loss.detach()}

    def train_actor(self, actor_inputs, log_prob, q_value):
        actor_loss = (self.hyper_params.alpha * log_prob - q_value).mean()
        self.actor_optimizer_step_fn(actor_loss)
        return {'actor_loss': actor_loss.detach()}

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def train(self, data_dict: dict):
        data_set = RLDataSet(
            data_dict=data_dict,
            env_nums=settings.env_num,
            rollout_steps=self._hyper_params.rollout_steps,
            default_type='tensor',
            default_device=settings.device,
        )
        
        with torch.no_grad():
            next_action, next_log_prob = self._train_action(data_set.get_corresponding_data(prefix='next_'))
            next_q1 = self.target_critic_1(*data_set.get_corresponding_data(prefix='next_'))[0]
            next_q2 = self.target_critic_2(*data_set.get_corresponding_data(prefix='next_'))[0]
            next_q = torch.min(next_q1, next_q2) - self.hyper_params.alpha * next_log_prob
            target_q = data_dict['reward'] + self.hyper_params.gamma * next_q * (1 - data_dict['terminal'])
        
        critic_losses = self.train_critic(
            critic_inputs=data_set.get_corresponding_data(),
            target=target_q
        )
        
        for _ in range(self.hyper_params.update_times):
            for batch_data in data_set.get_batch(self.hyper_params.batch_size):
                actor_out, _ = self._train_action(batch_data)
                q1 = self.critic_1(*batch_data.get_corresponding_data())[0]
                q2 = self.critic_2(*batch_data.get_corresponding_data())[0]
                q_value = torch.min(q1, q2)
                actor_losses = self.train_actor(
                    actor_inputs=batch_data.get_corresponding_data(),
                    log_prob=actor_out['log_prob'],
                    q_value=q_value
                )
        
        if self.epoch % self.hyper_params.target_update_interval == 0:
            self.soft_update(self.target_critic_1, self.critic_1, self.hyper_params.tau)
            self.soft_update(self.target_critic_2, self.critic_2, self.hyper_params.tau)
        
        self.epoch += 1
        
        self.loss_tracker.reset()
        self.loss_tracker.add_data(critic_losses)
        self.loss_tracker.add_data(actor_losses)
        
        return self.loss_tracker