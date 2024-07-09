import tensorflow as tf
import tensorflow_probability as tfp
from AquaML.tf.TFAlgoBase import TFRLAlgoBase
from AquaML.param.ParamBase import RLParmBase
from AquaML import logger, settings, data_module, communicator
from AquaML.algo.ModelBase import ModelBase
import numpy as np
from AquaML.tf.Dataset import RLDataset
import copy
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

class PPOAlgo(TFRLAlgoBase):
    
    def __init__(self, hyper_params:PPOParam,
                 model_dict:dict
                 ):
        """
        PPO算法。
        
        Args:
            hyper_params (PPOParam): PPO算法的超参数。
            model_dict (dict): 模型字典。
        """
        super().__init__(hyper_params, model_dict)
        
        self._algo_name = 'PPO' # 首先声明不要忘记
        
        self.actor:ModelBase = model_dict['actor']()
        self.critic:ModelBase = model_dict['critic']()
        

        
        ############################
        # 1. 初始化参数和接口
        ############################
        
        # 模型接口
        self._model_dict['actor'] = self.actor
        self._model_dict['critic'] = self.critic
        
        # 分别为actor和critic创建优化器
        # self.actor_optimizer = self.create_optimizer(self.actor)
        # self.critic_optimizer = self.create_optimizer(self.critic)
        
        # 创建额外可优化参数
        action_shape = self.actor.output_info.last_shape_dict['action']
        self._tf_log_std = tf.Variable(tf.ones(action_shape) * hyper_params.log_std, dtype=tf.float32,trainable=True)
        
        # self._actor_train_vars = self.actor.trainable_variables + [self._tf_log_std]
        
        # TODO:将_tf_log_std参数放入DataModule中，可以进行共享
        
        ############################
        # 2. 创建高斯分布
        ############################
        mu = tf.zeros(action_shape)
        sigma = tf.ones(action_shape)
        self._dist = tfp.distributions.Normal(mu, sigma)
        
        ############################
        # 3. PPO算法额外需要的数据
        ############################
        self._action_info.add_info(
            name='prob',
            shape=action_shape,
            dtype=np.float32,
        )
        
        self._fix_log = math.log(2 * math.pi)

        
        self.reward_mu = None
        self.reward_s = None
        self.reward_std = None
        
        self.epoch = 0
        # self._action_size = settings.env_num
        
    
    @tf.function
    def noise_and_prob(self, batch_size=1):
        noise = self._dist.sample(batch_size)
        prob = self._dist.prob(noise)

        return noise, prob
    
    @tf.function
    def resample_prob(self, mu, std, action, sum_axis=1):
        # sigma = tf.exp(log_std)
        noise = (action - mu) / std
        log_prob = self._dist.log_prob(noise)

        log_prob = tf.reduce_sum(log_prob, axis=sum_axis, keepdims=True)

        # dist = tfp.distributions.Normal(loc=mu, scale=std)
        # log_prob = dist.log_prob(action)

        return log_prob
        
    
    def _train_action(self, state):

        # state = copy.deepcopy(state)
        
        require_list = []
        
        for name in self.actor.input_names:
            require_list.append(state[name])
        
        actor_out_ = self.actor(*require_list)
        
        actor_out = dict(zip(self.actor.output_info.names, actor_out_))
        
        mu = actor_out_[0]
        
        noise, prob = self.noise_and_prob(self._action_size)
        
        # noise = self._dist.sample(self._action_size)
        # log_prob = self._dist.log_prob(noise)
        
        action = mu + noise * tf.exp(self._tf_log_std)
        
        actor_out['action'] = action
        actor_out['prob'] = prob
        
        return actor_out,mu
        
    
    @tf.function
    def train_critic(self,
                     critic_inputs: tuple,
                     target: tf.Tensor,
                     ):

        with tf.GradientTape() as tape:
            # tape.watch(self.critic.trainable_variables)
            critic_loss = tf.reduce_mean(tf.square(target - self.critic(*critic_inputs)))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        dic = {
            'critic_loss': critic_loss,
        }

        return dic


    
    @tf.function
    def train_actor(self,
                    actor_inputs: tuple,
                    advantage: tf.Tensor,
                    old_prob: tf.Tensor,
                    action: tf.Tensor,
                    clip_ratio: float,
                    entropy_coef: float,
                    ):
        
        old_log_prob = tf.reduce_sum(tf.math.log(old_prob), axis=1, keepdims=True)
        
        with tf.GradientTape() as tape:
            tape.watch(self.actor.trainable_variables + [self._tf_log_std])
            
            ############################
            # 1. 计算重要性采样比率
            ############################
            mu = self.actor(*actor_inputs)[0]
            std = tf.exp(self._tf_log_std)
            
            log_prob = self.resample_prob(std=std, mu=mu, action=action, sum_axis=1)
            
            importance_ratio = tf.exp(log_prob - old_log_prob)
            
            ############################
            # 2. 计算surr Loss
            ############################
            surr1 = importance_ratio * advantage
            surr2 = tf.clip_by_value(importance_ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
            surr = tf.minimum(surr1, surr2)
            
            surr_loss = tf.reduce_mean(surr)
            
            ############################
            # 3. 计算熵
            ############################
            entropy = 0.5 * (1.0 + tf.math.log(2.0 * np.pi)) + self._tf_log_std
            entropy_loss = tf.reduce_sum(entropy,keepdims=True)
            
            actor_loss = -surr_loss - entropy_coef * entropy_loss
            
        actor_grads = tape.gradient(actor_loss,self.actor.trainable_variables + [self._tf_log_std])
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables + [self._tf_log_std]))
            
        loss_dict = {
            'actor_loss': actor_loss,
            'surr_loss': surr_loss,
            'entropy_loss': entropy_loss,
        }
        
        return loss_dict,importance_ratio

    
    def train(self, data_dict:dict):
        
        data_set = RLDataset(
            data_dict=data_dict,
            env_nums=settings.env_num,
            rollout_steps=self.hyper_params.rollout_steps,
        )
        
        ############################
        # 1. 计算GAE
        ############################
        
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
        value = self.critic(*critic_inputs)[0]
        next_value = self.critic(*next_citic_inputs)[0]

        # distance = next_value[:,-1,] - value[:,1:]
        
        # 获取reward和mask
        rewards = data_dict['reward']

        
        # masks = data_dict['mask']
        if self._hyper_params.reward_norm:
            if self.reward_mu is None:
                self.reward_mu = np.mean(rewards)
                self.reward_std = np.std(rewards)
                self.reward_s = np.sqrt(self.reward_std)
            else:
                # new_std = rewards.std()
                new_mean = rewards.mean()
                old_mean = copy.deepcopy(self.reward_mu)
                self.reward_mu = (self.reward_mu * self.epoch  + new_mean) / (self.epoch + 1)
                self.reward_s = self.reward_s + (new_mean - old_mean) * (new_mean - self.reward_mu) 
                self.reward_std = np.sqrt(self.reward_s / (self.epoch + 1))
                
            rewards = (rewards - self.reward_mu) / self.reward_std
            self.epoch += 1
        # truncated = data_dict['truncated']
        terminateds = data_dict['terminal']
        done = 1 - terminateds
             
        gae = np.zeros_like(rewards)
        n_steps_target = np.zeros_like(rewards)
        cumulated_advantage = np.zeros_like(rewards[:, 0])
        # for env_num in range(settings.env_num):
        for step in range(self.hyper_params.rollout_steps):
            reversed_step = self.hyper_params.rollout_steps - step - 1
            
            
            delta = rewards[:, reversed_step] + self.hyper_params.gamma * next_value[:, reversed_step] * done[:, reversed_step] - value[:, reversed_step]
            cumulated_advantage = self.hyper_params.gamma * self.hyper_params.lamda * cumulated_advantage * done[:, reversed_step] + delta
            gae[:, reversed_step] = cumulated_advantage
            n_steps_target[:, reversed_step] = gae[:, reversed_step] + value[:, reversed_step]
                
        data_set.add_data('advantage', gae)
        data_set.add_data('target', n_steps_target)
        data_set.add_data('value',value)
        data_set.add_data('next_value', next_value)
        
        ############################
        # 2. 将数据shape转换为(steps, dims)
        ############################
        data_set.concat(axis=0)
        # data_set.reshape(-1)

        # distance1 = data_set['obs'][1:] - data_set['next_obs'][:-1]
        # distance2 = data_set['value'][1:] - data_set['next_value'][:-1]

        ############################
        # 3. 训练actor和critic
        ############################
        self.loss_tracker.reset()
        
        early_stop = False
        
        for _ in range(self.hyper_params.update_times):
            
            if early_stop:
                break
        
            for batch_data in data_set.get_batch(self.hyper_params.batch_size):
                
                critic_inputs = batch_data.get_corresponding_data(
                    names=self.critic.input_names,
                )
                
                actor_inputs = batch_data.get_corresponding_data(
                    names=self.actor.input_names,
                )
                
                advantage = batch_data['advantage']
                old_prob = batch_data['prob']
                action = batch_data['action']
                
                critic_loss = self.train_critic(
                    critic_inputs=critic_inputs,
                    target=batch_data['target'],
                )
                
                actor_loss, ratio = self.train_actor(
                    actor_inputs=actor_inputs,
                    advantage=advantage,
                    old_prob=old_prob,
                    action=action,
                    clip_ratio=self.hyper_params.clip_ratio,
                    entropy_coef=self.hyper_params.ent_coef,
                )
                
                
                self.loss_tracker.add_data(critic_loss)
                self.loss_tracker.add_data(actor_loss)
                
                log_ratio = np.log(ratio)
                approx_kl = np.mean(np.exp(log_ratio)- 1 - log_ratio)
                self.loss_tracker.add_data({'approx_kl': approx_kl})
                
                if self._hyper_params.target_kl is not None:
                    if approx_kl > 1.5 * self._hyper_params.target_kl:
                        # logger.info(f'Early stopping at epoch {_} due to reaching max kl')
                        early_stop = True
                        break

        
        return self.loss_tracker
        
        
        