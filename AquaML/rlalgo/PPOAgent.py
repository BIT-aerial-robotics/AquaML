import numpy as np
import tensorflow as tf

from AquaML.rlalgo.BaseRLAgent import BaseAgent
from AquaML.core.AgentIOInfo import AgentIOInfo
from AquaML.rlalgo.AgentParameters import PPOAgentParameter

from AquaML.rlalgo.ExplorePolicy import GaussianExplorePolicy


class PPOAgent(BaseAgent):

    def __init__(self,
        name:str,
        actor,  
        critic,
        agent_info:AgentIOInfo,
        agent_params:PPOAgentParameter,
        level:int=0, # 控制是否创建不交互的agent
    ):
        
        super().__init__(
            name=name,
            agent_info=agent_info,
            agent_params=agent_params,
            level=level,
            )
        

        self.actor = actor()

        if self.level == 0:

            self.critic = critic()

            # 初始化critic网络
            self.initialize_critic()

            # 创建优化器
            # 检测actor是否包含优化器参数
            if hasattr(self.actor, 'optimizer_info'):
                self.actor_optimizer = self.create_optimizer(self.actor.optimizer_info)
            else:
                raise AttributeError(f'{self.actor.__class__.__name__} has no optimizer_info attribute')
            
            # 检测critic是否包含优化器参数
            if hasattr(self.critic, 'optimizer_info'):
                self.critic_optimizer = self.create_optimizer(self.critic.optimizer_info)
            else:
                raise AttributeError(f'{self.critic.__class__.__name__} has no optimizer_info attribute')

        # 创建探索策略
        if self.agent_params.explore_policy == 'Default':
            explore_name = 'Gaussian'
            log_std_init = {
                'log_std':self.agent_params.log_std_init_value
            }
        else:
            explore_name = self.agent_params.explore_policy
            log_std_init = {}

        self.create_explorer(
            explore_name=explore_name,
            shape=self.agent_info.action_shape,
            ponited_value=log_std_init,
        )

        # 确定resample_prob函数
        if 'log_std' in self._explore_dict:
            self.resample_prob = self._resample_log_prob_no_std
        else:
            self.resample_prob = self._resample_action_log_std

        # 获取actor优化参数
        self.actor_train_vars = self.actor.trainable_variables
        for key, value in self._tf_explore_dict.items():
            self.actor_train_vars = self.actor_train_vars + value.trainable_variables
            

        
    def train_critic(self, 
                     critic_inputs:tuple,
                     target:tf.Tensor,
                     ):
        
        with tf.GradientTape() as tape:
            tape.watch(self.critic.trainable_variables)
            critic_loss = tf.reduce_mean(tf.square(target - self.critic(critic_inputs)))
        
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        dic = {
            'critic_loss':critic_loss,
        }

        return dic
    
    def train_actor(self,
                    actor_inputs:tuple,
                    advantage:tf.Tensor,
                    old_log_prob:tf.Tensor,
                    action:tf.Tensor,
                    clip_ratio:float,
                    entropy_coef:float,
                    ):
        with tf.GradientTape() as tape:
            tape.watch(self.actor_train_vars)

            out = self.resample_prob(actor_inputs, action)

            log_prob = out[0]

            ratio = tf.exp(log_prob - old_log_prob)

            actor_surrogate_loss = tf.reduce_mean(
                tf.minimum(
                    ratio * advantage,
                    tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage,
                )
            )
            
            entropy_loss = -tf.reduce_mean(log_prob)

            actor_loss = actor_surrogate_loss + entropy_coef * entropy_loss

        actor_grads = tape.gradient(actor_loss, self.actor_train_vars)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_train_vars))

        dic = {
            'actor_loss':actor_loss,
            'actor_surrogate_loss':actor_surrogate_loss,
            'entropy_loss':entropy_loss,
        }

        return dic
            

    @property
    def actor_train_vars(self):
        return self._actor_train_vars


    def _resample_log_prob_no_std(self, obs, action):

        """
        Re get log_prob of action.
        The output of actor model is (mu,).
        It is different from resample_action.

        Args:
            obs (tuple): observation.
            action (tf.Tensor): action.
        """

        out = self.actor(*obs)
        mu = out[0]
        std = tf.exp(self.tf_log_std)
        log_prob = self.explore_policy.resample_prob(mu, std, action)

        return (log_prob, *out)
    
    
    def _resample_action_log_std(self, actor_obs: tuple):
        """
        Explore policy in SAC2 is Gaussian  exploration policy.

        _resample_action_log_std is used when actor model's out has log_std.

        The output of actor model is (mu, log_std).

        Args:
            actor_obs (tuple): actor model's input
        Returns:
        action (tf.Tensor): action
        log_pi (tf.Tensor): log_pi
        """

        out = self.actor(*actor_obs)

        mu, log_std = out[0], out[1]

        noise, prob = self.explore_policy.noise_and_prob(self.hyper_parameters.batch_size)

        sigma = tf.exp(log_std)

        action = mu + noise * sigma

        log_prob = tf.math.log(prob)

        return (action, log_prob)
