import numpy as np
import tensorflow as tf

from AquaML.rlalgo.BaseRLAgent import BaseRLAgent
from AquaML.rlalgo.AgentParameters import PPOAgentParameter
from AquaML.core.Comunicator import Communicator
from AquaML.core.RLToolKit import RLStandardDataSet
from AquaML.buffer.RLPrePlugin import ValueFunctionComputer, GAEComputer, SplitTrajectory


class PPOAgent(BaseRLAgent):

    def __init__(self,
                 name: str,
                 actor,

                 agent_params: PPOAgentParameter,
                 level: int = 0,  # 控制是否创建不交互的agent
                 critic=None,
                 ):

        super().__init__(
            name=name,
            agent_params=agent_params,
            level=level,
        )

        self._episode_tool = None
        self.actor = actor()

        if critic is None:
            self.critic = self.actor
            self.model_type = 'share'
        else:
            self.critic = critic()
            self.model_type = 'independent'

    def init(self):
        self.initialize_actor()
        if self.level == 0:

            # 创建优化器
            # 检测actor是否包含优化器参数
            if hasattr(self.actor, 'optimizer_info'):
                self.actor_optimizer = self.create_optimizer(self.actor.optimizer_info)
            else:
                raise AttributeError(f'{self.actor.__class__.__name__} has no optimizer_info attribute')

            # 初始化critic网络
            if self.model_type == 'share':
                if 'log_std' in self.actor.output_info:
                    self.value_idx = 2
                else:
                    self.value_idx = 1

                self._all_model_dict = {
                    'actor': self.actor,
                }
            else:

                # 检测critic是否包含优化器参数
                if hasattr(self.critic, 'optimizer_info'):
                    self.critic_optimizer = self.create_optimizer(self.critic.optimizer_info)
                else:
                    raise AttributeError(f'{self.critic.__class__.__name__} has no optimizer_info attribute')

                self._all_model_dict = {
                    'actor': self.actor,
                    'critic': self.critic,
                }
                self.initialize_critic()

            ########################################
            # 创建buffer,及其计算插件
            ########################################
            self.actor_plugings_dict = {}

            if self.agent_params.min_steps <= 1:
                filter_name = None
                filter_args = {}
            else:
                filter_name = 'len'
                filter_args = {
                    'len_threshold': self.agent_params.min_steps
                }

            # 创建episode处理工具
            self._episode_tool = SplitTrajectory(
                filter_name=filter_name,
                filter_args=filter_args,
            )

            # 为tool添加处理插件
            adv_td_error = GAEComputer(
                gamma=self.agent_params.gamma,
                lamda=self.agent_params.lamda
            )

            value_dfn = ValueFunctionComputer(
                self.critic
            )

            self._episode_tool.add_plugin(
                value_dfn,
            )

            self._episode_tool.add_plugin(
                adv_td_error
            )

        # 创建探索策略
        if self.agent_params.explore_policy == 'Default':
            explore_name = 'Gaussian'
            log_std_init = {
                'log_std': self.agent_params.log_std_init_value
            }
        else:
            explore_name = self.agent_params.explore_policy
            log_std_init = {}

        self.create_explorer(
            explore_name=explore_name,
            shape=self.actor.output_info['action'],
            pointed_value=log_std_init,
        )

        # 确定resample_prob函数
        if 'log_std' in self._explore_dict:
            self.resample_prob = self._resample_log_prob_no_std
        else:
            self.resample_prob = self._resample_log_prob_log_std

        # 获取actor优化参数
        self._actor_train_vars = self.actor.trainable_variables
        for key, value in self._tf_explore_dict.items():
            self._actor_train_vars += [value]

        # 初始化模型同步器
        self._sync_model_dict = {
            'actor': self.actor,
        }

    @tf.function
    def train_critic(self,
                     critic_inputs: tuple,
                     target: tf.Tensor,
                     ):

        with tf.GradientTape() as tape:
            tape.watch(self.critic.trainable_variables)
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
                    old_log_prob: tf.Tensor,
                    action: tf.Tensor,
                    clip_ratio: float,
                    entropy_coef: float,
                    ):
        old_log_prob = tf.math.log(old_log_prob)
        with tf.GradientTape() as tape:
            tape.watch(self.actor_train_vars)

            out = self.resample_prob(actor_inputs, action)

            log_prob = out[0]
            log_std = out[1]
            mu = out[2]

            ratio = tf.exp(log_prob - old_log_prob)

            actor_surrogate_loss = tf.reduce_mean(
                tf.minimum(
                    ratio * advantage,
                    tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage,
                )
            )

            entropy_loss = tf.reduce_mean(self.explore_policy.get_entropy(mu, log_std))

            actor_loss = -actor_surrogate_loss - entropy_coef * entropy_loss

        actor_grads = tape.gradient(actor_loss, self.actor_train_vars)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_train_vars))

        dic = {
            'actor_loss': actor_loss,
            'actor_surrogate_loss': actor_surrogate_loss,
            'entropy_loss': entropy_loss,
        }

        return dic

    def train_shared(self,
                     target: tf.Tensor,
                     actor_inputs: list,
                     advantage: tf.Tensor,
                     old_log_prob: tf.Tensor,
                     action: tf.Tensor,
                     clip_ratio: float,
                     entropy_coef: float,
                     vf_coef: float,
                     ):
        old_log_prob = tf.math.log(old_log_prob)
        with tf.GradientTape() as tape:
            tape.watch(self.actor_train_vars)

            out = self.resample_prob(actor_inputs, action)

            log_prob = out[0]
            log_std = out[1]
            mu = out[2]
            value = out[self.value_idx + 2]

            ratio = tf.exp(log_prob - old_log_prob)

            actor_surrogate_loss = tf.reduce_mean(
                tf.minimum(
                    ratio * advantage,
                    tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage,
                )
            )

            entropy_loss = tf.reduce_mean(self.explore_policy.get_entropy(mu, log_std))

            value_loss = tf.reduce_mean(tf.square(target - value))

            total_loss = -actor_surrogate_loss - entropy_coef * entropy_loss + vf_coef * value_loss

        actor_grads = tape.gradient(total_loss, self.actor_train_vars)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_train_vars))

        dic = {
            'total_loss': total_loss,
            'actor_surrogate_loss': actor_surrogate_loss,
            'entropy_loss': entropy_loss,
            'value_loss': value_loss,
        }

        return dic

    @property
    def actor_train_vars(self):
        return self._actor_train_vars

    def optimize(self, data_set: RLStandardDataSet):

        # 检查当前是否为主线程
        if self.level != 0:
            raise RuntimeError('Only main agent can optimize')

        train_data, reward_info = self._episode_tool(data_set)

        for _ in range(self.agent_params.update_times):
            for batch_data in train_data(self.agent_params.batch_size):
                actor_input_obs = []
                critic_input_obs = []

                for name in self.actor.input_name:
                    actor_input_obs.append(batch_data[name])

                advantage = batch_data['advantage']

                if self.agent_params.batch_advantage_normalization:
                    advantage = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-8)

                if self.model_type == 'share':
                    shared_optimize_info = self.train_shared(
                        actor_inputs=actor_input_obs,
                        target=batch_data['target'],
                        advantage=advantage,
                        old_log_prob=batch_data['prob'],
                        action=batch_data['action'],
                        clip_ratio=self.agent_params.clip_ratio,
                        entropy_coef=self.agent_params.entropy_coef,
                        vf_coef=self.agent_params.vf_coef,
                    )
                    self.loss_tracker.add_data(shared_optimize_info, prefix='shared')
                else:

                    for name in self.critic.input_name:
                        critic_input_obs.append(batch_data[name])

                    for _ in range(self.agent_params.update_critic_times):
                        critic_optimize_info = self.train_critic(
                            critic_inputs=critic_input_obs,
                            target=batch_data['target'],
                        )
                        self.loss_tracker.add_data(critic_optimize_info, prefix='critic')

                    for _ in range(self.agent_params.update_actor_times):
                        actor_optimize_info = self.train_actor(
                            actor_inputs=actor_input_obs,
                            advantage=advantage,
                            old_log_prob=batch_data['prob'],
                            action=batch_data['action'],
                            clip_ratio=self.agent_params.clip_ratio,
                            entropy_coef=self.agent_params.entropy_coef,
                        )
                        self.loss_tracker.add_data(actor_optimize_info, prefix='actor')

        summary = self.loss_tracker.get_data()

        del data_set

        return summary, reward_info

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

        return (log_prob, self.tf_log_std, *out)

    def _resample_log_prob_log_std(self, obs: tuple, action):
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

        out = self.actor(*obs)

        mu, log_std = out[0], out[1]

        std = tf.exp(log_std)

        log_prob = self.explore_policy.resample_prob(mu, std, action)

        return (log_prob, log_std, *out)

    @staticmethod
    def get_algo_name():
        return 'PPO'

    def get_real_policy_out(self):

        out_list = []

        for name in self.actor.output_info.keys():
            # if 'hidden' not in name:
            out_list.append(name)
        for name in self.explore_policy.get_aditional_output.keys():
            out_list.append(name)
        return out_list
