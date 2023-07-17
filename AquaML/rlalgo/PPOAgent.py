import numpy as np
import tensorflow as tf

from AquaML.rlalgo.BaseRLAgent import BaseRLAgent
from AquaML.rlalgo.AgentParameters import PPOAgentParameter
from AquaML.core.Comunicator import Communicator
from AquaML.core.RLToolKit import RLStandardDataSet
from AquaML.buffer.RLPrePlugin import ValueFunctionComputer, GAEComputer, SplitTrajectory
import copy
import tensorflow_probability as tfp


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
                self.create_optimizer(self.actor.optimizer_info, 'actor_optimizer')
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
                    self.create_optimizer(self.critic.optimizer_info, 'critic_optimizer')
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

        if self.level == 0:
            ########################################
            # 获取训练参数
            ########################################
            self._actor_train_vars = self.actor.trainable_variables
            for key, value in self._tf_explore_dict.items():
                self._actor_train_vars += [value]

            if 'critic' in self._all_model_dict:
                self._critic_train_vars = self.critic.trainable_variables

                self._all_train_vars = self._actor_train_vars + self._critic_train_vars
            else:
                self._all_train_vars = self._actor_train_vars

        # 确定resample_prob函数
        if 'log_std' in self._explore_dict:
            self.resample_prob = self._resample_log_prob_no_std
        else:
            self.resample_prob = self._resample_log_prob_log_std

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
                    old_prob: tf.Tensor,
                    action: tf.Tensor,
                    clip_ratio: float,
                    entropy_coef: float,
                    normalize_advantage: bool = True,
                    ):
        # cancel state std
        with tf.GradientTape() as tape:
            tape.watch(self.actor_train_vars)

            old_log_prob = tf.reduce_sum(tf.math.log(old_prob), axis=1, keepdims=True)
            out = self.resample_prob(actor_inputs, action)

            log_prob = tf.reduce_sum(out[0], axis=1, keepdims=True)
            log_std = out[1]
            mu = out[2]

            ratio = tf.exp(log_prob - old_log_prob)

            if normalize_advantage:
                advantage = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-8)

            actor_surrogate_loss = tf.reduce_mean(tf.minimum(
                    ratio * advantage,
                    tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage,
                )
                )


            entropy_loss = tf.reduce_mean(self.explore_policy.get_entropy(mu, log_std))

            # entropy_loss = tf.reduce_sum(tf.reduce_mean(-log_prob, axis=0))

            actor_loss = -actor_surrogate_loss - entropy_coef * entropy_loss

        actor_grads = tape.gradient(actor_loss, self.actor_train_vars)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_train_vars))

        dic = {
            'actor_loss': actor_loss,
            'actor_surrogate_loss': actor_surrogate_loss,
            'entropy_loss': entropy_loss,
        }

        return dic, log_prob

    @tf.function
    def train_shared(self,
                     target: tf.Tensor,
                     actor_inputs: list,
                     advantage: tf.Tensor,
                     old_log_prob: tf.Tensor,
                     action: tf.Tensor,
                     clip_ratio: float,
                     entropy_coef: float,
                     vf_coef: float,
                     normalize_advantage: bool = True,
                     ):

        with tf.GradientTape() as tape:
            tape.watch(self.actor_train_vars)
            old_log_prob = tf.reduce_sum(tf.math.log(old_log_prob), axis=1, keepdims=True)

            out = self.resample_prob(actor_inputs, action)

            log_prob = tf.reduce_sum(out[0], axis=1, keepdims=True)
            log_std = out[1]
            mu = out[2]
            value = out[self.value_idx + 2]

            ratio = tf.exp(log_prob - old_log_prob)

            if normalize_advantage:
                advantage = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-8)

            actor_surrogate_loss = tf.reduce_mean(tf.minimum(
                    ratio * advantage,
                    tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage,
                ))


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

        return dic, log_prob

    @tf.function
    def train_all(self,
                  target: tf.Tensor,
                  actor_inputs: list,
                  critic_inputs: list,
                  advantage: tf.Tensor,
                  old_log_prob: tf.Tensor,
                  action: tf.Tensor,
                  clip_ratio: float,
                  entropy_coef: float,
                  vf_coef: float,
                  normalize_advantage: bool = True,
                  ):

        with tf.GradientTape() as tape:
            tape.watch(self.all_train_vars)
            old_log_prob_ = tf.reduce_sum(tf.math.log(old_log_prob), axis=1, keepdims=True)

            out = self.resample_prob(actor_inputs, action)

            log_prob = tf.reduce_sum(out[0], axis=1, keepdims=True)
            log_std = out[1]
            mu = out[2]

            ratio = tf.exp(log_prob - old_log_prob_)

            if normalize_advantage:
                advantage = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-8)

            actor_surrogate_loss = tf.reduce_mean(tf.minimum(
                    ratio * advantage,
                    tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage,
                ))

            entropy_loss = tf.reduce_mean(self.explore_policy.get_entropy(mu, log_std))

            critic_loss = tf.reduce_mean(tf.square(target - self.critic(*critic_inputs)))

            total_loss = -actor_surrogate_loss - entropy_coef * entropy_loss + vf_coef * critic_loss

        actor_grads = tape.gradient(total_loss, self.all_train_vars)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.all_train_vars))

        dic = {
            'total_loss': total_loss,
            'actor_surrogate_loss': actor_surrogate_loss,
            'entropy_loss': entropy_loss,
            'critic_loss': critic_loss,
        }

        return dic, log_prob

    @property
    def actor_train_vars(self):
        return self._actor_train_vars

    @property
    def all_train_vars(self):
        return self._all_train_vars

    def optimize(self, data_set: RLStandardDataSet):

        # 检查当前是否为主线程
        if self.level != 0:
            raise RuntimeError('Only main agent can optimize')

        train_data, reward_info = self._episode_tool(data_set)

        early_stop = False

        log_std = getattr(self, 'tf_log_std', None)
        if log_std is not None:
            old_log_std = copy.deepcopy(log_std)
        else:
            old_log_std = None

        for i in range(self.agent_params.update_times):
            for batch_data in train_data(self.agent_params.batch_size):
                actor_input_obs = []
                critic_input_obs = []

                for name in self.actor.input_name:
                    actor_input_obs.append(batch_data[name])

                advantage = batch_data['advantage']

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
                        normalize_advantage=self.agent_params.batch_advantage_normalization,
                    )
                    self.loss_tracker.add_data(shared_optimize_info, prefix='shared')
                else:

                    for name in self.critic.input_name:
                        critic_input_obs.append(batch_data[name])

                    if self.agent_params.train_all:
                        all_optimize_info, log_prob = self.train_all(
                            target=batch_data['target'],
                            actor_inputs=actor_input_obs,
                            critic_inputs=critic_input_obs,
                            advantage=advantage,
                            old_log_prob=batch_data['prob'],
                            action=batch_data['action'],
                            clip_ratio=self.agent_params.clip_ratio,
                            entropy_coef=self.agent_params.entropy_coef,
                            vf_coef=self.agent_params.vf_coef,
                            normalize_advantage=self.agent_params.batch_advantage_normalization,
                        )
                        self.loss_tracker.add_data(all_optimize_info, prefix='all')
                    else:

                        for _ in range(self.agent_params.update_critic_times):
                            critic_optimize_info = self.train_critic(
                                critic_inputs=critic_input_obs,
                                target=batch_data['target'],
                            )
                            self.loss_tracker.add_data(critic_optimize_info, prefix='critic')

                        for _ in range(self.agent_params.update_actor_times):
                            actor_optimize_info, log_prob = self.train_actor(
                                actor_inputs=actor_input_obs,
                                advantage=advantage,
                                old_prob=batch_data['prob'],
                                action=batch_data['action'],
                                clip_ratio=self.agent_params.clip_ratio,
                                entropy_coef=self.agent_params.entropy_coef,
                                normalize_advantage=self.agent_params.batch_advantage_normalization,
                            )
                            self.loss_tracker.add_data(actor_optimize_info, prefix='actor')

                    # compute kl divergence, general type
                    old_log_prob = tf.math.log(batch_data['prob'])
                    log_ratio = log_prob - old_log_prob
                    approx_kl_div = tf.reduce_mean(tf.reduce_sum((tf.exp(log_ratio) - 1) - log_ratio, axis=1)).numpy()

                    # compute kl divergence
                    # if old_log_std is not None:
                    #     old_log_prob = tf.math.log(batch_data['prob'])
                    #     log_ratio = log_prob - old_log_prob
                    #     approx_kl_div = tf.reduce_sum(tf.reduce_mean((tf.exp(log_ratio) - 1) - log_ratio, axis=1).numpy()).numpy()
                    #
                    # else:
                    #     new_log_std = getattr(self, 'tf_log_std')
                    #
                    #     old_dist = tfp.distributions.Normal(batch_data['mu'], tf.exp(old_log_std)**2)
                    #     new_dist = tfp.distributions.Normal(batch_data['mu'], tf.exp(new_log_std)**2)
                    #     approx_kl_div = tf.reduce_mean(tfp.distributions.kl_divergence(old_dist, new_dist)).numpy()
                    #
                    self.loss_tracker.add_data({'approx_kl_div': approx_kl_div}, prefix='kl_div')

                    # KL_div = tf.reduce_sum

                    if self.agent_params.target_kl is not None:
                        if approx_kl_div > self.agent_params.target_kl * 1.5:
                            print('Early stopping at step {} due to reaching max kl.'.format(i))
                            early_stop = True
                            break
                    # if approx_kl_div > self.agent_params.target_kl*1.5:
                    #     print('Early stopping at step {} due to reaching max kl.' .format(i))
                    #     early_stop = True
                    #     break

                if early_stop:
                    break

            if early_stop:
                break

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
