import numpy as np
import tensorflow as tf

from AquaML.rlalgo.BaseRLAgent import BaseRLAgent
from AquaML.rlalgo.AgentParameters import AMPAgentParameter
from AquaML.core.RLToolKit import RLStandardDataSet
from AquaML.buffer.RLPrePlugin import ValueFunctionComputer, GAEComputer, SplitTrajectory
import copy
import tensorflow_probability as tfp

from AquaML.core.DataParser import DataSet
import os


class AMPAgent(BaseRLAgent):

    def __init__(self,
                 name: str,
                 actor,
                 discriminator,
                 expert_dataset_path,
                 agent_params: AMPAgentParameter,
                 level: int = 0,  # 控制是否创建不交互的agent
                 critic=None,
                 ):

        super().__init__(
            name=name,
            agent_params=agent_params,
            level=level,
        )

        self.dataset_mode = None
        self._episode_tool = None
        self.actor = actor()

        self.discriminator = discriminator()

        self.expert_dataset_path = expert_dataset_path

        if critic is None:
            self.critic = self.actor
            self.model_type = 'share'
        else:
            self.critic = critic()
            self.model_type = 'independent'

        self._network_process_info = {
            'actor': {},
            'critic': {},
            'discriminator': {},
        }  # 网络输入数据处理信息

    def initialize_discriminator(self):
        """
        初始化discriminator。
        """
        # TODO: change to universal initialization
        self.discriminator_expand_dims_idx = []

        discriminator_rnn_flag = getattr(self.discriminator, 'rnn_flag', False)

        if discriminator_rnn_flag:
            self._network_process_info['discriminator']['rnn_flag'] = True

            idx = 0
            discriminator_input_names = self.discriminator.input_name

            for name in discriminator_input_names:
                if 'hidden' in name:
                    pass
                else:
                    self.discriminator_expand_dims_idx.append(idx)
                idx += 1
            self.discriminator_expand_dims_idx = tuple(self.discriminator_expand_dims_idx)

        else:
            self._network_process_info['discriminator']['rnn_flag'] = False

        self.initialize_network(
            model=self.discriminator,
            expand_dims_idx=self.discriminator_expand_dims_idx,
        )

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

            # 初始化discriminator网络
            self.initialize_discriminator()

            self._all_model_dict['discriminator'] = self.discriminator

            # 创建discriminator优化器
            if hasattr(self.discriminator, 'optimizer_info'):
                self.create_optimizer(self.discriminator.optimizer_info, 'discriminator_optimizer')
            else:
                raise AttributeError(f'{self.discriminator.__class__.__name__} has no optimizer_info attribute')

            # 加载专家数据集,创建专家Dataset
            expert_data_dict = {}

            for key in self.discriminator.input_name:
                data_file_path = os.path.join(self.expert_dataset_path, key + '.npy')
                expert_data_dict[key] = np.load(data_file_path)

            self.expert_dataset = DataSet(
                data_dict=expert_data_dict,
            )

            # 创建discriminator的 replay buffer
            self.discriminator_buffer = DataSet(
                data_dict=self.discriminator.input_name,
                max_size=self.agent_params.discriminator_replay_buffer_size,
                IOInfo=self.agent_info
            )

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
                summary_stype=self.agent_params.summary_style,
                summary_steps=self.agent_params.summary_steps,
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

            self.std_norm = tfp.distributions.Normal(0, 0.8)

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

        # config data set
        if self.agent_params.is_sequential:
            self.dataset_mode = 'seq'

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
    def train_discriminator(self,
                            expert_state: tuple,
                            learned_state: tuple,
                            gp_coef: float = 10.0,
                            ):
        """
        expert data must have the same shape as learned data

        Args:
            expert_state: 
            learned_state:
        """
        with tf.GradientTape() as tape:
            tape.watch(self.discriminator.trainable_variables)
            expert_score = self.discriminator(*expert_state)
            learned_score = self.discriminator(*learned_state)

            expert_loss = tf.reduce_mean(tf.square(expert_score - 1))
            learned_loss = tf.reduce_mean(tf.square(learned_score + 1))

            penalty = tf.reduce_mean(tf.square(expert_score))

            loss = expert_loss + learned_loss + gp_coef / 2.0 * penalty

        grads = tape.gradient(loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        dic = {
            'expert_loss': expert_loss,
            'learned_loss': learned_loss,
            'penalty': penalty,
            'discriminator': loss,
        }

        return dic

    @tf.function
    def train_actor(self,
                    actor_inputs: tuple,
                    advantage: tf.Tensor,
                    old_prob: tf.Tensor,
                    action: tf.Tensor,
                    bool_mask: tf.Tensor,
                    clip_ratio: float,
                    entropy_coef: float,
                    normalize_advantage: bool = True,
                    ):
        # cancel state std
        old_log_prob = tf.reduce_sum(tf.math.log(old_prob), axis=self.sum_axis, keepdims=True)
        with tf.GradientTape() as tape:
            tape.watch(self.actor_train_vars)

            out = self.resample_prob(actor_inputs, action, mask=bool_mask)

            log_prob = out[0]
            log_std = out[1]
            mu = out[2]

            # ratio = tf.reduce_sum(tf.exp(log_prob - old_log_prob), axis=1, keepdims=True)

            # 动作是独立的
            # ratio = tf.exp(tf.reduce_sum(log_prob - old_log_prob, axis=1, keepdims=True))
            ratio = tf.exp(log_prob - old_log_prob)

            mask_ratio = tf.boolean_mask(ratio, bool_mask)
            mask_advantage = tf.boolean_mask(advantage, bool_mask)

            if normalize_advantage:
                mask_advantage = (mask_advantage - tf.reduce_mean(mask_advantage)) / (
                        tf.math.reduce_std(mask_advantage) + 1e-8)

            surr1 = mask_ratio * mask_advantage
            surr2 = tf.clip_by_value(mask_ratio, 1 - clip_ratio, 1 + clip_ratio) * mask_advantage
            surr = tf.minimum(surr1, surr2)

            actor_surrogate_loss = tf.reduce_mean(surr)

            entropy_loss = self.explore_policy.get_entropy(mu, log_std)

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
                     bool_mask: tf.Tensor,
                     clip_ratio: float,
                     entropy_coef: float,
                     vf_coef: float,
                     normalize_advantage: bool = True,
                     ):
        old_log_prob = tf.reduce_sum(tf.math.log(old_log_prob), axis=1, keepdims=True)
        with tf.GradientTape() as tape:
            tape.watch(self.actor_train_vars)

            out = self.resample_prob(actor_inputs, action, mask=bool_mask)

            log_prob = out[0]
            log_std = out[1]
            mu = out[2]
            value = out[self.value_idx + 2]

            # ratio = tf.reduce_sum(tf.exp(log_prob - old_log_prob), axis=1, keepdims=True)

            # 动作是独立的
            ratio = tf.exp(log_prob - old_log_prob)

            mask_ratio = tf.boolean_mask(ratio, bool_mask)
            mask_advantage = tf.boolean_mask(advantage, bool_mask)

            if normalize_advantage:
                mask_advantage = (mask_advantage - tf.reduce_mean(mask_advantage)) / (
                        tf.math.reduce_std(mask_advantage) + 1e-8)

            surr1 = mask_ratio * mask_advantage
            surr2 = tf.clip_by_value(mask_ratio, 1 - clip_ratio, 1 + clip_ratio) * mask_advantage

            surr = tf.minimum(surr1, surr2)

            actor_surrogate_loss = tf.reduce_mean(surr)

            entropy_loss = self.explore_policy.get_entropy(mu, log_std)

            value_l = tf.square(target - value)

            mask_value_l = tf.boolean_mask(value_l, bool_mask)

            value_loss = tf.reduce_mean(mask_value_l)

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
                  bool_mask: tf.Tensor,
                  clip_ratio: float,
                  entropy_coef: float,
                  vf_coef: float,
                  normalize_advantage: bool = True,

                  ):

        old_log_prob = tf.reduce_sum(tf.math.log(old_log_prob), axis=self.sum_axis, keepdims=True)

        with tf.GradientTape() as tape:
            tape.watch(self.all_train_vars)

            out = self.resample_prob(actor_inputs, action, mask=bool_mask)

            log_prob = out[0]
            log_std = out[1]
            mu = out[2]

            # ratio = tf.reduce_sum(tf.exp(log_prob - old_log_prob), axis=1, keepdims=

            # 动作是独立的
            ratio = tf.exp(log_prob - old_log_prob)

            mask_ratio = tf.boolean_mask(ratio, bool_mask)
            mask_advantage = tf.boolean_mask(advantage, bool_mask)

            if normalize_advantage:
                mask_advantage = (mask_advantage - tf.reduce_mean(mask_advantage)) / (
                        tf.math.reduce_std(mask_advantage) + 1e-8)

            surr1 = mask_ratio * mask_advantage
            surr2 = tf.clip_by_value(mask_ratio, 1 - clip_ratio, 1 + clip_ratio) * mask_advantage

            surr = tf.minimum(surr1, surr2)

            # mask_surr = tf.boolean_mask(surr, bool_mask)

            actor_surrogate_loss = tf.reduce_mean(surr)

            entropy_loss = self.explore_policy.get_entropy(mu, log_std)

            critic_l = tf.square(target - self.critic(*critic_inputs))
            mask_critic_l = tf.boolean_mask(critic_l, bool_mask)
            critic_loss = tf.reduce_mean(mask_critic_l)

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

    @tf.function
    def compute_lam_key(self, critic_inputs, actor_inputs, target, bool_mask):

        # compute fusion loss weight
        # do not compute gradient
        c_value = self.critic(*critic_inputs)  # updated critic value, provent stop gradient
        mask_c_value = tf.boolean_mask(c_value, bool_mask)
        fusion_value = self.actor(*actor_inputs, mask=bool_mask)[1]  # fusion value
        mask_fusion_value = tf.boolean_mask(fusion_value, bool_mask)
        mask_target = tf.boolean_mask(target, bool_mask)

        c_target = tf.reduce_mean(tf.square(mask_target - mask_c_value))
        fusion_value_c = tf.reduce_mean(tf.square(mask_fusion_value - mask_c_value))

        fusion_value_error = tf.reduce_mean(tf.square(mask_fusion_value - mask_target))

        # flag = not fusion_value_error < c_target * 2.0

        distance = tf.sqrt(c_target) + tf.sqrt(fusion_value_c)

        # flag = tf.cast(flag, tf.float32)

        # min_value = tf.minimum(flag, 0.1)

        # lam = tf.clip_by_value(1.0 / distance, 0, min_value)

        return fusion_value_error, c_target, distance

    @tf.function
    def train_fusion(self,
                     target: tf.Tensor,
                     actor_inputs: list,
                     critic_inputs: list,
                     advantage: tf.Tensor,
                     old_log_prob: tf.Tensor,
                     action: tf.Tensor,
                     bool_mask: tf.Tensor,
                     clip_ratio: float,
                     entropy_coef: float,
                     vf_coef: float,
                     lam: float,
                     normalize_advantage: bool = True,
                     ):

        # lam = 0.1
        old_log_prob = tf.reduce_sum(tf.math.log(old_log_prob), axis=self.sum_axis, keepdims=True)
        with tf.GradientTape() as tape:
            tape.watch(self.all_train_vars)

            # update critic first
            critic_l = tf.square(target - self.critic(*critic_inputs))
            mask_critic_l = tf.boolean_mask(critic_l, bool_mask)
            critic_loss = tf.reduce_mean(mask_critic_l)

            # update actor
            out = self.resample_prob(actor_inputs, action, mask=bool_mask)
            log_prob = out[0]
            log_std = out[1]
            mu = out[2]

            f_v = out[3]

            ratio = tf.exp(log_prob - old_log_prob)
            mask_ratio = tf.boolean_mask(ratio, bool_mask)
            mask_advantage = tf.boolean_mask(advantage, bool_mask)
            if normalize_advantage:
                mask_advantage = (mask_advantage - tf.reduce_mean(mask_advantage)) / (
                        tf.math.reduce_std(mask_advantage) + 1e-8)
            surr1 = mask_ratio * mask_advantage
            surr2 = tf.clip_by_value(mask_ratio, 1 - clip_ratio, 1 + clip_ratio) * mask_advantage

            surr = tf.minimum(surr1, surr2)
            # mask_surr = tf.boolean_mask(surr, bool_mask)
            actor_surrogate_loss = tf.reduce_mean(surr)

            f_l = tf.square(f_v - target)
            mask_f_l = tf.boolean_mask(f_l, bool_mask)
            fusion_loss = tf.reduce_mean(mask_f_l)

            entropy_loss = self.explore_policy.get_entropy(mu, log_std)

            total_loss = -actor_surrogate_loss - entropy_coef * entropy_loss + vf_coef * critic_loss + lam * fusion_loss

        actor_grads = tape.gradient(total_loss, self.all_train_vars)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.all_train_vars))

        dic = {
            'total_loss': total_loss,
            'actor_surrogate_loss': actor_surrogate_loss,
            'entropy_loss': entropy_loss,
            'critic_loss': critic_loss,
            'fusion_loss': fusion_loss,
            'lam': lam,
            'ratio': tf.math.abs(fusion_loss / actor_surrogate_loss),
        }

        return dic, log_prob

    @property
    def actor_train_vars(self):
        return self._actor_train_vars

    @property
    def all_train_vars(self):
        return self._all_train_vars

    def optimize(self, data_set: RLStandardDataSet):

        # # 检查当前是否为主线程
        # if self.level != 0:
        #     raise RuntimeError('Only main agent can optimize')

        # compute style reward
        discriminator_input = data_set.get_data_by_list(self.discriminator.input_name, True)

        D_ss = self.discriminator(*discriminator_input)

        style_reward_1 = 1 - 0.25 * tf.square(D_ss - 1.0)

        zero = tf.zeros_like(style_reward_1)

        style_reward = tf.maximum(style_reward_1, zero).numpy()

        # change reward
        org_total_reward = copy.deepcopy(data_set.get_data('total_reward'))
        new_total_reward = org_total_reward * self.agent_params.task_rew_coef + style_reward * self.agent_params.style_rew_coef
        data_set.change_data(new_total_reward, 'total_reward')

        data_set.add_addtional_reward(style_reward, 'style_reward')

        # process trajectory
        train_data, reward_info = self._episode_tool(data_set, shuffle=self.agent_params.shuffle)

        # update discriminator repaly buffer
        discriminator_data = train_data.get_required_data(self.discriminator.input_name)
        discriminator_data_dict = dict(zip(self.discriminator.input_name, discriminator_data))
        self.discriminator_buffer.add_data_by_buffer(discriminator_data_dict)

        early_stop = False

        # update discriminator
        for _ in range(self.agent_params.update_discriminator_times):
            expert_data = self.expert_dataset.random_sample(self.agent_params.k_batch_size,
                                                            self.discriminator.input_name)
            learner_data = self.discriminator_buffer.random_sample(self.agent_params.k_batch_size,
                                                                   self.discriminator.input_name)

            loss_info = self.train_discriminator(expert_data, learner_data,
                                                 self.agent_params.gp_coef,
                                                 )

            self.loss_tracker.add_data(loss_info)

        # log_std = getattr(self, 'tf_log_std', None)
        # if log_std is not None:
        #     old_log_std = copy.deepcopy(log_std)
        # else:
        #     old_log_std = None

        for i in range(self.agent_params.update_times):
            for batch_data in train_data(self.agent_params.batch_size, mode=self.dataset_mode,
                                         args=self.agent_params.sequential_args):
                actor_input_obs = []
                critic_input_obs = []

                for name in self.actor.input_name:
                    actor_input_obs.append(batch_data[name])

                advantage = batch_data['advantage']
                bool_mask = batch_data['bool_mask']

                if self.model_type == 'share':
                    shared_optimize_info, log_prob = self.train_shared(
                        actor_inputs=actor_input_obs,
                        target=batch_data['target'],
                        advantage=advantage,
                        old_log_prob=batch_data['prob'],
                        action=batch_data['action'],
                        bool_mask=bool_mask,
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
                        if self.agent_params.train_fusion:

                            fusion_value_error, c_target, distance = self.compute_lam_key(
                                actor_inputs=actor_input_obs,
                                critic_inputs=critic_input_obs,
                                target=batch_data['target'],
                                bool_mask=bool_mask,
                            )

                            flag = not fusion_value_error < c_target * 2.0

                            flag = tf.cast(flag, tf.float32)

                            min_value = tf.minimum(flag, 0.1)

                            lam = tf.clip_by_value(tf.stop_gradient(1.0 / distance), 0, min_value)

                            all_optimize_info, log_prob = self.train_fusion(
                                target=batch_data['target'],
                                actor_inputs=actor_input_obs,
                                critic_inputs=critic_input_obs,
                                advantage=advantage,
                                bool_mask=bool_mask,
                                old_log_prob=batch_data['prob'],
                                action=batch_data['action'],
                                clip_ratio=self.agent_params.clip_ratio,
                                entropy_coef=self.agent_params.entropy_coef,
                                vf_coef=self.agent_params.vf_coef,
                                normalize_advantage=self.agent_params.batch_advantage_normalization,
                                lam=lam,
                            )
                        else:
                            all_optimize_info, log_prob = self.train_all(
                                target=batch_data['target'],
                                actor_inputs=actor_input_obs,
                                critic_inputs=critic_input_obs,
                                advantage=advantage,
                                bool_mask=bool_mask,
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
                                bool_mask=bool_mask,
                                clip_ratio=self.agent_params.clip_ratio,
                                entropy_coef=self.agent_params.entropy_coef,
                                normalize_advantage=self.agent_params.batch_advantage_normalization,
                            )
                            self.loss_tracker.add_data(actor_optimize_info, prefix='actor')

                    # compute kl divergence, general type
                    old_log_prob = tf.reduce_sum(tf.math.log(batch_data['prob']), axis=self.sum_axis, keepdims=True)
                    log_ratio = log_prob - old_log_prob  # 多维分布
                    log_ratio = tf.boolean_mask(log_ratio, bool_mask)
                    approx_kl_div = tf.reduce_mean(tf.exp(log_ratio) - 1 - log_ratio).numpy()
                    # approx_kl_div = tf.reduce_mean(((tf.exp(log_ratio) - 1) - log_ratio, axis=1)).numpy()

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

    def _resample_log_prob_no_std(self, obs, action, mask=None):

        """
        Re get log_prob of action.
        The output of actor model is (mu,).
        It is different from resample_action.

        Args:
            obs (tuple): observation.
            action (tf.Tensor): action.
        """

        out = self.actor(*obs, mask)
        mu = out[0]
        std = tf.exp(self.tf_log_std)
        log_prob = self.explore_policy.resample_prob(mu, std, action, sum_axis=self.sum_axis)

        return (log_prob, self.tf_log_std, *out)

    def _resample_log_prob_log_std(self, obs: tuple, action, bool_mask=None):
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

        out = self.actor(*obs, bool_mask=bool_mask)

        mu, log_std = out[0], out[1]

        std = tf.exp(log_std)

        log_prob = self.explore_policy.resample_prob(mu, std, action, sum_axis=self.sum_axis)

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
