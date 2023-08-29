import tensorflow as tf
from AquaML.core.ToolKit import LossTracker
from AquaML.rlalgo.BaseRLAgent import BaseRLAgent, LossTracker
from AquaML.rlalgo.AgentParameters import SACAgentParameters
from AquaML.core.RLToolKit import RLStandardDataSet
from AquaML.core.DataParser import DataSet
import numpy as np


class SACAgent(BaseRLAgent):

    def __init__(self,
                 name,
                 actor,
                 q_critic,
                 agent_params: SACAgentParameters,
                 level=0,
                 ):
        super().__init__(
            name=name,
            level=level,
            agent_params=agent_params,
        )

        self.actor = actor()

        self.q_critic1 = q_critic()
        self.q_critic2 = q_critic()
        self.target_q_critic1 = q_critic()
        self.target_q_critic2 = q_critic()

        self.n_update_times = 0

    def init(self):
        self.initialize_actor()

        if self.level == 0:
            # initialize the critic
            self.initialize_model(
                model_class=self.q_critic1,
                name='q_critic1',
            )

            self.initialize_model(
                model_class=self.q_critic2,
                name='q_critic2',
            )

            self.initialize_model(
                model_class=self.target_q_critic1,
                name='target_q_critic1',
            )

            self.initialize_model(
                model_class=self.target_q_critic2,
                name='target_q_critic2',
            )

            # copy parameters to target model
            self.copy_weights(
                self.q_critic1,
                self.target_q_critic1,
            )

            self.copy_weights(
                self.q_critic2,
                self.target_q_critic2,
            )

            # 创建soft alpha
            actor_out_dim = self.actor.output_info['action'][0]
            self.log_alpha = tf.Variable(-np.log(actor_out_dim*1.0), dtype=tf.float32, name='log_alpha',
                                         trainable=True)

            # config target entropy
            if self.agent_params.target_entropy == 'default':
                self.target_entropy = -np.prod(actor_out_dim).astype(np.float32)
            else:
                self.target_entropy = self.agent_params.target_entropy

            # 创建优化器

            # actor optimizer
            if hasattr(self.actor, 'optimizer_info'):
                self.create_optimizer(
                    optimizer_info=self.actor.optimizer_info,
                    name='actor_optimizer',
                )
            else:
                raise ValueError('actor optimizer info is not defined')

            # critic optimizer
            if hasattr(self.q_critic1, 'optimizer_info'):
                self.create_optimizer(
                    optimizer_info=self.q_critic1.optimizer_info,
                    name='critic_optimizer',
                )
            else:
                raise ValueError('critic optimizer info is not defined')

            # alpha optimizer
            self.create_optimizer(
                optimizer_info=self.agent_params.alpha_optimizer_info,
                name='alpha_optimizer',
            )

            # using default episode tool
            self.config_default_episode_tool()

            # config all model
            self._all_model_dict = {
                'actor': self.actor,
                'q_critic1': self.q_critic1,
                'q_critic2': self.q_critic2,
                'target_q_critic1': self.target_q_critic1,
                'target_q_critic2': self.target_q_critic2,
                # 'log_alpha': self.log_alpha, # TODO: 优化加载函数
            }

            # create replay buffer
            self.replay_buffer = DataSet(
                data_dict=tuple(self.agent_info.data_info.shape_dict.keys()),
                max_size=self.agent_params.replay_buffer_size,
                IOInfo=self.agent_info,
            )

            # 创建探索策略
            if self.agent_params.explore_policy == 'Default':
                explore_name = 'Gaussian'
                log_std_init = {
                    'log_std': 0.0,
                }
                args = {
                    'tanh': True,
                }
            else:
                explore_name = self.agent_params.explore_policy
                log_std_init = {}
                args = {}

            self.create_explorer(
                explore_name=explore_name,
                shape=self.actor.output_info['action'],
                pointed_value=log_std_init,
                args=args,
                filter=['log_std', ],  # SAC中log_std不需要被训练
            )
    @tf.function
    def compute_target_q(self,
                         next_actor_input,
                         next_q_input,
                         reward,
                         mask,
                         gamma=0.99,
                         ):
        actor_out = self.actor(*next_actor_input)

        mu = actor_out[0]
        log_std = actor_out[1]

        size = mu.shape[0]

        noise = self.explore_policy.dist.sample(size)
        log_prob = self.explore_policy.dist.log_prob(noise)

        next_action = mu + noise * tf.exp(log_std)

        q1 = self.target_q_critic1(*next_q_input, next_action)
        q2 = self.target_q_critic2(*next_q_input, next_action)

        minmun_q = tf.minimum(q1, q2)

        next_q = minmun_q - tf.exp(self.log_alpha) * log_prob

        target_q = reward + mask * gamma * next_q

        return target_q, log_prob, noise
    @tf.function
    def train_critic1(self,
                      current_q_input,
                      replay_action,
                      target_q
                      ):

        with tf.GradientTape() as tape:
            tape.watch(self.q_critic1.trainable_variables)
            q1 = self.q_critic1(*current_q_input, replay_action)
            loss = tf.reduce_mean(tf.square(q1 - target_q))

        grads = tape.gradient(loss, self.q_critic1.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.q_critic1.trainable_variables))

        dict_info = {
            'critic1_loss': loss,
        }

        return dict_info

    @tf.function
    def train_critic2(self,
                      current_q_input,
                      replay_action,
                      target_q
                      ):

        with tf.GradientTape() as tape:
            tape.watch(self.q_critic2.trainable_variables)
            q2 = self.q_critic2(*current_q_input, replay_action)
            loss = tf.reduce_mean(tf.square(q2 - target_q))

        grads = tape.gradient(loss, self.q_critic2.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.q_critic2.trainable_variables))

        dict_info = {
            'critic2_loss': loss,
        }

        return dict_info
    @tf.function
    def train_alpha(self,
                    log_prob,
                    target_entropy,
                    ):

        with tf.GradientTape() as tape:
            tape.watch(self.log_alpha)
            loss = -tf.reduce_mean(self.log_alpha * (log_prob + target_entropy))

        grads = tape.gradient(loss, self.log_alpha)
        self.alpha_optimizer.apply_gradients(zip([grads,], [self.log_alpha,]))

        return loss
    @tf.function
    def train_actor(self,
                    current_actor_input,
                    current_q_input,
                    target_entropy,
                    # noise,
                    ):
        with tf.GradientTape() as tape:
            tape.watch(self.actor.trainable_variables)

            actor_out = self.actor(*current_actor_input)
            mu = actor_out[0]
            log_std = actor_out[1]
            size = mu.shape[0]

            noise = self.explore_policy.dist.sample(size)
            log_prob = self.explore_policy.dist.log_prob(noise)
            action = mu + noise * tf.exp(log_std)
            action = self.explore_policy.activate_fn(action)

            q1 = self.q_critic1(*current_q_input, action)
            q2 = self.q_critic2(*current_q_input, action)

            min_q = tf.minimum(q1, q2)

            actor_loss = tf.reduce_mean(tf.exp(self.log_alpha) * log_prob - min_q)

        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        alpha_loss = self.train_alpha(
            log_prob=log_prob,
            target_entropy=target_entropy,
        )

        return_dict = {
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
        }

        return return_dict

    def optimize(self, data_set: RLStandardDataSet, run_mode='off-policy') -> (LossTracker, dict):

        if run_mode == 'off-policy':
            train_data = data_set.get_all_data(squeeze=True, rollout_steps=self.agent_params.rollout_steps,
                                               env_num=self.env_num)
            reward_info = {}

        elif run_mode == 'on-policy':
            train_data_, reward_info = self._episode_tool(data_set)
            train_data = train_data_.data_dict
        else:
            raise ValueError('run_mode must be off-policy or on-policy')

        self.replay_buffer.add_data_by_buffer(train_data)

        current_buffer_size = self.replay_buffer.buffer_size

        self.eval_flag = current_buffer_size > self.agent_params.learning_starts  # 启用优化

        if self.eval_flag:
            for _ in range(self.agent_params.update_times):
                sample_data = self.replay_buffer.random_sample_all(self.agent_params.batch_size,
                                                                   tf_dataset=False)  # dict

                next_actor_input = self.get_corresponding_data(sample_data, self.actor.input_name, 'next_')
                next_q_input = self.get_corresponding_data(sample_data, self.q_critic1.input_name, 'next_',
                                                           filter='next_action')
                current_q_input = self.get_corresponding_data(sample_data, self.q_critic2.input_name, filter='action')
                current_actor_input = self.get_corresponding_data(sample_data, self.actor.input_name)

                current_action = tf.convert_to_tensor(sample_data['action'], dtype=tf.float32)

                reward = tf.convert_to_tensor(sample_data['total_reward'], dtype=tf.float32)
                mask = tf.convert_to_tensor(sample_data['mask'], dtype=tf.float32)

                target_q = self.compute_target_q(
                    next_actor_input=next_actor_input,
                    next_q_input=next_q_input,
                    reward=reward,
                    mask=mask,
                    gamma=self.agent_params.gamma
                )

                q1_info = self.train_critic1(
                    current_q_input=current_q_input,
                    replay_action=current_action,
                    target_q=target_q
                )

                self.loss_tracker.add_data(q1_info)

                q2_info = self.train_critic2(
                    current_q_input=current_q_input,
                    replay_action=current_action,
                    target_q=target_q
                )

                self.loss_tracker.add_data(q2_info)

                actor_info = self.train_actor(
                    current_actor_input=current_actor_input,
                    current_q_input=current_q_input,
                    target_entropy=self.target_entropy
                )
                self.loss_tracker.add_data(actor_info)

                self.n_update_times += 1

                if self.n_update_times % self.agent_params.delay_update == 0:
                    self.soft_update(
                        source_model=self.q_critic1,
                        target_model=self.target_q_critic1,
                        tau=self.agent_params.tau,
                    )

                    self.soft_update(
                        source_model=self.q_critic2,
                        target_model=self.target_q_critic2,
                        tau=self.agent_params.tau,
                    )
        return self.loss_tracker, reward_info

    @staticmethod
    def get_algo_name():
        return 'SAC'
