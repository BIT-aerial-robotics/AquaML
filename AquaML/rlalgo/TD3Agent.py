import tensorflow as tf
from AquaML.rlalgo.BaseRLAgent import BaseRLAgent, LossTracker
from AquaML.rlalgo.AgentParameters import TD3AgentParameters
from AquaML.core.RLToolKit import RLStandardDataSet
from AquaML.core.DataParser import DataSet
import tensorflow_probability as tfp
import os
import numpy as np


class TD3Agent(BaseRLAgent):

    def __init__(self,
                 name,
                 actor,
                 q_critic,
                 agent_params: TD3AgentParameters,
                 level=0
                 ):

        super().__init__(
            name=name,
            level=level,
            agent_params=agent_params,
        )

        self.actor = actor()
        self.target_actor = actor()

        self.q_critit1 = q_critic()
        self.q_critit2 = q_critic()
        self.target_q_critic1 = q_critic()
        self.target_q_critic2 = q_critic()

        self.n_update_times = 0

        # self.expert_dataset_path = expert_dataset_path

    def init(self):

        self.initialize_actor()

        if self.level == 0:

            # 创建两个q网络
            self.initialize_model(
                model_class=self.q_critit1,
                name='q_critic1',
            )

            self.initialize_model(
                model_class=self.q_critit2,
                name='q_critic2',
            )

            # 创建target网络
            self.initialize_model(
                model_class=self.target_actor,
                name='target_actor',
            )

            self.initialize_model(
                model_class=self.target_q_critic1,
                name='target_q_critic1',
            )

            self.initialize_model(
                model_class=self.target_q_critic2,
                name='target_q_critic2',
            )

            # copy参数
            self.copy_weights(
                self.actor,
                self.target_actor,
            )

            self.copy_weights(
                self.q_critit1,
                self.target_q_critic1,
            )

            self.copy_weights(
                self.q_critit2,
                self.target_q_critic2,
            )

            # 创建优化器

            if hasattr(self.actor, 'optimizer_info'):
                self.create_optimizer(
                    self.actor.optimizer_info,
                    'actor_optimizer'
                )
            else:
                raise ValueError('optimizer_info is not defined in actor')

            if hasattr(self.q_critit1, 'optimizer_info'):
                self.create_optimizer(
                    self.q_critit1.optimizer_info,
                    'critic_optimizer'
                )

            else:
                raise ValueError('optimizer_info is not defined in agent_params')

            self.config_default_episode_tool() # will be used in traj collect mode

            self._all_model_dict = {
                'actor': self.actor,
                'target_actor': self.target_actor,
                'q_critic1': self.q_critit1,
                'q_critic2': self.q_critit2,
                'target_q_critic1': self.target_q_critic1,
                'target_q_critic2': self.target_q_critic2,
            }

            # create replay buffer
            self.replay_buffer = DataSet(
                data_dict=tuple(self.agent_info.data_info.shape_dict.keys()),
                max_size=self.agent_params.replay_buffer_size,
                IOInfo=self.agent_info,
            )

            # standardize gaussian noise
            # self.gaussian_noise = tfp.distributions.Normal(
            #     loc=0.,
            #     scale=self.agent_params.explore_noise,
            # )

        # 创建探索策略
        if self.agent_params.explore_policy == 'Default':
            explore_name = 'ClipGaussian'
            args = {
                'sigma': self.agent_params.explore_noise,
                'action_high': self.agent_params.action_high,
                'action_low': self.agent_params.action_low,
            }

        else:
            explore_name = self.agent_params.explore_policy
            args = {}

        self.create_explorer(
            explore_name=explore_name,
            shape=self.actor.output_info['action'],
            args=args,
        )

        self.standardize_noise = tfp.distributions.Normal(
            loc=tf.zeros(self.actor.output_info['action']),
            scale=tf.ones(self.actor.output_info['action'])
        )

        # 初始化模型同步器
        self._sync_model_dict = {
            'actor': self.actor,
        }

    def optimize(self, data_set: RLStandardDataSet, run_mode='off-policy') -> (LossTracker, dict):

        if run_mode == 'off-policy':
            train_data = data_set.get_all_data(squeeze=True, rollout_steps=self.agent_params.rollout_steps, env_num=self.env_num)
            reward_info = {}

        elif run_mode == 'on-policy':
            train_data_, reward_info = self._episode_tool(data_set)
            train_data = train_data_.data_dict
        else:
            raise ValueError('run_mode must be off-policy or on-policy')

        # train_data, reward_info = self._episode_tool(data_set)

        self.replay_buffer.add_data_by_buffer(train_data)

        current_buffer_size = self.replay_buffer.buffer_size

        self.eval_flag = current_buffer_size > self.agent_params.learning_starts

        if current_buffer_size > self.agent_params.learning_starts:
            for _ in range(self.agent_params.n_updates):

                sample_data = self.replay_buffer.random_sample_all(self.agent_params.batch_size,
                                                                   tf_dataset=False)  # dict

                # compute target y

                # get next actor input

                next_actor_input = self.get_corresponding_data(sample_data, self.actor.input_name, 'next_')
                next_q_input = self.get_corresponding_data(sample_data, self.q_critit1.input_name, 'next_',
                                                           filter='next_action')
                current_q_input = self.get_corresponding_data(sample_data, self.q_critit1.input_name, filter='action')
                current_actor_input = self.get_corresponding_data(sample_data, self.actor.input_name)

                current_action = tf.convert_to_tensor(sample_data['action'], dtype=tf.float32)

                reward = tf.convert_to_tensor(sample_data['total_reward'], dtype=tf.float32)
                mask = tf.convert_to_tensor(sample_data['mask'], dtype=tf.float32)

                target_y = self.compute_target_y(
                    next_actor_input=next_actor_input,
                    next_q_input=next_q_input,
                    reward=reward,
                    mask=mask,
                    policy_noise=self.agent_params.policy_noise,
                    noise_clip_range=self.agent_params.noise_clip_range,
                    action_low=self.agent_params.action_low,
                    action_high=self.agent_params.action_high,
                    gamma=self.agent_params.gamma,
                )

                # current_action = self.actor(*current_actor_input)[0]

                # train critic
                q1_info = self.train_critic1(
                    current_q_input=current_q_input,
                    target_y=target_y,
                    current_action=current_action,
                )

                self.loss_tracker.add_data(q1_info)

                q2_info = self.train_critic2(
                    current_q_input=current_q_input,
                    target_y=target_y,
                    current_action=current_action,
                )

                self.loss_tracker.add_data(q2_info)

                self.n_update_times += 1

                if self.n_update_times % self.agent_params.delay_update == 0:
                    loss = self.train_actor(
                        current_actor_input=current_actor_input,
                    )

                    self.loss_tracker.add_data(loss)

                    # update target model

                    self.soft_update(
                        source_model=self.actor,
                        target_model=self.target_actor,
                        tau=self.agent_params.tau,
                    )

                    self.soft_update(
                        source_model=self.q_critit1,
                        target_model=self.target_q_critic1,
                        tau=self.agent_params.tau,
                    )

                    self.soft_update(
                        source_model=self.q_critit2,
                        target_model=self.target_q_critic2,
                        tau=self.agent_params.tau,
                    )

        # loss_info = self.loss_tracker.get_data()

        return self.loss_tracker, reward_info

    @tf.function
    def compute_target_y(self,
                         next_actor_input,
                         next_q_input,  # state no action
                         reward,
                         mask,
                         action_low,
                         action_high,
                         policy_noise=0.2,
                         noise_clip_range=0.5,
                         gamma=0.99
                         ):
        # compute next action
        next_target_action = self.target_actor(*next_actor_input)[0]

        size = next_target_action.shape[0]

        # add noise
        org_noise = self.standardize_noise.sample(size) * policy_noise
        clip_noise = tf.clip_by_value(org_noise, -noise_clip_range, noise_clip_range)

        noise_next_target_action = next_target_action + clip_noise

        clip_noise_next_target_action = tf.clip_by_value(noise_next_target_action, action_low, action_high)

        q1 = self.target_q_critic1(*next_q_input, clip_noise_next_target_action)
        q2 = self.target_q_critic2(*next_q_input, clip_noise_next_target_action)

        minmun_q = tf.minimum(q1, q2)

        target_y = reward + gamma * mask * minmun_q

        return target_y

    @tf.function
    def train_critic1(self,
                      current_q_input,  # state no action
                      current_action,
                      target_y,
                      ):

        with tf.GradientTape() as tape:
            tape.watch(self.q_critit1.trainable_variables)

            q1 = self.q_critit1(*current_q_input, current_action)

            loss = tf.reduce_mean(tf.square(q1 - target_y))

        grads = tape.gradient(loss, self.q_critit1.trainable_variables)

        self.critic_optimizer.apply_gradients(zip(grads, self.q_critit1.trainable_variables))

        dict_info = {
            'critic1_loss': loss,
        }

        return dict_info

    @tf.function
    def train_critic2(self,
                      current_q_input,  # state no action
                      current_action,
                      target_y,
                      ):

        with tf.GradientTape() as tape:
            tape.watch(self.q_critit2.trainable_variables)

            q2 = self.q_critit2(*current_q_input, current_action)

            loss = tf.reduce_mean(tf.square(q2 - target_y))

        grads = tape.gradient(loss, self.q_critit2.trainable_variables)

        self.critic_optimizer.apply_gradients(zip(grads, self.q_critit2.trainable_variables))

        dict_info = {
            'critic2_loss': loss,
        }

        return dict_info

    @tf.function
    def train_actor(self,
                    current_actor_input,
                    ):

        with tf.GradientTape() as tape:
            tape.watch(self.actor.trainable_variables)

            current_action = self.actor(*current_actor_input)[0]

            q1 = self.q_critit1(*current_actor_input, current_action)

            loss = -tf.reduce_mean(q1)

        grads = tape.gradient(loss, self.actor.trainable_variables)

        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        dict_info = {
            'actor_loss': loss,
        }

        return dict_info

    @staticmethod
    def get_algo_name():
        return 'TD3'

