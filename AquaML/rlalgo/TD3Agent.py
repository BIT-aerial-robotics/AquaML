import tensorflow as tf
from AquaML.rlalgo.BaseRLAgent import BaseRLAgent
from AquaML.rlalgo.AgentParameters import TD3BCAgentParameters
from AquaML.core.RLToolKit import RLStandardDataSet
from AquaML.core.DataParser import DataSet

import os
import numpy as np


class TD3BCAgent(BaseRLAgent):

    def __init__(self,
                 name,
                 actor,
                 q_critic,
                 expert_dataset_path,
                 agent_params: TD3BCAgentParameters,
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

        self.expert_dataset_path = expert_dataset_path

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

            if hasattr(self.agent_params, 'optimizer_info'):
                self.create_optimizer(
                    self.actor.optimizer_info,
                    'actor_optimizer'
                )
            else:
                raise ValueError('optimizer_info is not defined in agent_params')

            if hasattr(self.agent_params, 'optimizer_info'):
                self.create_optimizer(
                    self.q_critit1.optimizer_info,
                    'critic_optimizer'
                )

            else:
                raise ValueError('optimizer_info is not defined in agent_params')

            self._all_model_dict = {
                'actor': self.actor,
                'target_actor': self.target_actor,
                'q_critic1': self.q_critit1,
                'q_critic2': self.q_critit2,
                'target_q_critic1': self.target_q_critic1,
                'target_q_critic2': self.target_q_critic2,
            }

            # create replay buffer
            # self.replay_buffer = DataSet(
            #     data_dict=self.agent_info.data_info.shape_dict.keys(),
            #     max_size=self.agent_params.replay_buffer_size,
            #     IOInfo=self.agent_info,
            # )

            # 创建expert dataset
            expert_dataset = {}

            for key in self.actor.input_name:
                data_file_path = os.path.join(self.expert_dataset_path, key + '.npy')
                expert_dataset[key] = np.load(data_file_path)

            # load expert action
            data_file_path = os.path.join(self.expert_dataset_path, 'action.npy')
            expert_dataset['action'] = np.load(data_file_path)

            self.expert_dataset = DataSet(
                data_dict=expert_dataset
            )

        # 创建探索策略
        if self.agent_params.explore_policy == 'Default':
            explore_name = 'ClipGaussian'
            args = {
                'sigma': self.agent_params.sigma,
                'clip_range': self.agent_params.clip_range,
            }

        else:
            explore_name = self.agent_params.explore_policy
            args = {}

        self.explore_policy = self.create_explorer(
            explore_name=explore_name,
            shape=self.actor.output_info['action'],
            args=args,
        )

        # 初始化模型同步器
        self._sync_model_dict = {
            'actor': self.actor,
        }

    def optimize(self, data_set: RLStandardDataSet):

        train_data, reward_info = self._episode_tool(data_set, shuffle=self.agent_params.shuffle)

        # self.replay_buffer.add_data_by_buffer(train_data)

        for _ in range(self.agent_params.n_updates):

            sample_data = self.expert_dataset.random_sample_all(self.agent_params.batch_size, tf_dataset=False)  # dict

            # compute target y

            # get next actor input

            next_actor_input = self.get_corresponding_data(sample_data, self.actor.input_name, 'next_')
            next_q_input = self.get_corresponding_data(sample_data, self.q_critit1.input_name, 'next_',
                                                       filter='next_action')
            current_q_input = self.get_corresponding_data(sample_data, self.q_critit1.input_name, filter='action')
            current_actor_input = self.get_corresponding_data(sample_data, self.actor.input_name)

            reward = tf.convert_to_tensor(sample_data['reward'], dtype=tf.float32)
            mask = tf.convert_to_tensor(sample_data['mask'], dtype=tf.float32)

            target_y = self.compute_target_y(
                next_actor_input=next_actor_input,
                next_q_input=next_q_input,
                reward=reward,
                mask=mask,
                noise_clip_range=self.agent_params.noise_clip_range,
                gamma=self.agent_params.gamma,
            )

            current_action = self.actor(*current_actor_input)

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

    def compute_target_y(self,
                         next_actor_input,
                         next_q_input,  # state no action
                         reward,
                         mask,
                         noise_clip_range=0.5,
                         gamma=0.99

                         ):
        # compute next action
        next_target_action = self.target_actor(*next_actor_input)

        size = next_target_action.shape[0]

        # add noise
        org_noise = self.explore_policy.dist.sample(size)
        clip_noise = tf.clip_by_value(org_noise, -noise_clip_range, noise_clip_range)

        noise_next_target_action = next_target_action + clip_noise

        q1 = self.target_q_critic1(*next_q_input, noise_next_target_action)
        q2 = self.target_q_critic2(*next_q_input, noise_next_target_action)

        minmun_q = tf.minimum(q1, q2)

        target_y = reward + gamma * mask * minmun_q

        return target_y

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

        self.q_critit1.optimizer.apply_gradients(zip(grads, self.q_critit1.trainable_variables))

        dict_info = {
            'critic1_loss': loss,
        }

        return dict_info

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

        self.q_critit1.optimizer.apply_gradients(zip(grads, self.q_critit2.trainable_variables))

        dict_info = {
            'critic2_loss': loss,
        }

        return dict_info

    def train_actor(self,
                    current_actor_input,
                    ):

        with tf.GradientTape() as tape:
            tape.watch(self.actor.trainable_variables)

            current_action = self.actor(*current_actor_input)

            q1 = self.q_critit1(*current_actor_input, current_action)

            loss = -tf.reduce_mean(q1)

        grads = tape.gradient(loss, self.actor.trainable_variables)

        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        dict_info = {
            'actor_loss': loss,
        }

        return dict_info
