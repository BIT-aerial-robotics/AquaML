import tensorflow as tf
from AquaML.rlalgo.BaseRLAgent import BaseRLAgent
from AquaML.rlalgo.AgentParameters import IQLAgentParameters
from AquaML.core.RLToolKit import RLStandardDataSet
from AquaML.core.DataParser import DataSet

import os
import numpy as np


class IQLAgent(BaseRLAgent):

    def __init__(self,
                 name,
                 actor,
                 q_critic,
                 s_value,
                 expert_dataset_path,
                 agent_params: IQLAgentParameters,   # TODO 需要设置
                 level=0
                 ):

        super().__init__(
            name=name,
            level=level,
            agent_params=agent_params,
        )

        self.actor = actor()

        self.q_critic = q_critic()
        self.target_q_critic = q_critic()

        self.s_value = s_value()

        self.n_update_times = 0

        self.expert_dataset_path = expert_dataset_path

    def init(self):

        self.initialize_actor()

        if self.level == 0:

            # 创建q网络
            self.initialize_model(
                model_class=self.q_critic,
                name='q_critic',
            )

            # 创建状态价值网络
            self.initialize_model(
                model_class=self.s_value,
                name='s_value',
            )

            # 创建target网络
            self.initialize_model(
                model_class=self.target_q_critic,
                name='target_q_critic',
            )

            # copy参数
            self.copy_weights(
                self.q_critic,
                self.target_q_critic,
            )

            # 创建优化器

            if hasattr(self.actor, 'optimizer_info'):
                self.create_optimizer(
                    self.actor.optimizer_info,
                    'actor_optimizer'
                )
            else:
                raise ValueError('optimizer_info is not defined in actor')

            if hasattr(self.q_critic, 'optimizer_info'):
                self.create_optimizer(
                    self.q_critic.optimizer_info,
                    'q_critic_optimizer'
                )
            else:
                raise ValueError('optimizer_info is not defined in agent_params')

            if hasattr(self.s_value, 'optimizer_info'):
                self.create_optimizer(
                    self.s_value.optimizer_info,
                    's_value_optimizer'
                )
            else:
                raise ValueError('optimizer_info is not defined in agent_params')

            self._all_model_dict = {
                'actor': self.actor,
                'q_critic': self.q_critic,
                'target_q_critic': self.target_q_critic,
                's_value': self.s_value,
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
                next_data_file_path = os.path.join(self.expert_dataset_path, 'next_' + key + '.npy')

                obs = np.load(data_file_path)
                next_obs = np.load(next_data_file_path)

                if self.agent_params.normalize:
                    mean = np.mean(obs,axis=0)
                    std = np.std(obs, axis=0)
                    obs = (obs - mean) / (std + 1e-8)
                    next_obs = (next_obs - mean) / (std + 1e-8)
                expert_dataset[key] = obs
                expert_dataset['next_' + key] = next_obs

            # load expert action
            data_file_path = os.path.join(self.expert_dataset_path, 'action.npy')

            action = np.load(data_file_path)

            if self.agent_params.normalize:
                mean = np.mean(action, axis=0)
                std = np.std(action, axis=0)
                action = (action - mean) / (std+1e-8)

            expert_dataset['action'] = action

            # if self.agent_params.normalize:
            #     expert_dataset

            # load reward
            data_file_path = os.path.join(self.expert_dataset_path, 'total_reward.npy')

            total_reward = np.load(data_file_path)

            if self.agent_params.normalize_reward:
                total_reward = (total_reward - np.mean(total_reward)) / np.std(total_reward)
            expert_dataset['total_reward'] = total_reward

            # load mask
            data_file_path = os.path.join(self.expert_dataset_path, 'mask.npy')
            expert_dataset['mask'] = np.load(data_file_path)

            self.expert_dataset = DataSet(
                data_dict=expert_dataset
            )

            # get action bound
            self.action_max = np.max(expert_dataset['action'])
            self.action_min = np.min(expert_dataset['action'])

        # 创建探索策略
        if self.agent_params.explore_policy == 'Default':
            explore_name = 'ClipGaussian'
            args = {
                'sigma': self.agent_params.sigma,
                'action_high': self.action_max,
                'action_low': self.action_min,
            }

        else:
            explore_name = self.agent_params.explore_policy
            args = {}

        self.create_explorer(
            explore_name=explore_name,
            shape=self.actor.output_info['action'],
            args=args,
        )

        # TODO 多线程
        # 初始化模型同步器
        self._sync_model_dict = {
            'actor': self.actor,
        }

    def optimize(self, **kwargs):

        # train_data, reward_info = self._episode_tool(data_set, shuffle=self.agent_params.shuffle)

        # self.replay_buffer.add_data_by_buffer(train_data)

        for _ in range(self.agent_params.update_times):

            sample_data = self.expert_dataset.random_sample_all(self.agent_params.batch_size, tf_dataset=False)  # dict

                # get next actor input

            next_actor_input = self.get_corresponding_data(sample_data, self.actor.input_name, 'next_')
            next_q_input = self.get_corresponding_data(sample_data, self.q_critic.input_name, 'next_',
                                                       filter='next_action')
            current_q_input = self.get_corresponding_data(sample_data, self.q_critic.input_name, filter='action')
            current_actor_input = self.get_corresponding_data(sample_data, self.actor.input_name)

            reward = tf.convert_to_tensor(sample_data['total_reward'], dtype=tf.float32)
            mask = tf.convert_to_tensor(sample_data['mask'], dtype=tf.float32)

            target_action = tf.convert_to_tensor(sample_data['action'], dtype=tf.float32)  # TODO 有什么区别

            current_action = tf.convert_to_tensor(sample_data['action'], dtype=tf.float32)

            # train s_value
            v_info = self.train_s_value(
                current_actor_input=current_actor_input,
                current_action=current_action,
            )
            self.loss_tracker.add_data(v_info)

            # train critic   # TODO 噪声
            q_info = self.train_critic(
                current_q_input=current_q_input,  # state no action
                reward=reward,
                current_action=current_action,
                next_q_input=next_q_input,
                mask=mask,
                gamma=self.agent_params.gamma,
            )

            self.loss_tracker.add_data(q_info)

            self.n_update_times += 1

            if self.n_update_times % self.agent_params.delay_update == 0:
                loss = self.train_actor(
                    current_actor_input=current_actor_input,
                    current_action=current_action,
                    temperature=self.agent_params.temperature  # TODO
                )

                self.loss_tracker.add_data(loss)

                # update target model
                self.soft_update(
                    source_model=self.q_critic,
                    target_model=self.target_q_critic,
                    tau=self.agent_params.tau,
                )

        # summary = self.loss_tracker.get_data()
        return self.loss_tracker, {}

    @tf.function
    def train_s_value(self,
                      current_actor_input,
                      current_action,
                      expectile=0.7
                      ):

        with tf.GradientTape() as tape:
            tape.watch(self.s_value.trainable_variables)

            q1, q2 = self.target_q_critic(*current_actor_input, current_action)
            q = tf.minimum(q1, q2)
            v = self.s_value(*current_actor_input)
            diff = q - v

            weight = tf.where(diff > 0, expectile, (1 - expectile))
            loss = tf.reduce_mean(weight * tf.square(diff))

        grads = tape.gradient(loss, self.s_value.trainable_variables)

        self.s_value_optimizer.apply_gradients(zip(grads, self.s_value.trainable_variables))  # TODO 优化器为什么是NONE

        dict_info = {
            's_value_loss': loss,
        }

        return dict_info

    @tf.function
    def train_critic(self,
                     current_q_input,  # state no action
                     current_action,
                     reward,
                     next_q_input,
                     mask,
                     gamma=0.99
                     ):
        next_v = self.s_value(*next_q_input)
        target_y = reward + gamma * mask * next_v
        with tf.GradientTape() as tape:
            tape.watch(self.q_critic.trainable_variables)

            q1, q2 = self.q_critic(*current_q_input, current_action)

            loss = tf.reduce_mean(tf.square(q1 - target_y) + tf.square(q2 - target_y))

        grads = tape.gradient(loss, self.q_critic.trainable_variables)

        self.q_critic_optimizer.apply_gradients(zip(grads, self.q_critic.trainable_variables))

        dict_info = {
            'critic_loss': loss,
        }

        return dict_info

    @tf.function
    def train_actor(self,
                    current_actor_input,
                    current_action,
                    temperature=3,
                    ):

        with tf.GradientTape() as tape:
            tape.watch(self.actor.trainable_variables)

            v = self.s_value(*current_actor_input)
            q1, q2 = self.target_q_critic(*current_actor_input, current_action)
            q = tf.minimum(q1, q2)
            exp_a = tf.exp((q - v) * temperature)
            exp_a = tf.clip_by_value(exp_a, -100, 100)

            mu = self.actor(*current_actor_input)[0]
            loss = tf.reduce_mean(exp_a * tf.square(mu - current_action))

        grads = tape.gradient(loss, self.actor.trainable_variables)

        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        dict_info = {
            'actor_loss': loss,
        }

        return dict_info



    @staticmethod
    def get_algo_name():
        return 'IQL'


