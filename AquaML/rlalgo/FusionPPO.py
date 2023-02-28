from AquaML.rlalgo.BaseRLAlgo import BaseRLAlgo
from AquaML.rlalgo.Parameters import FusionPPO_parameter
from AquaML.DataType import RLIOInfo
import tensorflow as tf
import warnings
import copy


class FusionPPO(BaseRLAlgo):
    def __init__(self,
                 env,
                 rl_io_info: RLIOInfo,
                 parameters: FusionPPO_parameter,
                 actor,
                 critic,
                 computer_type: str = 'PC',
                 name: str = 'PPO',
                 level: int = 0,
                 thread_id: int = -1,
                 total_threads: int = 1, ):
        """
        We recommend to use this algorithm for POMDP env.
        :arg
        :param env:
        :param rl_io_info:
        :param parameters:
        :param actor:
        :param critic:
        :param computer_type:
        :param name:
        :param level:
        :param thread_id:
        :param total_threads:
        """
        super().__init__(
            env=env,
            rl_io_info=rl_io_info,
            name=name,
            hyper_parameters=parameters,
            update_interval=parameters.update_interval,
            computer_type=computer_type,
            level=level,
            thread_ID=thread_id,
            total_threads=total_threads,
        )

        self.hyper_parameters = parameters

        self.actor = actor()

        self.initialize_actor_config()

        # check algo config status
        if self.rnn_actor_flag is not True:
            warnings.warn('The best performance of PPO is achieved with RNN actor in POMDP env.')

        self.initialize_model_weights(self.actor, self.rnn_actor_flag)

        if self.level == 0:
            if self.resample_log_prob is None:
                raise ValueError(
                    'resample_log_prob must be set when using PPO. You probably make an actor with output ''log_prob''.')

            self.critic = critic()

            # judge number hidden state
            self.hidden_state_num = 0
            for input_name in self.actor.input_name:
                if 'hidden' in input_name:
                    self.hidden_state_num += 1

            # initialize actor and critic

            self.initialize_model_weights(self.critic)

            # get fusion value idx
            self.fusion_value_idx = 0
            idx = 0
            fusion_flag = False
            actor_output_names = tuple(self.actor.output_info.keys())
            for name in actor_output_names:
                if 'value' in name:
                    self.fusion_value_idx = idx
                    fusion_flag = True
                    break
                idx += 1
            # self.fusion_value_idx += 1
            if not fusion_flag:
                raise ValueError('Fusion value must be in actor output. '
                                 'Please check your actor output.')
            self.fusion_value_idx += 1

            # all models dict
            self._all_model_dict = {
                'actor': self.actor,
                'critic': self.critic,
            }
        # else:
        # self.actor = actor()

        # initialize actor
        # self.initialize_model_weights(self.actor)

        self._sync_model_dict['actor'] = self.actor

        # create optimizer
        if self.level == 0:
            self.create_optimizer(name='actor', optimizer=self.actor.optimizer, lr=self.actor.learning_rate)
            self.create_optimizer(name='critic', optimizer=self.critic.optimizer, lr=self.critic.learning_rate)
        else:
            self.actor_optimizer = None
            self.critic_optimizer = None

        # create exploration policy
        self.create_gaussian_exploration_policy()

    @tf.function
    def train_critic(self,
                     critic_obs: tuple,
                     target: tf.Tensor,
                     ):
        with tf.GradientTape() as tape:
            v = self.critic(*critic_obs)
            critic_loss = tf.reduce_mean(tf.square(v - target))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        dic = {
            'critic_loss': critic_loss,
        }
        return dic

    @tf.function
    def train_actor(self,
                    actor_obs: tuple,
                    advantage: tf.Tensor,
                    old_log_prob: tf.Tensor,
                    action: tf.Tensor,
                    target: tf.Tensor,
                    lam,
                    epsilon,
                    entropy_coefficient,
                    ):
        with tf.GradientTape() as tape:
            out = self.resample_log_prob(actor_obs, action)
            log_prob = out[0]
            fusion_value = out[self.fusion_value_idx]

            ratio = tf.exp(log_prob - old_log_prob)

            actor_surrogate_loss = tf.reduce_mean(
                tf.minimum(
                    ratio * advantage,
                    tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon) * advantage,
                )
            )

            fusion_value_loss = tf.reduce_mean(tf.square(fusion_value - target))

            entropy_loss = -tf.reduce_mean(log_prob)

            loss = -actor_surrogate_loss + lam * fusion_value_loss - entropy_coefficient * entropy_loss

        actor_grad = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        # with tf.GradientTape() as tape:
        #     out = self.resample_log_prob(actor_obs, action)
        #     log_prob = out[0]
        #     fusion_value = out[self.fusion_value_idx]
        #
        #     ratio = tf.exp(log_prob - old_log_prob)
        #
        #     actor_surrogate = tf.minimum(
        #         ratio * advantage,
        #         tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon) * advantage,
        #     )
        #
        #     entropy = -log_prob
        #     fusion_value_d = tf.square(fusion_value - target)
        #
        #     normalized_surrogate_loss = tf.reduce_mean(tf.math.l2_normalize(actor_surrogate, axis=0))
        #
        #     normalized_entropy_loss = tf.reduce_mean(tf.math.l2_normalize(entropy, axis=0))
        #
        #     normalized_fusion_value_loss = tf.reduce_mean(tf.math.l2_normalize(fusion_value_d, axis=0))
        #
        #     normalized_loss = -normalized_surrogate_loss + lam * normalized_fusion_value_loss - entropy_coefficient * normalized_entropy_loss
        #
        # normalized_actor_grad = tape.gradient(normalized_loss, self.actor.trainable_variables)
        # self.actor_optimizer.apply_gradients(zip(normalized_actor_grad, self.actor.trainable_variables))

        # dic = {
        #     'actor_surrogate_loss': tf.reduce_mean(actor_surrogate),
        #     'actor_loss': normalized_loss,
        #     'fusion_value_loss': tf.reduce_mean(fusion_value_d),
        #     'entropy_loss': tf.reduce_mean(entropy),
        #     # 'normalized_actor_loss': normalized_loss,
        #     'normalized_actor_surrogate_loss': normalized_surrogate_loss,
        #     'normalized_fusion_value_loss': normalized_fusion_value_loss,
        #     'normalized_entropy_loss': normalized_entropy_loss,
        # }
        dic = {
            'actor_surrogate_loss': actor_surrogate_loss,
            'actor_loss': loss,
            'fusion_value_loss': fusion_value_loss,
            'entropy_loss': entropy_loss,
        }
        return dic

    def _optimize_(self):
        data_dict = self.get_all_data

        # get critic obs
        critic_obs = self.get_corresponding_data(data_dict=data_dict, names=self.critic.input_name)
        next_critic_obs = self.get_corresponding_data(data_dict=data_dict, names=self.critic.input_name,
                                                      prefix='next_')

        # get actor obs
        actor_obs = self.get_corresponding_data(data_dict=data_dict, names=self.actor.input_name)
        # next_actor_obs = self.get_corresponding_data(data_dict=data_dict, names=self.actor.input_name, prefix='next_')

        # get total reward
        rewards = data_dict['total_reward']

        # get old prob
        old_prob = data_dict['prob']

        # get mask
        masks = data_dict['mask']

        # get action
        actions = data_dict['action']

        #######calculate advantage and target########
        # get value

        if 'value' in self.actor.output_info:
            values = data_dict['value']
            next_values = data_dict['next_value']
        else:
            values = self.critic(*critic_obs).numpy()
            next_values = self.critic(*next_critic_obs).numpy()

            # get target and advantage
        advantage, target = self.calculate_GAE(rewards=rewards,
                                               values=values,
                                               next_values=next_values,
                                               masks=masks,
                                               gamma=self.hyper_parameters.gamma,
                                               lamda=self.hyper_parameters.lambada
                                               )

        # convert to tensor
        advantage = tf.convert_to_tensor(advantage, dtype=tf.float32)
        target = tf.convert_to_tensor(target, dtype=tf.float32)
        # rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        # masks = tf.convert_to_tensor(masks, dtype=tf.float32)
        old_prob = tf.convert_to_tensor(old_prob, dtype=tf.float32)
        old_log_prob = tf.math.log(old_prob)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)

        train_actor_input = {
            'actor_obs': actor_obs,
            'advantage': advantage,
            'old_log_prob': old_log_prob,
            'action': actions,
            'target': copy.deepcopy(target),
        }

        train_critic_input = {
            'critic_obs': critic_obs,
            'target': target,
        }

        if self.rnn_actor_flag:
            if self.hyper_parameters.batch_trajectory:
                train_actor_input = self.get_batch_timesteps(train_actor_input)
                actor_obs = train_actor_input['actor_obs']  # input

                hidden_lists = []

                for key, shape in self.actor.output_info.items():
                    if 'hidden' in key:
                        hidden = tf.zeros(shape=(actor_obs[0].shape[0], shape[0]), dtype=tf.float32)
                        hidden_lists.append(hidden)

                actor_obs = (*actor_obs[:-self.hidden_state_num], *hidden_lists)
                train_actor_input['actor_obs'] = actor_obs

            else:
                for idx in self.expand_dims_idx:
                    actor_obs[idx] = tf.expand_dims(actor_obs[idx], axis=1)
        info_list = []
        buffer_size = train_actor_input['actor_obs'][0].shape[0]

        if self.hyper_parameters.batch_trajectory:
            critic_batch_steps = self.hyper_parameters.batch_size * train_actor_input['actor_obs'][0].shape[1]
        else:
            critic_batch_steps = self.hyper_parameters.batch_size

        critic_buffer_size = self.hyper_parameters.buffer_size

        for _ in range(self.hyper_parameters.update_times):
            # fusion ppo firstly update critic
            start_index = 0
            end_index = 0
            critic_start_index = 0
            while end_index < buffer_size:
                end_index = min(start_index + self.hyper_parameters.batch_size,
                                buffer_size)
                critic_end_index = min(critic_start_index + critic_batch_steps, critic_buffer_size)
                critic_optimize_info_list = []
                actor_optimize_info_list = []
                batch_train_actor_input = self.get_batch_data(train_actor_input, start_index, end_index)
                batch_train_critic_input = self.get_batch_data(train_critic_input, critic_start_index, critic_end_index)
                start_index = end_index
                critic_start_index = critic_end_index
                for _ in range(self.hyper_parameters.update_critic_times):
                    critic_optimize_info = self.train_critic(
                        critic_obs=batch_train_critic_input['critic_obs'],
                        target=batch_train_critic_input['target'],
                    )
                    critic_optimize_info_list.append(critic_optimize_info)

                critic_value = self.critic(*batch_train_critic_input['critic_obs'])
                critic_value_target = tf.math.reduce_mean(tf.square(critic_value - batch_train_critic_input['target']))

                out = self.resample_log_prob(batch_train_actor_input['actor_obs'], batch_train_actor_input['action'])
                fusion_value = out[self.fusion_value_idx]

                # fusion_value = tf.reshape(fusion_value, critic_value.shape)
                critic_value = tf.reshape(critic_value, shape=fusion_value.shape)

                fusion_value_critic = tf.math.reduce_mean(tf.square(fusion_value - critic_value))

                # distance = tf.sqrt(critic_value_target) + tf.sqrt(fusion_value_critic)
                distance = critic_value_target + fusion_value_critic

                lam = 1. / distance
                lam = tf.clip_by_value(lam, 0, 0.2)
                # lam = 1
                lam = 0

                for _ in range(self.hyper_parameters.update_actor_times):
                    actor_optimize_info = self.train_actor(
                        actor_obs=batch_train_actor_input['actor_obs'],
                        advantage=batch_train_actor_input['advantage'],
                        old_log_prob=batch_train_actor_input['old_log_prob'],
                        action=batch_train_actor_input['action'],
                        target=batch_train_actor_input['target'],
                        lam=lam,
                        epsilon=tf.cast(self.hyper_parameters.epsilon, dtype=tf.float32),
                        entropy_coefficient=tf.cast(self.hyper_parameters.entropy_coeff, dtype=tf.float32),
                    )
                    actor_optimize_info_list.append(actor_optimize_info)
                critic_optimize_info = self.cal_average_batch_dict(critic_optimize_info_list)
                actor_optimize_info = self.cal_average_batch_dict(actor_optimize_info_list)
                info = {**critic_optimize_info, **actor_optimize_info, 'lam': lam}
                info_list.append(info)

            info = self.cal_average_batch_dict(info_list)

            return info

        #     for _ in range(self.hyper_parameters.update_critic_times):
        #         start_index = 0
        #         end_index = 0
        #         critic_optimize_info_list = []
        #         for _ in range(self.hyper_parameters.update_critic_times):
        #             while end_index < self.hyper_parameters.buffer_size:
        #                 end_index = min(start_index + self.hyper_parameters.batch_size,
        #                                 self.hyper_parameters.buffer_size)
        #
        #                 batch_train_critic_input = self.get_batch_data(train_critic_input, start_index, end_index)
        #
        #                 start_index = end_index
        #
        #                 critic_optimize_info = self.train_critic(
        #                     critic_obs=batch_train_critic_input['critic_obs'],
        #                     target=batch_train_critic_input['target'],
        #                 )
        #                 critic_optimize_info_list.append(critic_optimize_info)
        #
        #
        #
        #     # fusion ppo secondly update actor
        #     # compute lam
        #     critic_value = self.critic(*critic_obs)
        #     critic_value_target = tf.reduce_mean(tf.square(critic_value - target))
        #
        #     out = self.resample_log_prob(actor_obs, train_actor_input['action'])
        #     fusion_value = out[self.fusion_value_idx]
        #
        #     fusion_value = tf.reshape(fusion_value, critic_value.shape)
        #
        #     fusion_value_critic = tf.reduce_mean(tf.square(fusion_value - critic_value))
        #
        #     distance = tf.sqrt(critic_value_target) + tf.sqrt(fusion_value_critic)
        #
        #     batch_size = train_actor_input['actor_obs'][0].shape[0]
        #
        #     lam = 1. / distance
        #     # lam = 0
        #     for _ in range(self.hyper_parameters.update_actor_times):
        #         start_index = 0
        #         end_index = 0
        #         actor_optimize_info_list = []
        #         for _ in range(self.hyper_parameters.update_actor_times):
        #             while end_index < batch_size:
        #                 end_index = min(start_index + self.hyper_parameters.batch_size,
        #                                 batch_size)
        #
        #                 batch_train_actor_input = self.get_batch_data(train_actor_input, start_index, end_index)
        #
        #                 start_index = end_index
        #
        #                 actor_optimize_info = self.train_actor(
        #                     actor_obs=batch_train_actor_input['actor_obs'],
        #                     advantage=batch_train_actor_input['advantage'],
        #                     old_log_prob=batch_train_actor_input['old_log_prob'],
        #                     action=batch_train_actor_input['action'],
        #                     target=batch_train_actor_input['target'],
        #                     lam=lam,
        #                     epsilon=tf.cast(self.hyper_parameters.epsilon, dtype=tf.float32),
        #                     entropy_coefficient=tf.cast(self.hyper_parameters.entropy_coeff, dtype=tf.float32),
        #                 )
        #                 actor_optimize_info_list.append(actor_optimize_info)
        #             actor_optimize_info = self.cal_average_batch_dict(actor_optimize_info_list)
        #
        # return_dict = {**critic_optimize_info, **actor_optimize_info, 'lam': lam}
        # return return_dict
