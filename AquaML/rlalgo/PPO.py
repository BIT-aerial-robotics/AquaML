from AquaML.rlalgo.BaseRLAlgo import BaseRLAlgo
from AquaML.DataType import RLIOInfo
from AquaML.rlalgo.ExplorePolicy import VoidExplorePolicy
import tensorflow as tf
from AquaML.rlalgo.Parameters import PPO_parameter


class PPO(BaseRLAlgo):

    def __init__(self,
                 env,
                 rl_io_info: RLIOInfo,
                 parameters: PPO_parameter,
                 actor,
                 critic,
                 computer_type: str = 'PC',
                 name: str = 'PPO',
                 prefix_name: str = None,
                 level: int = 0,
                 thread_id: int = -1,
                 total_threads: int = 1, ):
        """
        Proximal Policy Optimization (PPO) algorithm is an on-policy algorithm.

        Reference:
        ----------
        [1] Schulman, John, et al. "Proximal policy optimization algorithms."
            arXiv preprint arXiv:1707.06347 (2017).

        Args:
            env (gym.Env): gym environment.
            rl_io_info (RLIOInfo): store input and output information.
            parameters (PPO_parameter): parameters.
            actor (BaseModel): actor network.
            critic (BaseModel): critic network.
            computer_type (str, optional): 'PC' or 'GPU'. Defaults to 'PC'.
            name (str, optional): name. Defaults to 'SAC'.
            level (int, optional): level. Defaults to 0.
            thread_id (int, optional): thread id. Defaults to -1.
            total_threads (int, optional): total threads. Defaults to 1.
        """

        super().__init__(
            env=env,
            rl_io_info=rl_io_info,
            name=name,
            hyper_parameters=parameters,
            update_interval=parameters.update_interval,
            calculate_episodes=parameters.calculate_episodes,
            display_interval=parameters.display_interval,
            computer_type=computer_type,
            level=level,
            thread_ID=thread_id,
            prefix_name=prefix_name,
            total_threads=total_threads,
        )

        self.hyper_parameters = parameters

        self.actor = actor()
        self.initialize_actor_config()
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
            # self.initialize_model_weights(self.actor)
            self.initialize_model_weights(self.critic)

            # all models dict
            self._all_model_dict = {
                'actor': self.actor,
                'critic': self.critic,
            }

        # create optimizer
        if self.level == 0:
            self.create_optimizer(name='actor', optimizer=self.actor.optimizer, lr=self.actor.learning_rate)
            self.create_optimizer(name='critic', optimizer=self.critic.optimizer, lr=self.critic.learning_rate)
        else:
            self.actor_optimizer = None
            self.critic_optimizer = None

        # create exploration policy
        self.create_explore_policy()
        # self.explore_policy = VoidExplorePolicy(shape=self.rl_io_info.actor_out_info['action'])


        self._sync_model_dict['actor'] = self.actor

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
            # 'critic_grad': critic_grad,
        }
        return dic

    def compute_critic_loss(self,
                            critic_obs: tuple,
                            target: tf.Tensor,
                            ):
        v = self.critic(*critic_obs)
        critic_loss = tf.reduce_mean(tf.square(v - target))

        return critic_loss

    @tf.function
    def train_actor(self,
                    actor_obs: tuple,
                    advantage: tf.Tensor,
                    old_log_prob: tf.Tensor,
                    action: tf.Tensor,
                    epsilon: float = 0.2,
                    entropy_coefficient: float = 0.01,
                    ):

        with tf.GradientTape() as tape:
            out = self.resample_log_prob(actor_obs, action)

            log_prob = out[0]

            # importance sampling
            ratio = tf.exp(log_prob - old_log_prob)

            actor_surrogate_loss = tf.reduce_mean(
                tf.minimum(
                    ratio * advantage,
                    tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon) * advantage,
                )
            )

            # surr1 = ratio * advantage
            # surr2 = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon) * advantage
            # actor_surr_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # entropy loss
            entropy_loss = -tf.reduce_mean(log_prob)

            actor_loss = -actor_surrogate_loss - entropy_coefficient * entropy_loss

        actor_grad = tape.gradient(actor_loss, self.get_trainable_actor)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.get_trainable_actor))

        dic = {
            'actor_loss': actor_loss,
            'actor_surr_loss': actor_surrogate_loss,
            'entropy_loss': entropy_loss,
        }

        return dic

    def compute_actor_loss(self,
                           actor_obs: tuple,
                           advantage: tf.Tensor,
                           old_log_prob: tf.Tensor,
                           action: tf.Tensor,
                           epsilon:tf.Tensor,
                           entropy_coefficient: tf.Tensor,
                           ):
        out = self.resample_log_prob(actor_obs, action)

        log_prob = out[0]

        # importance sampling
        ratio = tf.exp(log_prob - old_log_prob)

        actor_surrogate_loss = tf.reduce_mean(
            tf.minimum(
                ratio * advantage,
                tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon) * advantage,
            )
        )

        entropy_loss = -tf.reduce_mean(log_prob)

        actor_loss = -actor_surrogate_loss - entropy_coefficient * entropy_loss

        return actor_loss

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

        if self.hyper_parameters.batch_advantage_normlization:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

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
        critic_buffer_size = self.hyper_parameters.buffer_size
        critic_batch_steps = self.hyper_parameters.batch_size

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

                for _ in range(self.hyper_parameters.update_actor_times):
                    actor_optimize_info = self.train_actor(
                        actor_obs=batch_train_actor_input['actor_obs'],
                        advantage=batch_train_actor_input['advantage'],
                        old_log_prob=batch_train_actor_input['old_log_prob'],
                        action=batch_train_actor_input['action'],
                        epsilon=self.hyper_parameters.epsilon,
                        entropy_coefficient=self.hyper_parameters.entropy_coeff,
                    )
                    actor_optimize_info_list.append(actor_optimize_info)
                # stable update_actor_times
                if self.hyper_parameters.update_actor_times == 0:
                    actor_optimize_info_ = {
                            'actor_loss': tf.constant(0, dtype=tf.float32),
                            'actor_surr_loss': tf.constant(0, dtype=tf.float32),
                            'entropy_loss': tf.constant(0, dtype=tf.float32),
                        }
                    actor_optimize_info_list.append(actor_optimize_info_)
                critic_optimize_info = self.cal_average_batch_dict(critic_optimize_info_list)
                actor_optimize_info = self.cal_average_batch_dict(actor_optimize_info_list)
                info = {**critic_optimize_info, **actor_optimize_info}
                info_list.append(info)

            info = self.cal_average_batch_dict(info_list)

            return info

        # for _ in range(self.hyper_parameters.update_times):
        #     # train actor
        #     # TODO: wrap this part into a function
        #     for _ in range(self.hyper_parameters.update_actor_times):
        #         start_index = 0
        #         end_index = 0
        #         actor_optimize_info_list = []
        #         while end_index < self.hyper_parameters.buffer_size:
        #             end_index = min(start_index + self.hyper_parameters.batch_size, self.hyper_parameters.buffer_size)
        #
        #             batch_train_actor_input = self.get_batch_data(train_actor_input, start_index, end_index)
        #
        #             start_index = end_index
        #
        #             actor_optimize_info = self.train_actor(
        #                 actor_obs=batch_train_actor_input['actor_obs'],
        #                 advantage=batch_train_actor_input['advantage'],
        #                 old_log_prob=batch_train_actor_input['old_log_prob'],
        #                 action=batch_train_actor_input['action'],
        #                 epsilon=tf.cast(self.hyper_parameters.epsilon, dtype=tf.float32),
        #                 entropy_coefficient=tf.cast(self.hyper_parameters.entropy_coeff, dtype=tf.float32),
        #             )
        #             actor_optimize_info_list.append(actor_optimize_info)
        #         actor_optimize_info = self.cal_average_batch_dict(actor_optimize_info_list)
        #
        #     # train critic
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
        #             critic_optimize_info = self.cal_average_batch_dict(critic_optimize_info_list)
        #
        # return_dict = {**actor_optimize_info, **critic_optimize_info}
        # return return_dict
