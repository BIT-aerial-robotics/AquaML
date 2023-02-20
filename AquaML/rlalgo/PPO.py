from AquaML.rlalgo.BaseRLAlgo import BaseRLAlgo
from AquaML.DataType import RLIOInfo
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
                 name: str = 'SAC',
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
            update_interval=parameters.update_interval,
            computer_type=computer_type,
            level=level,
            thread_ID=thread_id,
            total_threads=total_threads,
        )

        self.hyper_parameters = parameters

        if self.level == 0:
            self.actor = actor()
            self.critic = critic()

            # initialize actor and critic
            self.initialize_model_weights(self.actor)
            self.initialize_model_weights(self.critic)

            # all models dict
            self._all_model_dict = {
                'actor': self.actor,
                'critic': self.critic,
            }
        else:
            self.actor = actor()

            # initialize actor
            self.initialize_model_weights(self.actor)

        # create optimizer
        if self.level == 0:
            self.create_optimizer(name='actor', optimizer=self.actor.optimizer, lr=self.actor.learning_rate)
            self.create_optimizer(name='critic', optimizer=self.critic.optimizer, lr=self.critic.learning_rate)
        else:
            self.actor_optimizer = None
            self.critic_optimizer = None

        # create exploration policy
        self.create_gaussian_exploration_policy()

    def train_critic(self,
                     critic_obs: tuple,
                     target: tf.Tensor,
                     ):

        with tf.GradientTape() as tape:
            v = self.critic(critic_obs)
            critic_loss = tf.reduce_mean(tf.square(v - target))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        dic = {
            'critic_loss': critic_loss,
        }
        return dic

    def train_actor(self,
                    actor_obs: tuple,
                    advantage: tf.Tensor,
                    old_log_prob: tf.Tensor,
                    epsilon: float = 0.2,
                    ):

        with tf.GradientTape() as tape:
            action, log_prob = self.resample_action(actor_obs)

            # importance sampling
            ratio = tf.exp(log_prob - old_log_prob)

            surr1 = ratio * advantage
            surr2 = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon) * advantage
            actor_surr_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # entropy loss
            entropy_loss = -tf.reduce_mean(log_prob)

            actor_loss = actor_surr_loss + entropy_loss

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        dic = {
            'actor_loss': actor_loss,
            'actor_surr_loss': actor_surr_loss,
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
        next_actor_obs = self.get_corresponding_data(data_dict=data_dict, names=self.actor.input_name, prefix='next_')

        # get total reward
        rewards = data_dict['total_reward']

        # get old prob
        old_prob = data_dict['prob']

        # get mask
        masks = data_dict['mask']

        #######calculate advantage and target########
        # get value

        if 'value' in self.actor.output_info:
            values = data_dict['value']
            next_values = data_dict['next_value']
        else:
            values = self.critic(*critic_obs).numpy()
            next_values = self.critic(*next_critic_obs).numpy()

        # get target and advantage
        advantage, target = self.calculate_GAE(rewards=masks,
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

        for _ in range(self.hyper_parameters.update_times):
            # train actor
            for _ in range(self.hyper_parameters.update_actor_times):
                start_index = 0
                while True:
                    end_index = min(start_index + self.hyper_parameters.batch_size, self.hyper_parameters.buffer_size)

                    actor_optimize_info = self.train_actor(
                        actor_obs=actor_obs,
                        advantage=advantage,
                        old_log_prob=old_log_prob,
                        epsilon=tf.cast(self.hyper_parameters.epsilon, dtype=tf.float32),
                    )

            # train critic
            for _ in range(self.hyper_parameters.update_critic_times):
                critic_optimize_info = self.train_critic(
                    critic_obs=critic_obs,
                    target=target,
                )

        return_dict = {**actor_optimize_info, **critic_optimize_info}
        return return_dict
