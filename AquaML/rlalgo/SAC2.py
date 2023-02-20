from AquaML.rlalgo.BaseRLAlgo import BaseRLAlgo
from AquaML.DataType import RLIOInfo
from AquaML.rlalgo.Parameters import SAC2_parameter
import tensorflow as tf


class SAC2(BaseRLAlgo):

    def __init__(self,
                 env,
                 rl_io_info: RLIOInfo,
                 parameters: SAC2_parameter,
                 # policy is class do not instantiate
                 actor,  # base class is BasePolicy
                 qf1,
                 qf2,
                 computer_type: str = 'PC',
                 name: str = 'SAC',
                 level: int = 0,
                 thread_id: int = -1,
                 total_threads: int = 1):

        """
        Soft Actor-Critic (SAC) is an algorithm which optimizes a stochastic policy in an off-policy way,
        forming a bridge between stochastic policy optimization and DDPG-style approaches.

        Reference:
        ----------
        [1] Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy
        deep reinforcement learning with a stochastic actor."
        arXiv preprint arXiv:1801.01290 (2018).
        [2] Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications."
        arXiv preprint arXiv:1812.05905 (2018).

        Args:
        env: environment.
        rl_io_info: record the input and output information of the RL algorithm.
        parameters: parameters of the RL algorithm.
        actor: It is a class, not an instance. And it must be a subclass of AquaML.BaseClass.RLBaseModel
        qf1: It is a class, not an instance. And it must be a subclass of AquaML.BaseClass.RLBaseModel
        qf2: It is a class, not an instance. And it must be a subclass of AquaML.BaseClass.RLBaseModel
        -----------------
        Those args are automatically added by the framework:
        computer_type: the type of computer.
        name: the name of the RL algorithm.
        level: the level of the RL algorithm.
        total_threads: the total number of threads.
        thread_ID: the ID of the thread.

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
            mini_buffer_size=parameters.mini_buffer_size,
            display_interval=parameters.display_interval,
            calculate_episodes=parameters.calculate_episodes,
        )

        # TODO: initialize the network in the future
        # Notice: qf just be used  in main thread, actor for all threads
        if self.level == 0:
            # main thread
            self.actor = actor()
            self.qf1 = qf1()
            self.qf2 = qf2()

            # create target network
            self.target_qf1 = qf1()
            self.target_qf2 = qf2()

            # initialize the network
            self.initialize_model_weights(self.actor)
            self.initialize_model_weights(self.qf1)
            self.initialize_model_weights(self.qf2)
            self.initialize_model_weights(self.target_qf1)
            self.initialize_model_weights(self.target_qf2)

            # create tf.Variable for temperature parameter

            self.tf_log_alpha = tf.Variable(tf.math.log(0.01), dtype=tf.float32, trainable=True)

            # copy the weights
            self.copy_weights(self.qf1, self.target_qf1)
            self.copy_weights(self.qf2, self.target_qf2)

            self._all_model_dict = {'actor': self.actor,
                                    'qf1': self.qf1,
                                    'qf2': self.qf2,
                                    'target_qf1': self.target_qf1,
                                    'target_qf2': self.target_qf2, }
        else:
            self.actor = actor()

            # initialize the network
            self.initialize_model_weights(self.actor)

            # None
            self.qf1 = None
            self.qf2 = None
            self.target_qf1 = None
            self.target_qf2 = None

            self.tf_log_alpha = None

        # create the optimizer
        if self.level == 0:
            self.create_optimizer(name='actor', optimizer=self.actor.optimizer,
                                  lr=self.actor.learning_rate)
            self.create_optimizer(name='qf1', optimizer=self.qf1.optimizer,
                                  lr=self.qf1.learning_rate)
            self.create_optimizer(name='qf2', optimizer=self.qf2.optimizer,
                                  lr=self.qf2.learning_rate)

            self.create_optimizer(name='alpha', optimizer='Adam', lr=parameters.alpha_learning_rate)
        else:
            # create the none optimizer
            self.actor_optimizer = None
            self.qf1_optimizer = None
            self.qf2_optimizer = None
            self.alpha_optimizer = None

        # create gaussian noise
        self.create_gaussian_exploration_policy()

            # target entropy
        self.target_entropy = -tf.constant(self.rl_io_info.actor_out_info['action'], dtype=tf.float32)

        self.hyper_parameters = parameters

        self._sync_model_dict = {'actor': self.actor, }

    @tf.function
    def train_q_fun(self, qf_obs: tuple,
                    next_qf_obs: tuple,
                    next_actor_obs: tuple,
                    reward: tf.Tensor,
                    mask: tf.Tensor,
                    action: tf.Tensor,
                    gamma,
                    # alpha,
                    # epoch: int,
                    # recoder,
                    ):
        """
        
        train the q function

        Args:
            qf_obs (tuple): input of q function
            next_qf_obs (tuple): input of q function
            actor_obs (tuple): input of actor
            next_actor_obs (tuple): next input of actor
            reward (tf.Tensor): reward
            mask (tf.Tensor): mask
            gamma (float): hyperparameter, gamma.
            epoch (int): epoch
            recoder: recoder

        Returns:
            info: dict, information of training
        """
        next_action, next_log_pi = self.resample_action(next_actor_obs)
        # log_pi, action = self.resample_action(actor_obs)

        # compute min Q_target(s',a')
        next_q_target1 = self.target_qf1(*next_qf_obs, next_action)
        next_q_target2 = self.target_qf2(*next_qf_obs, next_action)
        min_q_target = tf.minimum(next_q_target1, next_q_target2)

        soft_q_target = min_q_target - tf.exp(self.tf_log_alpha) * next_log_pi
        y = tf.stop_gradient(reward + gamma * mask * soft_q_target)
        # compute y(r,s',d)
        # y = reward + mask * gamma * (min_q_target - tf.exp(self.tf_log_alpha) * next_log_pi)

        with tf.GradientTape() as tape1:
            # compute Q(s,a)
            q1 = self.qf1(*qf_obs, action)
            q1_loss = tf.reduce_mean(tf.square(q1 - y))

        q1_grad = tape1.gradient(q1_loss, self.qf1.trainable_variables)
        self.qf1_optimizer.apply_gradients(zip(q1_grad, self.qf1.trainable_variables))

        with tf.GradientTape() as tape2:
            # compute Q(s,a)
            q2 = self.qf2(*qf_obs, action)
            q2_loss = tf.reduce_mean(tf.square(q2 - y))

        q2_grad = tape2.gradient(q2_loss, self.qf2.trainable_variables)
        self.qf2_optimizer.apply_gradients(zip(q2_grad, self.qf2.trainable_variables))

        target_q = tf.reduce_mean(y)
        q1 = tf.reduce_mean(q1)
        q2 = tf.reduce_mean(q2)

        # return dict
        return_dict = {'q1_loss': q1_loss,
                       'q2_loss': q2_loss,
                       'soft_q_target': target_q,
                       # 'q1': q1,
                       # 'q2': q2,
                       # 'next_action': tf.reduce_mean(next_action),
                       # 'next_log_pi': tf.reduce_mean(next_log_pi),
                       # 'action': tf.reduce_mean(action),
                       # 'alpha': tf.exp(self.tf_log_alpha),
                       }
        # if self.optimize_epoch % 10 == 0:
        #     self.recoder.record(return_dict, self.optimize_epoch, 'q_fun')
        return return_dict

    @tf.function
    def train_alpha(self, actor_obs: tuple,
                    # epoch: int,
                    # recoder
                    ):
        """
        train the alpha

        Args:
            actor_obs (tuple): input of actor
            epoch (int): epoch
            recoder (Recoder): recoder

        Returns:
            info: optional, information of training
        """

        with tf.GradientTape() as tape:
            action, log_pi = self.resample_action(actor_obs)
            alpha_loss = -tf.reduce_mean(tf.exp(self.tf_log_alpha) * (log_pi + self.target_entropy))

        alpha_grad = tape.gradient(alpha_loss, [self.tf_log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.tf_log_alpha]))

        return_dict = {'alpha_loss': alpha_loss,}

        # recoder.record(return_dict, epoch, 'alpha')

        return return_dict

    @tf.function
    def train_actor(self, q_obs: tuple, actor_obs: tuple,
                    # epoch: int,
                    # recoder
                    ):

        """
        train actor
        
        Args:
        q_obs: input of q function.
        actor_obs: input of actor.
        epoch: epoch.
        recoder: recoder.

        Returns:
            info: dict, information of actor.
        """
        # compute log_pi

        with tf.GradientTape() as tape:
            action, log_pi = self.resample_action(actor_obs)

            # compute min Q(s,a)
            q1 = self.qf1(*q_obs, action)
            q2 = self.qf2(*q_obs, action)
            min_q = tf.minimum(q1, q2)

            actor_loss = -tf.reduce_mean(min_q - tf.exp(self.tf_log_alpha) * log_pi)

        grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grad, self.actor.trainable_variables))
        return_dict = {'actor_loss': -actor_loss,
                       'min_q': tf.reduce_mean(min_q),
                       }
        # recoder.record(return_dict, epoch, 'actor')
        return return_dict



    @tf.function
    def train_all(self, qf_obs, next_qf_obs, actor_obs, next_actor_obs, action, mask, reward, gamma):
        next_log_pi, next_new_action = self.resample_action(next_actor_obs)

        # compute min Q_target(s',a')
        next_q_target1 = self.target_qf1(*next_qf_obs, next_new_action)
        next_q_target2 = self.target_qf2(*next_qf_obs, next_new_action)
        min_q_target = tf.minimum(next_q_target1, next_q_target2)

        # compute y(r,s',d)
        y = reward + mask * gamma * (min_q_target - tf.exp(self.tf_log_alpha) * next_log_pi)

        with tf.GradientTape() as q1_tape:
            # compute Q(s,a)
            q1 = self.qf1(*qf_obs, action)
            q1_loss = tf.reduce_mean(tf.square(q1 - y))

        q1_grad = q1_tape.gradient(q1_loss, self.qf1.trainable_variables)
        self.qf1_optimizer.apply_gradients(zip(q1_grad, self.qf1.trainable_variables))

        with tf.GradientTape() as q2_tape:
            # compute Q(s,a)
            q2 = self.qf2(*qf_obs, action)
            q2_loss = tf.reduce_mean(tf.square(q2 - y))

        q2_grad = q2_tape.gradient(q2_loss, self.qf2.trainable_variables)
        self.qf2_optimizer.apply_gradients(zip(q2_grad, self.qf2.trainable_variables))

        with tf.GradientTape() as a_tape:
            # compute min Q(s,a)
            log_pi, new_action = self.resample_action(actor_obs)
            q1 = self.qf1(*qf_obs, new_action)
            q2 = self.qf2(*qf_obs, new_action)
            min_q = tf.minimum(q1, q2)

            actor_loss = tf.reduce_mean(self.tf_log_alpha * log_pi - min_q)

        grad = a_tape.gradient(actor_loss, self.get_trainable_actor)
        self.actor_optimizer.apply_gradients(zip(grad, self.get_trainable_actor))

        with tf.GradientTape() as alpha_tape:
            alpha_loss = -tf.reduce_mean(tf.exp(self.tf_log_alpha) * (tf.stop_gradient(log_pi) + self.target_entropy))

        alpha_grad = alpha_tape.gradient(alpha_loss, [self.tf_log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.tf_log_alpha]))

        return_dict = {
            'q1_loss': q1_loss,
            'q2_loss': q2_loss,
            'alpha_loss': alpha_loss,
            'actor_loss': actor_loss,
        }
        # recoder.record(return_dict, epoch, 'actor')
        return return_dict

    # @property
    def _optimize_(self):
        for _ in range(self.hyper_parameters.update_times):
            data_dict = self.random_sample(self.hyper_parameters.batch_size)

            qf_obs = self.get_corresponding_data(data_dict=data_dict, names=self.qf1.input_name[:-1])
            next_qf_obs = self.get_corresponding_data(data_dict=data_dict, names=self.qf1.input_name[:-1],
                                                      prefix='next_')
            actor_obs = self.get_corresponding_data(data_dict=data_dict, names=self.actor.input_name)
            next_actor_obs = self.get_corresponding_data(data_dict=data_dict, names=self.actor.input_name,
                                                         prefix='next_')

            mask = tf.cast(data_dict['mask'], dtype=tf.float32)
            reward = tf.cast(data_dict['total_reward'], dtype=tf.float32)
            action = tf.cast(data_dict['action'], dtype=tf.float32)

            tf_gamma = tf.cast(self.hyper_parameters.gamma, dtype=tf.float32)

            q_optimize_info = self.train_q_fun(
                qf_obs=qf_obs,
                next_qf_obs=next_qf_obs,
                next_actor_obs=next_actor_obs,
                reward=reward,
                mask=mask,
                gamma=tf_gamma,
                action=action,
                # epoch=self.optimize_epoch,
                # recoder=self.recoder
            )

            policy_optimize_info = self.train_actor(
                q_obs=qf_obs,
                actor_obs=actor_obs,
                # epoch=self.optimize_epoch,
                # recoder=self.recoder
            )

            alpha_optimize_info = self.train_alpha(
                actor_obs=actor_obs,
                # epoch=self.optimize_epoch,
                # recoder=self.recoder
            )

            # soft update
            self.soft_update_weights(self.qf1, self.target_qf1, self.hyper_parameters.tau)
            self.soft_update_weights(self.qf2, self.target_qf2, self.hyper_parameters.tau)
            # print('soft update weights')

        return_dict = {**q_optimize_info, **policy_optimize_info, **alpha_optimize_info}

        return return_dict
