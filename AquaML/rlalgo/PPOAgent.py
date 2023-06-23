import tensorflow as tf

from AquaML.rlalgo.BaseRLAgent import BaseRLAgent
from AquaML.rlalgo.AgentParameters import PPOAgentParameter

from AquaML.buffer.RLBuffer import OnPolicyDefaultReplayBuffer

from AquaML.core.Comunicator import Communicator


class PPOAgent(BaseRLAgent):

    def __init__(self,
                 name: str,
                 actor,
                 critic,
                 agent_params: PPOAgentParameter,
                 level: int = 0,  # 控制是否创建不交互的agent
                 ):

        super().__init__(
            name=name,
            agent_params=agent_params,
            level=level,
        )

        self.actor = actor()

        self.critic = critic()

    def init(self):
        self.initialize_actor()
        if self.level == 0:

            # 初始化critic网络
            self.initialize_critic()

            # 创建优化器
            # 检测actor是否包含优化器参数
            if hasattr(self.actor, 'optimizer_info'):
                self.actor_optimizer = self.create_optimizer(self.actor.optimizer_info)
            else:
                raise AttributeError(f'{self.actor.__class__.__name__} has no optimizer_info attribute')

            # 检测critic是否包含优化器参数
            if hasattr(self.critic, 'optimizer_info'):
                self.critic_optimizer = self.create_optimizer(self.critic.optimizer_info)
            else:
                raise AttributeError(f'{self.critic.__class__.__name__} has no optimizer_info attribute')

            self._all_model_dict = {
                'actor': self.actor,
                'critic': self.critic,
            }

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

        # 确定resample_prob函数
        if 'log_std' in self._explore_dict:
            self.resample_prob = self._resample_log_prob_no_std
        else:
            self.resample_prob = self._resample_log_prob_log_std

        # 获取actor优化参数
        self._actor_train_vars = self.actor.trainable_variables
        for key, value in self._tf_explore_dict.items():
            self._actor_train_vars += [value]

        # 创建buffer

        self.actor_buffer = OnPolicyDefaultReplayBuffer(
            capacity=self.agent_info.buffer_size,
            data_names=self.actor.output_info.keys(),
        )

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
                    old_log_prob: tf.Tensor,
                    action: tf.Tensor,
                    clip_ratio: float,
                    entropy_coef: float,
                    ):
        with tf.GradientTape() as tape:
            tape.watch(self.actor_train_vars)

            out = self.resample_prob(actor_inputs, action)

            log_prob = out[0]
            log_std = out[1]
            mu = out[2]

            ratio = tf.exp(log_prob - old_log_prob)

            actor_surrogate_loss = tf.reduce_mean(
                tf.minimum(
                    ratio * advantage,
                    tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage,
                )
            )

            entropy_loss = tf.reduce_mean(self.explore_policy.get_entropy(mu, log_std))

            actor_loss = -actor_surrogate_loss - entropy_coef * entropy_loss

        actor_grads = tape.gradient(actor_loss, self.actor_train_vars)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_train_vars))

        dic = {
            'actor_loss': actor_loss,
            'actor_surrogate_loss': actor_surrogate_loss,
            'entropy_loss': entropy_loss,
        }

        return dic

    @property
    def actor_train_vars(self):
        return self._actor_train_vars

    # TODO:需要规定好接口
    def optimize(self, communicator: Communicator):

        # 检查当前是否为主线程
        if self.level != 0:
            raise RuntimeError('Only main agent can optimize')

        buffer_size = communicator.get_data_pool_size(self.name)

        # 获取所有的数据
        data_dict = communicator.get_data_pool_dict(self.name)

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
                                               gamma=self.agent_params.gamma,
                                               lamda=self.agent_params.lamda,
                                               )

        # convert to tensor
        advantage = tf.convert_to_tensor(advantage, dtype=tf.float32)
        target = tf.convert_to_tensor(target, dtype=tf.float32)
        # rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        # masks = tf.convert_to_tensor(masks, dtype=tf.float32)
        old_prob = tf.convert_to_tensor(old_prob, dtype=tf.float32)
        old_log_prob = tf.math.log(old_prob)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)

        if self.agent_params.batch_advantage_normalization:
            advantage = (advantage - tf.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-8)

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
        buffer_size = train_actor_input['actor_obs'][0].shape[0]
        critic_buffer_size = buffer_size
        critic_batch_steps = self.agent_params.batch_size

        for _ in range(self.agent_params.update_times):
            # fusion ppo firstly update critic
            start_index = 0
            end_index = 0
            critic_start_index = 0
            while end_index < buffer_size:
                end_index = min(start_index + self.agent_params.batch_size,
                                buffer_size)
                critic_end_index = min(critic_start_index + critic_batch_steps, critic_buffer_size)
                batch_train_actor_input = self.get_batch_data(train_actor_input, start_index, end_index)
                batch_train_critic_input = self.get_batch_data(train_critic_input, critic_start_index, critic_end_index)
                start_index = end_index
                critic_start_index = critic_end_index
                for _ in range(self.agent_params.update_critic_times):
                    critic_optimize_info = self.train_critic(
                        critic_inputs=batch_train_critic_input['critic_obs'],
                        target=batch_train_critic_input['target'],
                    )
                    self.loss_tracker.add_data(critic_optimize_info, prefix='critic')

                for _ in range(self.agent_params.update_actor_times):
                    actor_optimize_info = self.train_actor(
                        actor_inputs=batch_train_actor_input['actor_obs'],
                        advantage=batch_train_actor_input['advantage'],
                        old_log_prob=batch_train_actor_input['old_log_prob'],
                        action=batch_train_actor_input['action'],
                        clip_ratio=self.agent_params.clip_ratio,
                        entropy_coef=self.agent_params.entropy_coef,
                    )
                    self.loss_tracker.add_data(actor_optimize_info, prefix='actor')

        summary = self.loss_tracker.get_data()

        return summary

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

        return (log_prob, self.tf_log_std, mu)

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

        return (log_prob, log_std, mu)

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
