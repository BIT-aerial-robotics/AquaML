from AquaML.rlalgo.BaseRLAlgo import BaseRLAlgo
from AquaML.rlalgo.Parameters import FusionPPO_parameter
from AquaML.DataType import RLIOInfo
import tensorflow as tf


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
            if self.resample_log_prob is None:
                raise ValueError(
                    'resample_log_prob must be set when using PPO. You probably make an actor with output ''log_prob''.')
            self.actor = actor()
            self.critic = critic()

            # initialize actor and critic
            self.initialize_model_weights(self.actor)
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
            if not fusion_flag:
                raise ValueError('Fusion value must be in actor output. '
                                 'Please check your actor output.')
            self.fusion_value_idx += 1

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
            v = self.critic(*critic_obs)
            critic_loss = tf.reduce_mean(tf.square(v - target))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        dic = {
            'critic_loss': critic_loss,
            'critic_grad': critic_grad,
        }
        return dic

    def train_actor(self,
                    actor_obs: tuple,
                    advantage: tf.Tensor,
                    old_log_prob: tf.Tensor,
                    action: tf.Tensor,
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

            fusion_value_loss = tf.reduce_mean(tf.square(fusion_value - advantage))

            entropy_loss = -tf.reduce_mean(log_prob)

            loss = -actor_surrogate_loss + lam * fusion_value_loss - entropy_coefficient * entropy_loss

        actor_grad = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        dic = {
            'actor_surrogate_loss': actor_surrogate_loss,
            'fusion_value_loss': fusion_value_loss,
            'entropy_loss': entropy_loss,
        }
        return dic

    def _optimize_(self):
        data_dict = self.get_all_data