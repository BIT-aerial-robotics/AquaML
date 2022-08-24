from AquaML.rlalgo.BaseAlgo import BaseRLAlgo
from AquaML.args.RLArgs import PPOHyperParam, TrainArgs
from AquaML.manager.DataManager import DataManager
from AquaML.manager.RLPolicyManager import RLPolicyManager
import tensorflow as tf
import numpy as np


class ProximalPolicyOptimization(BaseRLAlgo):
    def __init__(self, algo_param: PPOHyperParam, train_args: TrainArgs, data_manager: DataManager,
                 policy: RLPolicyManager):
        """
        https://arxiv.org/abs/1707.06347

        :param algo_param: The param of this algorithm.
        :param train_args: Pre-processing args for data.
        :param data_manager: Manage data.
        :param policy: Manage the policy.
        """
        super().__init__(algo_param=algo_param, train_args=train_args, data_manager=data_manager, policy=policy)

        # self.algo_param = algo_param
        self.data_manager = data_manager
        # self.train_args = train_args
        self.policy = policy

        self.critic_optimizer = tf.optimizers.Adam(learning_rate=self.algo_param.critic_learning_rate)
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=self.algo_param.actor_learning_rate)

    def _optimize(self, data_dict_ac, args: dict):
        # prepare data

        data_dict_ac['actor'].append(True)
        data_dict_ac['critic'].append(True)

        actor_inputs = data_dict_ac['actor']
        critic_inputs = data_dict_ac['critic']
        act = data_dict_ac['action']

        # next_actor_inputs = data_dict_ac['next_actor'].append(True)
        data_dict_ac['next_critic'].append(True)
        next_critic_inputs = data_dict_ac['next_critic']

        # get value
        if self.data_manager.action.get('value') is None:
            if self.train_args.critic_is_batch_timesteps:
                tf_value = self.policy.critic(*critic_inputs)
                tf_next_value = self.policy.critic(*next_critic_inputs)
            else:
                new_critic_inputs = self.data_manager.batch_features(critic_inputs[:-1], True)
                new_critic_inputs.append(True)
                new_next_critic_inputs = self.data_manager.batch_features(next_critic_inputs[:-1], True)
                new_next_critic_inputs.append(True)
                tf_value = self.policy.critic(*new_critic_inputs)
                tf_next_value = self.policy.critic(*new_next_critic_inputs)
        else:
            tf_value = tf.convert_to_tensor(self.data_manager.action['value'].data, dtype=tf.float32)
            tf_next_value = tf.zeros_like(tf_value)
            tf_next_value[:-1] = tf_value[1:]

        gae, target = self.cal_gae_target(self.data_manager.reward['total_reward'].data, tf_value.numpy(),
                                          tf_next_value,
                                          self.data_manager.mask.data)

        if self.train_args.actor_is_batch_timesteps:
            buffer = self.data_manager.batch_timesteps({'gae': gae, 'target': target}, args.get('traj_length'),
                                                       args.get('overlap_size'))
            gae = buffer['gae']
            target = buffer['target']

        tf_gae = tf.convert_to_tensor(gae, dtype=tf.float32)
        tf_target = tf.convert_to_tensor(target, dtype=tf.float32)

        max_step = gae.shape[0]

        # optimize_time = 0

        total_opt_info = []

        for _ in range(self.algo_param.update_times):
            start_pointer = 0
            end_pointer = self.algo_param.batch_size

            while True:
                batch_actor_inputs = self.data_manager.slice_tuple_list(actor_inputs[:-1], start_pointer, end_pointer)
                batch_actor_inputs.append(True)
                batch_critic_inputs = self.data_manager.slice_tuple_list(critic_inputs[:-1], start_pointer, end_pointer)
                batch_critic_inputs.append(True)
                batch_gae = tf_gae[start_pointer:end_pointer]
                batch_target = tf_target[start_pointer: end_pointer]
                batch_act = self.data_manager.slice_tuple_list(act, start_pointer, end_pointer)

                self.policy.reset_actor(batch_target.shape[0])

                if self.train_args.critic_is_batch_timesteps:
                    pass
                else:
                    batch_critic_inputs = self.data_manager.batch_features(batch_critic_inputs[:-1], True)
                    batch_target = self.data_manager.batch_features([batch_target, ], True)
                    batch_target = batch_target[0]
                    batch_critic_inputs.append(True)

                optimization_info = self.train_policy(batch_actor_inputs, batch_critic_inputs, batch_act, batch_gae,
                                                      batch_target)

                total_opt_info.append(tuple(optimization_info.values()))

                # some samples will not be used
                start_pointer = end_pointer

                end_pointer = end_pointer + self.algo_param.batch_size

                if end_pointer > max_step:
                    break

                # print("training")

        total_opt_info = np.mean(total_opt_info, axis=0)

        total_opt_info = dict(zip(tuple(optimization_info), total_opt_info))

        return total_opt_info

    # @tf.function
    def train_policy(self, act_obs: tuple, critic_obs: tuple, act: tuple, gae, target):
        """
        Optimize the policy.

        :param critic_obs:
        :param act_obs:
        :param act:
        :param reward:
        :param next_obs:
        :param gae:
        :param target:
        :return:
        """
        actor_model_inputs = (*act_obs,)  # (obs1,obs2,..,training)
        action = act[0]

        actor_inputs = (actor_model_inputs, (action,))

        with tf.GradientTape() as tape1:
            out = self.policy.actor(*actor_inputs)
            mu, new_prob = out[0], out[1]
            ratio = new_prob / act[1]

            actor_surrogate_loss = tf.reduce_mean(
                tf.minimum(
                    ratio * gae,
                    tf.clip_by_value(ratio, 1 - self.algo_param.clip_ratio,
                                     1 + self.algo_param.clip_ratio
                                     ) * gae
                )
            )

            entropy_loss = -tf.reduce_mean(new_prob * tf.math.log(new_prob))

            loss = -(actor_surrogate_loss - entropy_loss * self.algo_param.entropy_ratio)

        actor_grad = tape1.gradient(loss, self.policy.get_actor_trainable_variables())

        self.actor_optimizer.apply_gradients(zip(actor_grad, self.policy.get_actor_trainable_variables()))

        with tf.GradientTape() as tape2:
            v = self.policy.critic(*critic_obs)
            critic_loss = tf.reduce_mean(tf.square(target - v))

        critic_grad = tape2.gradient(critic_loss, self.policy.get_critic_trainable_variables())
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.policy.get_critic_trainable_variables()))

        dic = {
            'surrogate': actor_surrogate_loss,
            'entropy': entropy_loss,
            'actor_loss': loss,
            'critic_loss': critic_loss
        }

        return dic
