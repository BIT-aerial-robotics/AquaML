from AquaML.rlalgo.BaseAlgo import BaseRLAlgo
from AquaML.args.RLArgs import PPGHyperParam, TrainArgs
from AquaML.manager.DataManager import DataManager
from AquaML.manager.RLPolicyManager import RLPolicyManager
import tensorflow as tf
import numpy as np
import copy


class PhasicPolicyGradient(BaseRLAlgo):
    def __init__(self, algo_param: PPGHyperParam, train_args: TrainArgs, data_manager: DataManager,
                 policy: RLPolicyManager, recoder):
        """
        Phasic Policy Gradient https://arxiv.org/abs/2009.04416

        The framework is the same as PPO.

        :param algo_param: The param of this algorithm.
        :param train_args: Pre-processing args for data.
        :param data_manager: Manage data.
        :param policy: Manage the policy.
        """

        super().__init__(algo_param=algo_param, train_args=train_args, data_manager=data_manager, policy=policy,
                         recoder=recoder)
        self.name = 'PPG'

        self.data_manager = data_manager
        self.policy = policy

        self.critic_optimizer = tf.optimizers.Adam(learning_rate=self.algo_param.critic_learning_rate)
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=self.algo_param.actor_learning_rate)

        # store data
        self.buffer = {'actor': [], 'critic': [], 'action': [], 'target': []}

    @tf.function
    def train_phase1(self, act_obs: tuple, critic_obs: tuple, act: tuple, gae, target, target_):
        """
        Optimize the phase1 policy.

        :param target_:
        :param critic_obs:
        :param act_obs:
        :param act:
        :param gae:
        :param target:
        :return:
        """

        actor_model_inputs = (*act_obs,)  # (obs1,obs2,..,training)
        action = act[0]

        actor_inputs = (actor_model_inputs, (action,))
        for _ in range(self.algo_param.update_actor_times):
            with tf.GradientTape() as tape1:
                out = self.policy.actor(*actor_inputs)
                mu, new_prob, joint_value = out[0], out[1], out[2]
                ratio = new_prob / act[1]

                actor_surrogate_loss = tf.reduce_mean(
                    tf.minimum(
                        ratio * gae,
                        tf.clip_by_value(ratio, 1 - self.algo_param.clip_ratio,
                                         1 + self.algo_param.clip_ratio
                                         ) * gae
                    )
                )

                joint_value_loss = tf.reduce_mean(tf.square(joint_value - target))

                entropy_loss = -tf.reduce_mean(new_prob * tf.math.log(new_prob))

                loss = -(
                        actor_surrogate_loss + entropy_loss * self.algo_param.c1 - self.algo_param.c2 * joint_value_loss)

            actor_grad = tape1.gradient(loss, self.policy.get_actor_trainable_variables())

            self.actor_optimizer.apply_gradients(zip(actor_grad, self.policy.get_actor_trainable_variables()))

        for _ in range(self.algo_param.update_critic_times):
            with tf.GradientTape() as tape2:
                v = self.policy.critic(*critic_obs)
                critic_loss = tf.reduce_mean(tf.square(target_ - v))

            critic_grad = tape2.gradient(critic_loss, self.policy.get_critic_trainable_variables())
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.policy.get_critic_trainable_variables()))
        dic = {
            'surrogate': actor_surrogate_loss,
            'entropy': entropy_loss,
            'actor_loss': loss,
            'critic_loss': critic_loss,
            'joint_value_loss': joint_value_loss
        }

        return dic

    @tf.function
    def train_phase2(self, act_obs: tuple, critic_obs: tuple, act: tuple, target, target_):
        actor_model_inputs = (*act_obs,)  # (obs1,obs2,..,training)
        action = act[0]
        actor_inputs = (actor_model_inputs, (action,))

        with tf.GradientTape() as tape2:
            v = self.policy.critic(*critic_obs)
            value_loss = tf.reduce_mean(tf.square(v - target_))
        value_grad = tape2.gradient(value_loss, self.policy.get_critic_trainable_variables())
        self.actor_optimizer.apply_gradients(zip(value_grad, self.policy.get_critic_trainable_variables()))

        # optimize policy parts
        with tf.GradientTape() as tape1:
            joint_policy_out = self.policy.get_actor_value(*actor_inputs)
            mu, new_prob, value = joint_policy_out[0], joint_policy_out[1], joint_policy_out[2]
            aux_loss = 0.5 * tf.reduce_mean(tf.square(value - target))
            kl_div = tf.reduce_mean(tf.keras.losses.KLD(act[1], new_prob))
            # kl_div = tf.reduce_sum(act[1] * (tf.math.log(act[1]) - tf.math.log(new_prob)))
            joint_loss = aux_loss + self.algo_param.beta_clone * kl_div

        joint_grad = tape1.gradient(joint_loss, self.policy.get_actor_critic_variable)
        self.actor_optimizer.apply_gradients(zip(joint_grad, self.policy.get_actor_critic_variable))

        # target_ = tf.concat(target, axis=0)


        dic = {
            'aux_loss': aux_loss,
            'kl_div': kl_div,
            'joint_loss': joint_loss,
            'value_loss': value_loss
        }

        return dic

    def _optimize(self, data_dict_ac, args: dict):
        # prepare data
        # start = time.time()
        self.buffer['actor'].append(copy.deepcopy(data_dict_ac['actor']))
        self.buffer['critic'].append(copy.deepcopy(data_dict_ac['critic']))
        self.buffer['action'].append(copy.deepcopy(data_dict_ac['action']))

        data_dict_ac['actor'].append(True)
        data_dict_ac['critic'].append(True)

        actor_inputs = data_dict_ac['actor']
        critic_inputs = data_dict_ac['critic']
        act = data_dict_ac['action']

        # next_actor_inputs = data_dict_ac['next_actor'].append(True)
        data_dict_ac['next_critic'].append(True)
        next_critic_inputs = data_dict_ac['next_critic']

        # end = time.time()

        # print('get data time:{}'.format(end - start))

        # start = time.time()
        # get value
        if self.data_manager.action.get('value') is None:
            if self.train_args.actor_is_batch_timesteps:
                new_critic_inputs = self.data_manager.batch_features(critic_inputs[:-1], True)
                new_critic_inputs.append(True)
                new_next_critic_inputs = self.data_manager.batch_features(next_critic_inputs[:-1], True)
                new_next_critic_inputs.append(True)
                tf_value = self.policy.critic(*new_critic_inputs)
                tf_next_value = self.policy.critic(*new_next_critic_inputs)
            else:
                tf_value = self.policy.critic(*critic_inputs)
                tf_next_value = self.policy.critic(*next_critic_inputs)
        else:
            tf_value = tf.convert_to_tensor(self.data_manager.action['value'].data, dtype=tf.float32)
            tf_next_value = tf.zeros_like(tf_value)
            tf_next_value[:-1] = tf_value[1:]

        # end = time.time()

        # print('get value time:{}'.format(end-start))

        # start = time.time()

        gae, target = self.cal_gae_target((self.data_manager.reward['total_reward'].data + 8) / 8, tf_value.numpy(),
                                          tf_next_value.numpy(),
                                          self.data_manager.mask.data)

        # end = time.time()

        # print('get gae time:{}'.format(end - start))

        if self.train_args.actor_is_batch_timesteps:
            buffer = self.data_manager.batch_timesteps({'gae': gae, 'target': target}, args.get('traj_length'),
                                                       args.get('overlap_size'))
            gae = buffer['gae']
            target = buffer['target']

        self.buffer['target'].append(copy.deepcopy([target, ]))

        tf_gae = tf.convert_to_tensor(gae, dtype=tf.float32)
        tf_target = tf.convert_to_tensor(target, dtype=tf.float32)

        max_step = gae.shape[0]

        # optimize_time = 0

        total_opt_info = []

        for _ in range(self.algo_param.update_times):
            start_pointer = 0
            end_pointer = self.algo_param.batch_size

            while True:
                # start = time.time()

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
                    batch_target_ = self.data_manager.batch_features([batch_target, ], True)
                    batch_target_ = batch_target_[0]
                    batch_critic_inputs.append(True)

                # end = time.time()

                # print('prepare time for training:{}'.format(end - start))

                # start = time.time()

                optimization_info = self.train_phase1(batch_actor_inputs, batch_critic_inputs, batch_act, batch_gae,
                                                      batch_target, batch_target_)

                # end = time.time()

                # print('training time:{}'.format(end - start))

                total_opt_info.append(tuple(optimization_info.values()))

                # some samples will not be used
                start_pointer = end_pointer

                end_pointer = end_pointer + self.algo_param.batch_size

                if end_pointer > max_step:
                    break

                # print("training")

        total_opt_info = np.mean(total_opt_info, axis=0)

        total_opt_info = dict(zip(tuple(optimization_info), total_opt_info))

        if (self.epoch + 1) % self.algo_param.n_pi == 0:
            total_opt_info2 = []
            buffer = dict()

            for key in list(self.buffer):
                buffer[key] = []
            for key, values in self.buffer.items():
                for value in values:
                    if len(value) > 1:
                        value = tf.concat(values, axis=1)
                        for data in value:
                            buffer[key].append(data)
                        break
                    else:
                        value = tf.concat(values, axis=1)
                        buffer[key].append(tf.squeeze(value, axis=0))
                        break

            buffer['actor'].append(True)
            buffer['critic'].append(True)
            critic_inputs = buffer['critic']
            actor_inputs = tuple(buffer['actor'])
            act = tuple(buffer['action'])
            target = buffer['target']

            # target_ = self.data_manager.batch_features(target, convert_tensor=True)[0]

            target = target[0]

            for _ in range(self.algo_param.update_aux_times):
                start_pointer = 0
                end_pointer = self.algo_param.batch_size
                while True:
                    batch_actor_inputs = self.data_manager.slice_tuple_list(actor_inputs[:-1], start_pointer,
                                                                            end_pointer)
                    batch_actor_inputs.append(True)
                    batch_critic_inputs = self.data_manager.slice_tuple_list(critic_inputs[:-1], start_pointer,
                                                                             end_pointer)
                    batch_critic_inputs.append(True)

                    if self.train_args.actor_is_batch_timesteps:

                        if self.train_args.critic_is_batch_timesteps:
                            pass
                        else:
                            batch_critic_inputs = self.data_manager.batch_features(batch_critic_inputs[:-1], True)
                            batch_critic_inputs.append(True)
                    batch_target = target[start_pointer:end_pointer]
                    batch_target_ = self.data_manager.batch_features((batch_target,), True)[0]
                    batch_act = self.data_manager.slice_tuple_list(act, start_pointer, end_pointer)
                    self.policy.reset_actor(batch_target.shape[0])
                    optimization_info = self.train_phase2(batch_actor_inputs, batch_critic_inputs, batch_act,
                                                          batch_target, batch_target_)
                    total_opt_info2.append(tuple(optimization_info.values()))

                    start_pointer = end_pointer

                    end_pointer = end_pointer + self.algo_param.PPG_batch_size

                    if end_pointer > max_step:
                        break

            total_opt_info2 = np.mean(total_opt_info2, axis=0)
            total_opt_info2 = dict(zip(list(optimization_info), total_opt_info2))

            name1 = list(total_opt_info)
            name2 = list(total_opt_info2)

            value1 = list(total_opt_info.values())
            value2 = list(total_opt_info2.values())

            name = name1 + name2
            value = value1 + value2

            total_opt_info = dict(zip(name, value))

            self.buffer = {'actor': [], 'critic': [], 'action': [], 'target': []}

        return total_opt_info
