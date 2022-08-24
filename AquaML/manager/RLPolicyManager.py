import numpy as np

from AquaML.policy.GaussianPolicy import GaussianPolicy
from AquaML.policy.CriticPolicy import CriticPolicy

import AquaML as A
import tensorflow as tf


class RLPolicyManager:
    def __init__(self, actor_policy: GaussianPolicy, hierarchical, action_info: dict, actor_input_info: list,
                 work_space: str,
                 critic_model=None, actor_is_batch_timestep: bool = False):
        self.actor_policy = actor_policy
        # self.critic_model = critic_model
        self.actor_input_info = actor_input_info
        if critic_model is None:
            self.model_type = A.SHARE_ACTOR_CRITIC
        else:
            self.model_type = A.SEPARATE_ACTOR_CRITIC
            self.critic = CriticPolicy(critic_model, work_space + '_critic')

        self.hierarchical = hierarchical

        self.action_name = list(action_info)

        self.actor_is_batch_timestep = actor_is_batch_timestep

        # TODO: optimize this, bug.
        if self.actor_policy.type == A.STOCHASTIC:
            self.actor_policy.create_log_std(action_info['action'], hierarchical, work_space)
            self.actor_policy.create_distribution(action_info['action'])

    def actor(self, *args):
        """
        Computing actor model output for optimizing stage.
        :param args: ((obs1, obs2,..., training), (action,)) depends on your model.
        :return: action corresponding to the algo and basic policy.
            If it uses gaussian distribution, return is (action, prob)
        """

        return self.actor_policy(*args)

    def get_actor_trainable_variables(self):
        return self.actor_policy.get_variable

    # def critic(self, *args):
    #     """
    #     Computing critic model output for optimizing stage.
    #
    #     :param args: (obs1,obs2,...,training)
    #     :return: (Tensor) value
    #     """
    #     return self.critic_model(*args)

    def get_critic_trainable_variables(self):
        return self.critic.get_variable

    def get_action(self, obs: dict):
        input_obs = []

        for key in self.actor_input_info:
            if self.actor_is_batch_timestep:
                buffer = np.expand_dims(obs[key], axis=0)
            else:
                buffer = obs[key]
            input_obs.append(tf.cast(buffer, dtype=tf.float32))

        input_obs.append(False)

        action = self.actor_policy.get_action(*input_obs)

        action = dict(zip(self.action_name, action))

        return action

    def sync(self, sync_path):
        if self.hierarchical == A.MAIN_THREAD:
            self.actor_policy.save_weights(sync_path)
            self.critic.save_weights(sync_path)
        else:
            self.actor_policy.load_weights(sync_path)
            self.critic.load_weights(sync_path)

        self.actor_policy.sync()

    def close(self):
        self.actor_policy.close()

    def reset_actor(self, args):
        self.actor_policy.reset(args)

    # def save_model(self):
