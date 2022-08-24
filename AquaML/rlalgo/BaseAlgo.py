import abc
import numpy as np
from AquaML.manager.DataManager import DataManager


class BaseRLAlgo(abc.ABC):
    def __init__(self, algo_param, train_args, data_manager: DataManager,
                 policy):
        """
        Provide basic manipulate.

        :param algo_param: The param of this algorithm.
        :param train_args: Pre-processing args for data.
        :param data_manager: Manage data.
        :param policy: Manage the policy.
        """
        self.data_manager = data_manager

        self.work_space = self.data_manager.work_space

        self.algo_param = algo_param

        self.train_args = train_args

        self.epoch = 0

        # create tensorboard recorder
        # TODO: add recoder module
        # log_dir = self.work_space + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
        # self.min_summary_writer = tf.summary.create_file_writer(log_dir + "/min")
        # self.max_summary_writer = tf.summary.create_file_writer(log_dir + "/max")
        # self.average_summary_writer = tf.summary.create_file_writer(log_dir + "/average")
        # self.main_summary_writer = tf.summary.create_file_writer(log_dir + "/main")
        # self.before_summary_writer = tf.summary.create_file_writer(log_dir + "/training_before")
        # self.after_summary_writer = tf.summary.create_file_writer(log_dir + "/training_after")

        self.epoch = 0

    def cal_discount_reward(self, rewards, mask):
        discount_rewards = []
        mask_ = mask[::-1]
        value_ = 0
        for i, reward in enumerate(rewards[::-1]):
            value_ = reward + self.hyper_parameters.gamma * value_ * mask_[i]
            discount_rewards.append(value_)

        discount_rewards.reverse()
        discount_rewards = np.hstack(discount_rewards)

        return discount_rewards

    def cal_gae_target(self, rewards, values, next_values, mask):
        gae = np.zeros_like(rewards)
        n_steps_target = np.zeros_like(rewards)
        cumulate_gae = 0
        # next_val = 0

        for i in reversed(range(0, len(rewards))):
            delta = rewards[i] + self.algo_param.gamma * next_values[i] - values[i]
            cumulate_gae = self.algo_param.gamma * self.algo_param.lambada * cumulate_gae * mask[i] + delta
            gae[i] = cumulate_gae
            # next_val = values[i]
            n_steps_target[i] = gae[i] + values[i]

        return gae, n_steps_target

    def cal_episode_info(self, verbose=False):
        """
        Summarize reward information.

        :param verbose: Display reward information.
        :return: Dict contains every part of reward info.
        """
        index_done = np.where(self.data_manager.mask.data == 0)[0] + 1  # we need to use numpy slice

        start_index = 0

        # reward category
        reward_cat = list(self.data_manager.mapping_dict['reward'])

        reward_dic = dict()

        for key in reward_cat:
            reward_dic[key] = []

        for end_index in index_done:
            episode_whole_reward = self.data_manager.slice_data(['reward'], start_index, end_index)
            episode_whole_reward = episode_whole_reward['reward']

            for key, val in episode_whole_reward.items():
                reward_dic[key].append(np.sum(val))

            start_index = end_index

        reward_summary = dict()

        reward_summary['std'] = np.std(reward_dic['total_reward'])
        reward_summary['max_reward'] = np.max(reward_dic['total_reward'])
        reward_summary['min_reward'] = np.min(reward_dic['total_reward'])

        for key, val in reward_dic.items():
            reward_summary[key] = np.mean(val)

        if verbose:
            for key, val in reward_summary.items():
                print(key + ": {}".format(val))

        del reward_dic
        del episode_whole_reward

        return reward_summary

    def optimize(self):
        print("---------------epoch:{}---------------".format(self.epoch + 1))
        self.cal_episode_info(True)
        args = {
            'tf_data': True,
            'traj_length': self.train_args.traj_length,
            'overlap_size': self.train_args.overlap_size
        }

        data_dict_ac = self.data_manager.get_input_data(args_dict=args)

        opt_info = self._optimize(data_dict_ac)
        for key, val in opt_info.items():
            print(key + ": {}".format(val))
        self.epoch += 1

    @abc.abstractmethod
    def _optimize(self, data_dict_ac):
        """
        Policy optimization method.
        :return:
        """
