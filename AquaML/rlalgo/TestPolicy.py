from AquaML.rlalgo.CompletePolicy import CompletePolicy
import numpy as np
import os
from copy import deepcopy


# class TrajectoryTracker:
#     def __init__(self):
#         self.data_dict = {}

#     def add_data(self, data_dict: dict, prefix: str = ''):

#         if len(prefix) > 0:
#             prefix = prefix + '_'

#         for key, value in data_dict.items():

#             all_name = prefix + key

#             if all_name not in self.data_dict:
#                 self.data_dict[all_name] = []
#             self.data_dict[all_name].append(value)

#     def save_data(self, path, episode_steps):
#         for key, value in self.data_dict.items():
#             if 'reward' not in key:
#                 data = np.vstack(value)
#                 file = os.path.join(path, key + '.npy')
#                 np.save(file, data)
#             else:
#                 value = np.reshape(value, (-1, episode_steps))
#                 sum_value = np.sum(value, axis=1)
#                 max_value = np.max(sum_value)
#                 min_value = np.min(sum_value)
#                 avg_value = np.mean(sum_value)
#                 print("{}:{}".format(key, avg_value))
#                 print("{}:{}".format(key, max_value))
#                 print("{}:{}".format(key, min_value))


class DataSetTracker:
    """
    用于追踪多线程或者需要分段处理数据，最后将数据进行汇总为标准训练集
    """

    def __init__(self):
        self.data_dict = {}

    def add_data(self, data_dict: dict, prefix: str = ''):

        if len(prefix) > 0:
            prefix = prefix + '_'

        for key, value in data_dict.items():

            all_name = prefix + key

            if all_name not in self.data_dict:
                self.data_dict[all_name] = []
            self.data_dict[all_name].append(value)

    def save_data(self, path, episode_steps):
        for key, value in self.data_dict.items():
            if 'reward' in key:
                key = key[7:]
            data = np.vstack(value)
            file = os.path.join(path, key + '.npy')
            np.save(file, data)
            if 'reward' in key:
                value = np.reshape(value, (-1, episode_steps))
                sum_value = np.sum(value, axis=1)
                max_value = np.max(sum_value)
                min_value = np.min(sum_value)
                avg_value = np.mean(sum_value)
                print("{}:{}".format(key, avg_value))
                print("{}:{}".format(key, max_value))
                print("{}:{}".format(key, min_value))

    def get_data(self):

        ret_dict = {}

        for key, value in self.data_dict.items():
            data = np.vstack(value)
            ret_dict[key] = data

        return ret_dict


class TestPolicy:
    def __init__(self,
                 env,
                 policy: CompletePolicy,
                 ):
        self.env = env
        self.policy = policy

        self.env.set_action_state_info(self.policy.actor.output_info, self.policy.actor.input_name)

        self.policy.initialize_actor(self.env.obs_info)

    def evaluate(self,
                 episode_steps,
                 episode_num,
                 data_path,
                 ):
        """
        Evaluate the policy.
        """
        data_tracker = DataSetTracker()

        for _ in range(episode_num):
            obs, _ = self.env.reset()
            done = False

            for i in range(episode_steps):
                action = self.policy.get_action(obs)
                obs_, reward, done, _ = self.env.step(action)
                data_tracker.add_data(deepcopy(obs))
                data_tracker.add_data(deepcopy(obs_), 'next')
                data_tracker.add_data(deepcopy(action))
                data_tracker.add_data(reward, 'reward')

                mask = 1 - done

                if i == episode_steps - 1:
                    mask = 0

                mask_dict = {
                    'mask': mask,
                }
                data_tracker.add_data(mask_dict)

                obs = obs_
                # if done:
                #     break

        data_tracker.save_data(data_path, episode_steps)

    def collect(self,
                episode_steps,
                episode_num,
                data_path,
                ):
        """
        Collect the data.
        """
        data_tracker = DataSetTracker()

        for _ in range(episode_num):
            obs, _ = self.env.reset()
            done = False
            traj_tracker = DataSetTracker()
            for _ in range(episode_steps):
                action = self.policy.get_action(obs)
                obs_, reward, done, _ = self.env.step(action)

                traj_tracker.add_data(deepcopy(obs))
                traj_tracker.add_data(deepcopy(obs_), 'next')
                traj_tracker.add_data(deepcopy(action))
                traj_tracker.add_data(reward, 'reward')

                # judge_dict = {
                #     'judge': self.env.dis_judge,
                # }
                # traj_tracker.add_data(judge_dict)

                obs = obs_

                if done:
                    break

            traj_data = traj_tracker.get_data()
            data_tracker.add_data(traj_data)

            # judge = traj_data['judge'][-1]
            #
            # if judge > 200:
            #     data_tracker.add_data(traj_data)

        data_tracker.save_data(data_path, episode_steps)
