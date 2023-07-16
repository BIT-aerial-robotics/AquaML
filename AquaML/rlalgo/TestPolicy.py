from AquaML.rlalgo.CompletePolicy import CompletePolicy
import numpy as np
import os


class DataSetTracker:
    """
    用于追踪多线程或者需要分段处理数据，最后将数据进行汇总为标准训练集
    """

    def __init__(self):
        self.data_dict = {}

    def add_data(self, data_dict: dict, prefix: str = ''):

        if len(prefix) > 0:
            prefix = prefix + '/'

        for key, value in data_dict.items():

            all_name = prefix + key

            if all_name not in self.data_dict:
                self.data_dict[all_name] = []
            self.data_dict[all_name].append(value)

    def save_data(self, path):
        for key, value in self.data_dict.items():
            if 'reward' not in key:
                data = np.vstack(value)
                file = os.path.join(path, key + '.npy')
                np.save(file, data)
            else:
                print("{}:{}".format(key,np.sum(value)))


class TestPolicy:
    def __init__(self,
                 env,
                 policy: CompletePolicy,
                 ):
        self.env = env
        self.policy = policy

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

            for _ in range(episode_steps):
                action = self.policy.get_action(obs)
                obs, reward, done, _ = self.env.step(action)
                data_tracker.add_data(obs)
                data_tracker.add_data(action)
                data_tracker.add_data(reward, 'reward')
                if done:
                    break

        data_tracker.save_data(data_path)
