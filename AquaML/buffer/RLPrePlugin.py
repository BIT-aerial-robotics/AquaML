"""

这里是强化学习的预处理插件，预处理首先会根据episode的长度进行截断，
然后将episode的数据进行整理，分成episode时候可以写入一些插件，比如对
episode的数据进行统计，统计reward，过滤部分episode等等。
"""
from AquaML.core.RLToolKit import RLStandardDataSet
from AquaML.core.ToolKit import DataSetTracker, LossTracker
from AquaML.core.DataParser import DataSet
import numpy as np
from abc import ABC, abstractmethod
import tensorflow as tf


class PluginBase(ABC):

    def __call__(self, data: dict, mask_end):
        return self._process(data, mask_end)

    @abstractmethod
    def _process(self, data: dict, mask_end):
        pass


class ValueFunctionComputer(PluginBase):

    def __init__(self,
                 value_model,
                 ):

        self.value_model = value_model

        if 'log_std' in self.value_model.output_info.keys():
            self.vf_idx = 2
        else:
            self.vf_idx = 1

    def _process(self, data: dict, mask_end):

        """
            data is an episode trajectory
            """
        input_name = self.value_model.input_name

        if 'value' not in data:
            input_data = []
            for name in input_name:
                input_data.append(data[name])

            output = self.value_model(*input_data)
            output_names = self.value_model.output_info.keys()
            if isinstance(output, tf.Tensor):
                value = output.numpy()
            else:
                dic = dict(zip(output_names, output))
                value = dic['value'].numpy()

        else:
            value = data['value']

        next_input = []
        for name in input_name:
            next_input.append(data['next_' + name][-1].reshape((1, -1)))
            # print(data['next_' + name].shape)

        output2 = self.value_model(*next_input)

        if isinstance(output2, tf.Tensor):
            end_value = output2.numpy()
        elif isinstance(output2, list) or isinstance(output2, tuple):
            end_value = output2[self.vf_idx].numpy()
        else:
            raise ValueError('model is not a tensor or a list or a tuple')

        end_value = np.squeeze(end_value)

        # next value, the last value is 0
        # if mask_end == 0:
        #     end_value = 0.0
        # else:
        #     next_input = []
        #     for name in input_name:
        #         next_input.append(data['next_' + name][-1].reshape((1, -1)))
        #
        #     output2 = self.value_model(*next_input)
        #
        #     if isinstance(output2, tf.Tensor):
        #         end_value = output2.numpy()
        #     elif isinstance(output2, list) or isinstance(output2, tuple):
        #         end_value = output2[self.vf_idx].numpy()
        #     else:
        #         raise ValueError('model is not a tensor or a list or a tuple')

        # end_value = np.squeeze(end_value)

        # end_value = np.squeeze(self.value_model(*next_input).numpy())

        next_value = np.append(value[1:], end_value)

        processed_data = {'value': value.reshape((-1, 1)),
                          'next_value': next_value.reshape((-1, 1))}

        return processed_data


class GAEComputer(PluginBase):
    def __init__(self,
                 gamma,
                 lamda,
                 ):
        super().__init__()
        self.gamma = gamma
        self.lamda = lamda

    def _process(self, data: dict, mask_end):
        values = data['value']
        next_values = data['next_value']
        rewards = data['total_reward']

        last_value = next_values[-1]

        mask = data['mask']

        gae = np.zeros_like(rewards)
        n_steps_target = np.zeros_like(rewards)
        cumulated_advantage = 0.0
        length = len(rewards)
        index = length

        for i in range(length):
            index -= 1
            if index == length - 1:
                next_value = last_value
                done = mask_end
                reward = rewards[index] + self.gamma * next_value
            else:
                next_value = values[index + 1]
                done = mask[index]
                reward = rewards[index]

            delta = reward + self.gamma * next_value * done - values[index]
            cumulated_advantage = self.gamma * self.lamda * cumulated_advantage * done + delta
            gae[index] = cumulated_advantage
            n_steps_target[index] = gae[index] + values[index]

        processed_data = {'advantage': gae,
                          'target': n_steps_target}
        return processed_data


class SplitTrajectory:

    def __init__(self,
                 filter_name=None,
                 filter_args: dict = {}
                 ):
        from AquaML import traj_filter_register

        self.filter = traj_filter_register.get_filter(filter_name)
        self.filter_args = filter_args

        self._plugin_list = []

    def __call__(self, trajectory: RLStandardDataSet, shuffle=False):
        """
        The dataset store in the trajectory is a dict, the key is the name of the data, the value is the data.

        The shape of the data is (num_envs, rollout_steps, ...)

        Args:
            trajectory: Collection of trajectories
            shuffle: shuffle the data according to the first dimension.
        """
        data_set_tracker = DataSetTracker()
        reward_tracker = LossTracker()
        rollout_steps = trajectory.rollout_steps
        num_envs = trajectory.num_envs
        for env_traj in trajectory.get_env_data(shuffle=shuffle):
            masks = env_traj['mask']

            terminal = np.where(masks == 0)[0] + 1
            last_mask = masks[-1]

            if last_mask == 1:
                terminal = np.append(terminal, [masks.shape[0]])

            start = np.append([0], terminal[:-1])

            terminal = terminal.tolist()
            start = start.tolist()

            for start_idx, terminal_idx in zip(start, terminal):
                episode_traj = {}
                end_mask = masks[terminal_idx - 1]
                for key, value in env_traj.items():
                    episode_traj[key] = value[start_idx:terminal_idx]

                episode_length = terminal_idx - start_idx

                if end_mask == 1:
                    store_flag = True
                else:
                    store_flag = self.filter(episode_traj, episode_length, **self.filter_args)

                if store_flag:

                    # 插件处理部分
                    for plugin in self._plugin_list:
                        processed_data = plugin(episode_traj, end_mask)

                        episode_traj.update(processed_data)

                    reward = {
                        # 'total_reward': np.sum(episode_traj['total_reward']),
                        'length': episode_length,
                    }

                    for name in trajectory.reward_names:
                        reward[name] = np.sum(episode_traj[name])
                    # if 'indicate' in episode_traj:
                    #     reward['indicate'] = np.sum(episode_traj['indicate'])
                    reward_tracker.add_data(reward, 'episode')

                    data_set_tracker.add_data(episode_traj)

        data = data_set_tracker.gett_data()

        return DataSet(data, rollout_steps, num_envs), reward_tracker.get_data()

    def add_plugin(self, plugin):
        if isinstance(plugin, PluginBase):
            self._plugin_list.append(plugin)


class RLPrePluginRegister:
    def __init__(self):
        self.plugins = []

    def add_plugin(self, plugin):
        self.plugins.append(plugin)
