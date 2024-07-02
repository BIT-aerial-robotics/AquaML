import random

from AquaML.buffer.MixtureBufferBase import MixtureBufferBase
from AquaML.communicator.CommunicatorBase import CommunicatorBase
from AquaML.core.old.DataModule import DataModule
from AquaML.core.old.FileSystem import DefaultFileSystem
import numpy as np
import copy
from threading import Thread, ThreadError


class MixtureBuffer(MixtureBufferBase):
    def __init__(self,
                 capacity: int,
                 data_names: list,
                 data_module: DataModule,
                 communicator: CommunicatorBase,
                 static_data_path: str = None,
                 shared_list=False
                 ):
        super(MixtureBuffer, self).__init__(
            capacity=capacity,
            data_names=data_names,
            data_module=data_module,
            communicator=communicator,
        )

        self._static_data_path = static_data_path
        self._static_data_num = 0

        if self._static_data_path is not None:
            self.load_npy(data_names, static_data_path)
            self._static_data_num = len(self.static_data[data_names[0]])
            self._communicator.logger_success('MixtureBuffer: load {} static data from {}'.format(
                self._static_data_num, self._static_data_path))
            self._communicator.logger_info('MixtureBuffer: static data now is {}'.format(
                self.static_data))

    def append_real(self):
        """
        从paird data中获取数据，用于数据的收集。
        """

        # 判断paired data flag是否处于可写入状态

        self._data_module.wait_program_start()

        while not self._data_module.get_program_end:

            # 等待当前交互数据准备就绪
            self._data_module.wait_for_flag_to_true(
                flag=self._data_module.send_flag,
                time_out=self._communicator.wait_time_out,
                check_interval=self._communicator.check_time_interval
            )
            self._communicator.logger_info('flag:{} is {}'.format(self._data_module.send_flag.name,
                                                                  self._data_module.send_flag.get_data()))

            if self.capacity_count == 0:
                # 获得s,a,s',r
                # 将state添加到buffer中
                for state_name, state_unit in self._data_module.robot_state_dict.items():
                    # self._communicator.logger_info('RealCollectBufferBase aquire state:{}'.format(state_unit.get_data()))
                    state = copy.deepcopy(state_unit.get_data())
                    self.dynamic_data[state_name] = state

                for next_state_name, next_state_unit in self._data_module.robot_next_state_dict.items():
                    # self._communicator.debug_info('RealCollectBufferBase', 'Aquire next state:{}'.format(next_state_unit.get_data()))
                    next_state = copy.deepcopy(next_state_unit.get_data())
                    self.dynamic_data[next_state_name] = next_state

                action = copy.deepcopy(self._data_module.robot_control_action.get_data())
                self.dynamic_data['action'] = action

                mask = copy.deepcopy(self._data_module.env_mask.get_data())
                self.dynamic_data['mask'] = mask

                for reward_name, reward_unit in self._data_module.rewards_dict.items():
                    # self._communicator.debug_info('RealCollectBufferBase', 'Aquire reward:{}'.format(reward_unit.get_data()))
                    reward = copy.deepcopy(reward_unit.get_data())
                    self.dynamic_data[reward_name] = reward

                self.capacity_count += 1
                self._communicator.logger_info('0 buffer now is {}'.format(self.dynamic_data))
            elif self.capacity_count < self._capacity:
                # 获得s,a,s',r
                # 将state添加到buffer中
                for state_name, state_unit in self._data_module.robot_state_dict.items():
                    # self._communicator.logger_info('RealCollectBufferBase aquire state:{}'.format(state_unit.get_data()))
                    state = copy.deepcopy(state_unit.get_data())
                    self.dynamic_data[state_name] = np.append(self.dynamic_data[state_name], state, 0)

                for next_state_name, next_state_unit in self._data_module.robot_next_state_dict.items():
                    # self._communicator.debug_info('RealCollectBufferBase', 'Aquire next state:{}'.format(next_state_unit.get_data()))
                    next_state = copy.deepcopy(next_state_unit.get_data())
                    self.dynamic_data[next_state_name] = np.append(self.dynamic_data[next_state_name], next_state, 0)

                action = copy.deepcopy(self._data_module.robot_control_action.get_data())
                self.dynamic_data['action'] = np.append(self.dynamic_data['action'], action, 0)

                mask = copy.deepcopy(self._data_module.env_mask.get_data())
                self.dynamic_data['mask'] = np.append(self.dynamic_data['mask'], mask, 0)

                for reward_name, reward_unit in self._data_module.rewards_dict.items():
                    # self._communicator.debug_info('RealCollectBufferBase', 'Aquire reward:{}'.format(reward_unit.get_data()))
                    reward = copy.deepcopy(reward_unit.get_data())
                    self.dynamic_data[reward_name] = np.append(self.dynamic_data[reward_name], reward, 0)

                self.capacity_count += 1
                self._communicator.logger_info('1 buffer now is {}'.format(self.dynamic_data))
            else:
                Index = self.capacity_count % self._capacity
                # 获得s,a,s',r
                # 将state添加到buffer中
                for state_name, state_unit in self._data_module.robot_state_dict.items():
                    # self._communicator.debug_info('RealCollectBufferBase', 'Aquire state:{}'.format(state_unit.get_data()))
                    state = copy.deepcopy(state_unit.get_data())
                    self.dynamic_data[state_name][Index] = state

                for next_state_name, next_state_unit in self._data_module.robot_next_state_dict.items():
                    # self._communicator.debug_info('RealCollectBufferBase', 'Aquire next state:{}'.format(next_state_unit.get_data()))
                    self.dynamic_data[next_state_name][Index] = copy.deepcopy(next_state_unit.get_data())

                self.dynamic_data['action'][Index] = copy.deepcopy(self._data_module.robot_control_action.get_data())

                mask = copy.deepcopy(self._data_module.env_mask.get_data())
                self.dynamic_data['mask'][Index] = mask

                for reward_name, reward_unit in self._data_module.rewards_dict.items():
                    # self._communicator.debug_info('RealCollectBufferBase', 'Aquire reward:{}'.format(reward_unit.get_data()))
                    self.dynamic_data[reward_name][Index] = copy.deepcopy(reward_unit.get_data())

                self.capacity_count += 1
                self._communicator.logger_info('2 buffer now is {}'.format(self.dynamic_data))

            # 重置send_flag
            self._data_module.send_flag[0] = [False]
            self._communicator.logger_info('flag:{} is {}'.format(self._data_module.send_flag.name,
                                                                  self._data_module.send_flag.get_data()))

    def run(self):
        self.append_real()

    def sample_data(self, batch_size:int) ->dict:
        """
                采样数据。

                Args:
                    batch_size (int): 采样数据的大小。
                """
        # 得到在线数据数量和数据总量
        real_dynamic_data_num = self._capacity if self.capacity_count > self._capacity else self.capacity_count
        real_data_num = real_dynamic_data_num + self._static_data_num

        # 将离线数据集和在线数据集打乱
        static_indices = np.random.permutation(self._static_data_num)
        dynamic_indices = np.random.permutation(real_dynamic_data_num)

        if batch_size > real_data_num:
            batch_size = real_data_num

        static_sample_num = random.randint(batch_size - real_dynamic_data_num, batch_size)
        dynamic_sample_num = batch_size - static_sample_num

        return_dict = {}

        static_random_indices = static_indices[:static_sample_num]
        dynamic_random_indices = dynamic_indices[:dynamic_sample_num]

        for name in self._data_names:
            return_dict[name] = self.static_data[name][static_random_indices]
            return_dict[name] = np.append(return_dict[name], self.dynamic_data[name][dynamic_random_indices], 0)

        return return_dict
