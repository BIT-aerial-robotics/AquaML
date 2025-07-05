from AquaML.buffer.DynamicBufferBase import DynamicBufferBase
from AquaML.communicator.CommunicatorBase import CommunicatorBase
from AquaML.core.old.DataModule import DataModule
from AquaML.core.old.FileSystem import DefaultFileSystem
import numpy as np
import copy
from threading import Thread, ThreadError


# TODO: 这个地方需要重新开发。
class RealCollectBuffer(DynamicBufferBase):
    """
    RealCollectBuffer算法基类，用于定义RealCollectBuffer算法的基本功能。
    
    该类用于在真机使用时候的数据收集，用于数据的存储和读取。
    
    在真机下啊，算法和数据收集部分很难同步进行，该类可以创建一个新的线程用于数据的收集。
    """

    def __init__(self,
                 capacity: int,
                 data_names: list,
                 data_module: DataModule,
                 communicator: CommunicatorBase,
                 file_system: DefaultFileSystem,
                 ):
        """
        RealCollectBuffer算法基类，用于定义RealCollectBuffer算法的基本功能。

        Args:
            capacity (int): buffer的容量。
            data_names (list): 数据的名称列表。
            data_module (DataModule): 数据模块。用于数据的存储和读取。
            communicator (CommunicatorBase): 通信模块。用于多进程通信以及log等。
            shared_list (bool, optional): 是否共享数据。默认为False。未来支持的功能。
        """

        super(RealCollectBuffer, self).__init__(
            capacity=capacity,
            data_names=data_names,
            data_module=data_module,
            communicator=communicator,
            file_system=file_system
        )

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
                    self.data[state_name] = state

                for next_state_name, next_state_unit in self._data_module.robot_next_state_dict.items():
                    # self._communicator.debug_info('RealCollectBufferBase', 'Aquire next state:{}'.format(next_state_unit.get_data()))
                    next_state = copy.deepcopy(next_state_unit.get_data())
                    self.data[next_state_name] = next_state

                action = copy.deepcopy(self._data_module.robot_control_action.get_data())
                self.data['action'] = action

                mask = copy.deepcopy(self._data_module.env_mask.get_data())
                self.data['mask'] = mask

                for reward_name, reward_unit in self._data_module.rewards_dict.items():
                    # self._communicator.debug_info('RealCollectBufferBase', 'Aquire reward:{}'.format(reward_unit.get_data()))
                    reward = copy.deepcopy(reward_unit.get_data())
                    self.data[reward_name] = reward

                self.capacity_count += 1
                self._communicator.logger_info('0 buffer now is {}'.format(self.data))
            elif self.capacity_count < self._capacity:
                # 获得s,a,s',r
                # 将state添加到buffer中
                for state_name, state_unit in self._data_module.robot_state_dict.items():
                    # self._communicator.logger_info('RealCollectBufferBase aquire state:{}'.format(state_unit.get_data()))
                    state = copy.deepcopy(state_unit.get_data())
                    self.data[state_name] = np.append(self.data[state_name], state, 0)

                for next_state_name, next_state_unit in self._data_module.robot_next_state_dict.items():
                    # self._communicator.debug_info('RealCollectBufferBase', 'Aquire next state:{}'.format(next_state_unit.get_data()))
                    next_state = copy.deepcopy(next_state_unit.get_data())
                    self.data[next_state_name] = np.append(self.data[next_state_name], next_state, 0)

                action = copy.deepcopy(self._data_module.robot_control_action.get_data())
                self.data['action'] = np.append(self.data['action'], action, 0)

                mask = copy.deepcopy(self._data_module.env_mask.get_data())
                self.data['mask'] = np.append(self.data['mask'], mask, 0)

                for reward_name, reward_unit in self._data_module.rewards_dict.items():
                    # self._communicator.debug_info('RealCollectBufferBase', 'Aquire reward:{}'.format(reward_unit.get_data()))
                    reward = copy.deepcopy(reward_unit.get_data())
                    self.data[reward_name] = np.append(self.data[reward_name], reward, 0)

                self.capacity_count += 1
                self._communicator.logger_info('1 buffer now is {}'.format(self.data))
            else:
                Index = self.capacity_count % self._capacity
                # 获得s,a,s',r
                # 将state添加到buffer中
                for state_name, state_unit in self._data_module.robot_state_dict.items():
                    # self._communicator.debug_info('RealCollectBufferBase', 'Aquire state:{}'.format(state_unit.get_data()))
                    state = copy.deepcopy(state_unit.get_data())
                    self.data[state_name][Index] = state

                for next_state_name, next_state_unit in self._data_module.robot_next_state_dict.items():
                    # self._communicator.debug_info('RealCollectBufferBase', 'Aquire next state:{}'.format(next_state_unit.get_data()))
                    self.data[next_state_name][Index] = copy.deepcopy(next_state_unit.get_data())

                self.data['action'][Index] = copy.deepcopy(self._data_module.robot_control_action.get_data())

                mask = copy.deepcopy(self._data_module.env_mask.get_data())
                self.data['mask'][Index] = mask

                for reward_name, reward_unit in self._data_module.rewards_dict.items():
                    # self._communicator.debug_info('RealCollectBufferBase', 'Aquire reward:{}'.format(reward_unit.get_data()))
                    self.data[reward_name][Index] = copy.deepcopy(reward_unit.get_data())

                self.capacity_count += 1
                self._communicator.logger_info('2 buffer now is {}'.format(self.data))
            # 重置send_flag
            self._data_module.send_flag[0] = [False]
            self._communicator.logger_info('flag:{} is {}'.format(self._data_module.send_flag.name,
                                                                  self._data_module.send_flag.get_data()))

            # flag = self._data_module.paired_data_flag.get_data()
            # self._communicator.debug_info('RealCollectBufferBase', 'paired flag:{}'.format(flag))

            # if np.sum(flag) == 2:
            #     # 数据处于可读写状态

            #     # 获取paired data
            #     store_data = {}

            #     for data_name, data_unit in self._data_module.paired_states_dict.items():
            #         try:
            #             store_data[data_name] = copy.deepcopy(data_unit.get_data())
            #             self._communicator.debug_info('RealCollectBufferBase', 'Aquire data:{}'.format(store_data[data_name]))
            #         except ThreadError:
            #             self._communicator.debug_info('RealCollectBufferBase', 'ThreadError')
            #             break
            #         except RuntimeError:
            #             self._communicator.debug_info('RealCollectBufferBase', 'RuntimeError')
            #             break
            #         except Exception as e:
            #             self._communicator.debug_info('RealCollectBufferBase', 'Error:{}'.format(e))
            #             break
            #         # store_data[data_name] = copy.deepcopy(data_unit.get_data())
            #         # self._communicator.debug_info('RealCollectBufferBase', 'Aquire data:{}'.format(store_data[data_name]))

            #     for data_name, data_unit in self._data_module.paired_actions_dict.items():

            #         try:
            #             store_data[data_name] = copy.deepcopy(data_unit.get_data())
            #             self._communicator.debug_info('RealCollectBufferBase', 'Aquire data:{}'.format(store_data[data_name]))
            #         except ThreadError:
            #             self._communicator.debug_info('RealCollectBufferBase', 'ThreadError')
            #             break
            #         except RuntimeError:
            #             self._communicator.debug_info('RealCollectBufferBase', 'RuntimeError')
            #             break
            #         except Exception as e:
            #             self._communicator.debug_info('RealCollectBufferBase', 'Error:{}'.format(e))
            #             break
            #         # store_data[data_name] = copy.deepcopy(data_unit.get_data())
            #         # self._communicator.debug_info('RealCollectBufferBase', 'Aquire data:{}'.format(store_data[data_name]))

            #     # 数据存储

            #     self.append(store_data)

    def run(self):
        """
        运行算法。
        """
        self.append_real()

    # 多线程启动数据收集
    def run_thread(self):
        """
        启动数据收集线程
        """

        self._thread = Thread(target=self.append_real)
        # t = Thread(target=self.append_real)
        # t.start()
        self._thread.start()

        return self._thread

    def sample_data(self, batch_size: int):
        """
        采样数据。

        Args:
            batch_size (int): 采样数据的大小。
        """
        real_data_num = self._capacity if self.capacity_count > self._capacity else self.capacity_count
        indices = np.random.permutation(real_data_num)

        return_dict = {}

        sample_data_num = batch_size if real_data_num > batch_size else real_data_num
        random_indices = indices[:sample_data_num]

        for name in self.data.keys():
            return_dict[name] = self.data[name][random_indices]

        return return_dict

    @property
    def thread(self):
        return self._thread
