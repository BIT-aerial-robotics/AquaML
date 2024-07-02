from AquaML.buffer.DynamicBufferBase import DynamicBufferBase
from AquaML.communicator.CommunicatorBase import CommunicatorBase
from AquaML.core.old.DataModule import DataModule
from AquaML.core.old.FileSystem import DefaultFileSystem
import numpy as np
import copy
from threading import Thread, ThreadError

# TODO: 这个地方需要重新开发。
class RealCollectBufferBase(DynamicBufferBase):
    """
    RealCollectBuffer算法基类，用于定义RealCollectBuffer算法的基本功能。
    
    该类用于在真机使用时候的数据收集，用于数据的存储和读取。
    
    在真机下啊，算法和数据收集部分很难同步进行，该类可以创建一个新的线程用于数据的收集。
    """
    
    def __init__(self,
                 capacity:int,
                 data_names:list,
                 data_module:DataModule,
                 communicator:CommunicatorBase,
                 file_system:DefaultFileSystem,
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
        
        super(RealCollectBufferBase, self).__init__(
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
            
            # 等待当前交互是否结束，结束开始读取数据
            self._data_module.wait_for_flag_to_true(
                flag=self._data_module.send_flag,
                time_out=self._communicator.wait_time_out,
                check_interval=self._communicator.check_time_interval
            )
            
            # 获得s,a,s',r
            # 将state添加到buffer中
            
            for state_name, state_unit in self._data_module.robot_state_dict.items():
                # self._communicator.debug_info('RealCollectBufferBase', 'Aquire state:{}'.format(state_unit.get_data()))
                self.data[state_name].append(state_unit.get_data())
                
            for next_state_name, next_state_unit in self._data_module.robot_next_state_dict.items():
                # self._communicator.debug_info('RealCollectBufferBase', 'Aquire next state:{}'.format(next_state_unit.get_data()))
                self.data[next_state_name].append(next_state_unit.get_data())
                
            self.data['action'].append(self._data_module.robot_control_action.get_data())
            
            for reward_name, reward_unit in self._data_module.rewards_dict.items():
                # self._communicator.debug_info('RealCollectBufferBase', 'Aquire reward:{}'.format(reward_unit.get_data()))
                self.data[reward_name].append(reward_unit.get_data())
                
            # 重置send_flag
            self._data_module.send_flag[0] = False
            
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
    
    @property
    def thread(self):
        return self._thread
                
                
                
            