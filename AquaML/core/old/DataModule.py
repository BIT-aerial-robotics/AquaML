'''
数据管理模块用于多进程，跨机器数据管理，也用于多进程信息交互管理模块。提供数据的访问，写入，删除，修改等功能。



The data management module is used for multi-process, cross-machine data management, and also for multi-process information interaction management module. Provide data access, write, delete, modify and other functions.
'''

import abc
from AquaML.core.DataUnit import DataUnit
from AquaML.param.DataInfo import DataInfo
import yaml 
import numpy as np
from typing import Union
from AquaML.communicator.CommunicatorBase import CommunicatorBase
# from AquaML.core.FileSystem import FileSystemBase
import time
from copy import deepcopy

class DataModule:
    '''
    数据管理模块基类，定义了数据管理模块的基本功能，包括数据的访问，写入，删除，修改等功能。
    
    这地方用于支持单机和多机数据管理，以及数据的创建。
    
    这个模块也会维护一些flag标识
    
    该模块会在所有机器上维护一个数据名称表

    # TODO: 需要设置一个全局结束标志位，用于结束所有的数据单元。ß


    The base class of the data management module defines the basic functions of the data management module, including data access, writing, deletion, modification and other functions.
    '''

    def __init__(self, name_scope:str, 
                #  file_system:FileSystemBase,
                 communicator:CommunicatorBase):
        """
        初始化数据管理模块。

        Args:
            name_scope (str): 数据的唯一标识符号，用于数据的访问，写入，删除，修改等功能。
            communicator (CommunicatorBase): 通信模块。用于多进程通信,log等。
        """
        
        
        # 维护一个数据名称表，用于记录所有数据的名称，以及数据的创建者进程，用于关闭数据块
        
        self._data_unit_table = []
        self._data_unit_dict = {}
        
        self._program_running_state = None
        
        self._communicator = communicator
        self._file_system = None
        self._name_scope = name_scope # 数据的唯一标识符号，用于数据的访问，写入，删除，修改等功能。
        
    
    def create_data_unit(self, name:str, unit_info:dict, exist:bool=False):
        """
        创建数据单元，用于数据的访问，写入，删除，修改等功能。
        
        Create a data unit for data access, writing, deletion, modification and other functions.
        """
        
        unit_name = self._name_scope + '_' + name
        
        self.__setattr__(name, DataUnit(name = unit_name,
                                        communicator=self._communicator,
                                        unit_info=unit_info,
                                        exist=exist,
                                        file_system=self._file_system,
                                        ))
        
        self._data_unit_table.append(name)
        
        self._data_unit_dict[name] = self.__getattribute__(name)
        
    ######################################## 初始化模块 ########################################
    def init_data_module(self, file_system):
        """
        初始化数据管理模块。

        Args:
            file_system (FileSystemBase): 文件系统模块。
        """
        
        self._file_system = file_system
        
        self._communicator.logger_info('Init DataModule')
    
   ######################################## 功能接口 ########################################
   
   # 将该scope的data_unit信息写入yaml中
    def write_data_unit_info_toyaml(self, yaml_path:str):
        
        ret_dict = {}
        
        for name, key in self._data_unit_dict.items():
            info = key.get_data_unit_info(str_all=True)
            ret_dict[name] = info
            
        with open(yaml_path, 'w') as f:
            yaml.dump(ret_dict, f)
    
    # 检查代码。 从yaml中读取data_unit信息，创建data_unit
    def read_data_unit_info_fromyaml(self, yaml_path:str, exist:bool=False):
        """
        从yaml中读取data_unit信息，创建data_unit

        Args:
            yaml_path (str): yaml文件路径。
            exist (bool, optional): 是否为已存在的数据单元，如果是，则不需要创建共享内存，直接读取即可。 Defaults to False.
        """
        
        with open(yaml_path, 'r') as f:
            info = yaml.load(f)
            
        for name, info in info.items():
               shape = eval(info['shape'])
               dtype = eval(info['dtype'])
               size = info['size']
               
               unit_info = {'shape':shape, 'dtype':dtype, 'size':size}
               
               self.create_data_unit(name, unit_info, exist=exist)
               
               
    def read_data_unit_info_from_datainfo(self, data_info:DataInfo, exist:bool=False):
        """
        从DataInfo中读取data_unit信息，创建data_unit

        Args:
            data_info (DataInfo): 数据信息。
            exist (bool, optional): 是否为已存在的数据单元，如果是，则不需要创建共享内存，直接读取即可。 Defaults to False.
        """
        
        for name, info in data_info.info_dict.items():
            
            unit_info = {'shape':info.shape, 'dtype':info.dtype, 'size':info.size}
            
            self.create_data_unit(name, unit_info, exist=exist)
            
            if not exist:
                self._communicator.debug_info('DataModule:'+'create data unit:{} in shared memory'.format(name))
            else:
                self._communicator.debug_info('DataModule:'+'read data unit:{}'.format(name))
            
            # if exist:
            #     self._communicator.logger_success('DataModule', 'create data unit:{}'.format(name)) 
            # else:
            #     self._communicator.logger_success('DataModule', 'create data unit:{} in shared memory'.format(name))     
        

    # TODO:是否存在性能问题
    def get_data(self, name: Union[str, list]):
        """
        获取数据
        
        Get data
        """
        if isinstance(name, str):
            return self.__getattribute__(name).get_data()
        
        elif isinstance(name, list):
            ret = []
            for i in name:
                ret.append(self.__getattribute__(i).get_data())
            return ret
    
    ######################################## 和其他模块的接口 ########################################
    def set_RobotAPI_state_unit(self, unit_name_list:list):
        """
        创建数据group,方便机器人API管理数据

        Args:
            unit_name_list (list): 数据名称列表。
        """
        
        self.robot_state_dict = {}
        
        for unit_name in unit_name_list:
            self.robot_state_dict[unit_name] = self.__getattribute__(unit_name)
            self._communicator.logger_info('DataModule:'+'set robot state unit:{}'.format(unit_name))
            
    def set_RobotAPI_state_unit_map(self, unit_name_list:list, map_names:list):
        """
        创建数据group,方便机器人API管理数据。
        
        按照unit_name_list的进行识别unit_name，按照map_names获取数据。
        
        存储的数据，使用时需要进行映射。

        Args:
            unit_name_list (list): 数据名称列表。
            map_names (list): 映射名称列表。
        """
        
        self.robot_state_dict = {}
        
        for unit_name, map_name in zip(unit_name_list, map_names):
            self.robot_state_dict[map_name] = self.__getattribute__(unit_name)
            self._communicator.logger_info('set robot state unit:{}'.format(unit_name))
            
    def set_RobotAPI_next_state_unit(self, unit_names:list):
        """
        创建数据group,方便机器人API管理数据

        Args:
            unit_names (list): 数据名称列表。
        """
        
        self.robot_next_state_dict = {}
        
        for unit_name in unit_names:
            self.robot_next_state_dict[unit_name] = self.__getattribute__(unit_name)
            self._communicator.logger_info('set robot next state unit:{}'.format(unit_name))
    
    def set_RobotAPI_next_state_unit_map(self, unit_names:list, map_names:list):
        """
        创建数据group,方便机器人API管理数据。
        
        存储的数据，使用时需要进行映射。

        Args:
            unit_names (list): 数据名称列表。
            map_names (list): 映射名称列表。
        """
        
        self.robot_next_state_dict = {}
        
        for unit_name, map_name in zip(unit_names, map_names):
            self.robot_next_state_dict[map_name] = self.__getattribute__(unit_name)
            self._communicator.logger_info('set robot next state unit:{}'.format(unit_name))
    
    def set_rewards_unit(self, unit_names:list):
        """
        创建数据group,方便机器人API管理数据

        Args:
            unit_names (list): 数据名称列表。
        """
        
        self.rewards_dict = {}
        
        for unit_name in unit_names:
            self.rewards_dict[unit_name] = self.__getattribute__(unit_name)
            self._communicator.logger_info('set rewards unit:{}'.format(unit_name))
            
    
    def set_rewards_unit_map(self, unit_names:list, map_names:list):
        """
        创建数据group,方便机器人API管理数据。
        
        存储的数据，使用时需要进行映射。

        Args:
            unit_names (list): 数据名称列表。
            map_names (list): 映射名称列表。
        """
        
        self.rewards_dict = {}
        
        for unit_name, map_name in zip(unit_names, map_names):
            self.rewards_dict[map_name] = self.__getattribute__(unit_name)
            self._communicator.logger_info('set rewards unit:{}'.format(unit_name))
    
    def set_mask_unit(self, unit_name:str):
        """
        创建数据group,方便机器人API管理数据

        Args:
            unit_name (str): 数据名称。
        """
        
        self.mask:DataUnit = self.__getattribute__(unit_name)
        self._communicator.logger_info('set mask unit:{}'.format(unit_name))
    
    # def set_mask_unit_map(self, unit_name:str, map_name:str):
    #     """
    #     创建数据group,方便机器人API管理数据。
        
    #     存储的数据，使用时需要进行映射。

    #     Args:
    #         unit_name (str): 数据名称。
    #         map_name (str): 映射名称。
    #     """
        
    #     self.mask = self.__getattribute__(unit_name)
    #     self._communicator.logger_info('DataModule', 'set mask unit:{}'.format(unit_name))
            
    def set_RobotAPI_control_actions_unit(self, unit_name_list:list):
        """
        创建数据group,方便机器人API管理数据

        Args:
            unit_name_list (list): 数据名称列表。
        """
        
        self.robot_control_actions_dict = {}
        
        for unit_name in unit_name_list:
            self.robot_control_actions_dict[unit_name] = self.__getattribute__(unit_name)
            self._communicator.logger_info('set robot control unit:{}'.format(unit_name))
    
    def set_RobotAPI_control_actions_unit_map(self, unit_name_list:list, map_names:list):
        """
        创建数据group,方便机器人API管理数据。
        
        存储的数据，使用时需要进行映射。

        Args:
            unit_name_list (list): 数据名称列表。
            map_names (list): 映射名称列表。
        """
        
        self.robot_control_actions_dict = {}
        
        for unit_name, map_name in zip(unit_name_list, map_names):
            self.robot_control_actions_dict[map_name] = self.__getattribute__(unit_name)
            self._communicator.logger_info('set robot control unit:{}'.format(unit_name))
    
    # def set_RobotAPI_control_action_unit(self, unit_name:str):
    #     """
    #     创建数据group,方便机器人API管理数据

    #     Args:
    #         unit_name (str): 数据名称。
    #     """
        
    #     self.robot_control_action = self.__getattribute__(unit_name)
    #     self._communicator.logger_info('DataModule', 'set robot control unit:{}'.format(unit_name))
    
    def set_RobotAPI_control_action_unit(self, unit_name:str):
        """
        创建数据group,方便机器人API管理数据

        Args:
            unit_name (str): 数据名称。
        """
        
        self.robot_control_action:DataUnit = self.__getattribute__(unit_name)
        self._communicator.logger_info('set robot control unit:{}'.format(unit_name))
        
    # def set_RobotAPI_control_action_unit_map(self, unit_name:str,):
        
    #     """
    #     创建数据group,方便机器人API管理数据。
        
    #     存储的数据，使用时需要进行映射。

    #     Args:
    #         unit_name (str): 数据名称。
    #         map_name (str): 映射名称。
    #     """
        
    #     self.robot_control_action = self.__getattribute__(unit_name)
    #     self._communicator.logger_info('DataModule', 'set robot control unit:{}'.format(unit_name))
    
    def set_RobotAPI_state_update_flag_unit(self, flag_name:str):
        """
        
        判断是否能够更新状态，用于确保每个策略的时间戳相同。

        Args:
            flag_name (str): flag名称。
        """
        
        self.robot_state_update_flag:DataUnit = self.__getattribute__(flag_name)
        self._communicator.logger_info('set robot state update flag:{}'.format(flag_name))
    
    def set_RobotAPI_control_update_flag_unit(self, flag_name:str):
        
        """
        判断是否能够更新控制信号，用于确保每个策略的时间戳相同。
        
        Args:
            flag_name (str): flag名称。
        
        """
        
        self.robot_control_update_flag = self.__getattribute__(flag_name)
        self._communicator.logger_info('set robot control update flag:{}'.format(flag_name))
        
    # def set_canidate_actions_unit(self, unit_name:str):
    #     """
    #     设置候选动作数据单元,候选动作unit格式为[policy_num, action_dim]
        

    #     Args:
    #         unit_name (str): 候选动作数据单元名称。
    #     """
        
    #     self.candidate_actions = self.__getattribute__(unit_name)
    #     self._communicator.logger_info('DataModule', 'set candidate actions unit:{}'.format(unit_name))
    
    def set_canidate_actions_unit(self, unit_names:list):
        """
        设置候选动作数据单元,候选动作unit格式为[policy_num, action_dim]
        
        现在仅仅只支持一个候选动作数据单元

        Args:
            unit_names (list): 候选动作数据单元名称列表。
        """
        
        self.candidate_actions_dict = {}
        
        for unit_name in unit_names:
            self.candidate_actions_dict[unit_name] = self.__getattribute__(unit_name)
            self._communicator.logger_info('set candidate actions unit:{}'.format(unit_name))
    
    def set_canidate_actions_unit_map(self, unit_names:list, map_names:list):
        """
        设置候选动作数据单元,候选动作unit格式为[policy_num, action_dim]
        
        现在仅仅只支持一个候选动作数据单元

        Args:
            unit_names (list): 候选动作数据单元名称列表。
            map_names (list): 映射名称列表。
        """
        
        self.candidate_actions_dict = {}
        
        for unit_name, map_name in zip(unit_names, map_names):
            self.candidate_actions_dict[map_name] = self.__getattribute__(unit_name)
            self._communicator.logger_info( 'set candidate actions unit:{}'.format(unit_name))
    
    def set_canidate_action_unit(self, unit_name:str):
        """
        设置候选动作数据单元,候选动作unit格式为[policy_num, action_dim]
        
        现在仅仅只支持一个候选动作数据单元

        Args:
            unit_name (str): 候选动作数据单元名称。
        """
        
        self.candidate_action:DataUnit = self.__getattribute__(unit_name)
        self._communicator.logger_info('set candidate actions unit:{}'.format(unit_name))
    
    def set_canidate_action_unit_map(self, unit_name:str, map_name:str):
        """
        设置候选动作数据单元,候选动作unit格式为[policy_num, action_dim]
        
        现在仅仅只支持一个候选动作数据单元

        Args:
            unit_name (str): 候选动作数据单元名称。
            map_name (str): 映射名称。
        """
        
        self.candidate_action:DataUnit = self.__getattribute__(unit_name)
        self._communicator.logger_info('set candidate actions unit:{}'.format(unit_name))

    @property
    def action(self):
        return self.candidate_action
    
    # def set_paired_actions_unit(self, unit_name:str):
    #     """
    #     设置配对动作数据单元,配对动作unit格式为[policy_num, action_dim]
        
    #     Args:
    #         unit_name (str): 配对动作数据单元名称。
    #     """
        
    #     self.paired_actions = self.__getattribute__(unit_name)
    #     self._communicator.logger_info('DataModule', 'set paired actions unit:{}'.format(unit_name))
        
    def set_paired_actions_unit(self, unit_names:list):
        """
        设置配对状态数据单元,配对状态unit格式为[policy_num, state_dim]
        
        Args:
            unit_names (list): 配对状态数据单元名称列表。
        """
        
        self.paired_actions_dict = {}
        
        for unit_name in unit_names:
            self.paired_actions_dict[unit_name] = self.__getattribute__(unit_name)
            self._communicator.logger_info('set paired actions unit:{}'.format(unit_name))
            
    def set_paired_action_unit(self, unit_name:str):
        """
        设置配对动作数据单元,配对动作unit格式为[policy_num, action_dim]
        
        Args:
            unit_name (str): 配对动作数据单元名称。
        """
        
        self.paired_action = self.__getattribute__(unit_name)
        self._communicator.logger_info('set paired actions unit:{}'.format(unit_name))
    
    def set_paired_states_unit(self, unit_names:list):
        """
        设置配对状态数据单元,配对状态unit格式为[policy_num, state_dim]
        
        Args:
            unit_names (list): 配对状态数据单元名称列表。
        """
        
        self.paired_states_dict = {}
        
        for unit_name in unit_names:
            self.paired_states_dict[unit_name] = self.__getattribute__(unit_name)
            self._communicator.logger_info('set paired states unit:{}'.format(unit_name))
    
    def set_paired_flags_unit(self, unit_name:str):
        """
        设置配对flag数据单元,配对flag unit格式为[policy_num]
        
        Args:
            unit_name (str): 配对flag数据单元名称。
        """
        
        self.paired_flags = self.__getattribute__(unit_name)
        self._communicator.logger_info('set paired flags unit:{}'.format(unit_name))
        
    ######################################## 各种flag ########################################
    def set_control_action_flag(self, flag_name:str):
        """
        
        设置控制动作的flag标志位。
        
        
        该标志位判断动作策略是否已经更新完毕。
        在生成位置只能将标志位从false变为true。
        使用位置能够将标志位从true变为false。
        
        
        

        Args:
            flag_name (str): flag名称。

        Raises:
            AttributeError: flag_name not in data_module
        """
        
        try:
            self.control_action_flag:DataUnit = self.__getattribute__(flag_name)
            self._communicator.logger_info('set control action flag:{}'.format(flag_name))
        except AttributeError:
            self._communicator.logger_error('flag_name not in data_module')
            raise AttributeError('flag_name not in data_module')
    
    def set_control_state_flag(self, flag_name:str):
        """
        设置控制状态的flag标志位。
        
        该标志位判断状态策略是否已经更新完毕。
        在生成位置只能将标志位从false变为true。
        使用位置能够将标志位从true变为false。

        Args:
            flag_name (str): flag名称。

        Raises:
            AttributeError: flag_name not in data_module
        """
        
        try:
            self.control_state_flag = self.__getattribute__(flag_name)
            self._communicator.logger_info('set control state flag:{}'.format(flag_name))
        except AttributeError:
            self._communicator.logger_error('flag_name not in data_module')
            raise AttributeError('flag_name not in data_module')
        
    def set_canidate_action_flag(self, flag_name:str):
        """
        设置候选动作的flag标志位。
        
        该标志位判断候选动作是否已经更新完毕。
        在生成位置只能将标志位从false变为true。
        使用位置能够将标志位从true变为false。

        Args:
            flag_name (str): flag名称。

        Raises:
            AttributeError: flag_name not in data_module
        """
        
        try:
            self.candidate_action_flag:DataUnit = self.__getattribute__(flag_name)
            self._communicator.logger_info('set candidate action flag:{}'.format(flag_name))
        except AttributeError:
            self._communicator.logger_error('flag_name not in data_module')
            raise AttributeError('flag_name not in data_module')
        
    def set_send_flag(self, flag_name:str):
        """
        设置发送flag标志位。
        
        该标志位判断数据是否已经发送完毕。
        在生成位置只能将标志位从false变为true。
        使用位置能够将标志位从true变为false。

        Args:
            flag_name (str): flag名称。

        Raises:
            AttributeError: flag_name not in data_module
        """
        
        try:
            self.send_flag = self.__getattribute__(flag_name)
            self._communicator.logger_info('set send flag:{}'.format(flag_name))
        except AttributeError:
            self._communicator.logger_error('flag_name not in data_module')
            raise AttributeError('flag_name not in data_module')


    ######################################## FileSystem接口 ########################################
    def set_history_number(self, scope_names:list):
        """
        设置历史数据的序列号同步通讯标志位。

        Args:
            scope_names (list): scope名称列表。
        """
        self.history_number_dict = {}
        
        for scope_name in scope_names:
            self.history_number_dict[scope_name] = self.__getattribute__('file_system_'+scope_name)
            self._communicator.logger_info('{} set history number:{}'.format(scope_name, self.history_number_dict[scope_name]))
    
    def set_history_number_map_inv(self, scope_names:list, map_names:list):
        """
        设置历史数据的序列号同步通讯标志位。
        
        从data_module中获取数据使用的是map_names。
        当直接从dict中获取数据时，使用的是scope_names。

        Args:
            scope_names (list): scope名称列表。
            map_names (list): 映射名称列表。
        """
        self.history_number_dict = {}
        
        for scope_name, map_name in zip(scope_names, map_names):
            self.history_number_dict[scope_name] = self.__getattribute__(map_name)
            self._communicator.logger_info('{} set history number:{}'.format(scope_name, self.history_number_dict[scope_name]))
        
    ######################################## 获取信息接口部分 ########################################
    
    def get_unit_shape(self, name:str):
        """
        获取数据单元的shape信息。
        
        Get the shape information of the data unit.
        """
        return self.__getattribute__(name).shape
    
    
    def get_unit(self, unit):
        """
        获取数据单元。

        Args:
            unit (str,data_unit): 数据单元名称。

        Raises:
            AttributeError: unit not in data_module
        """
        
        if isinstance(unit, str):
            try:
                return self.__getattribute__(unit)
            except AttributeError:
                self._communicator.logger_error('unit not in data_module')
                raise AttributeError('unit not in data_module')
        elif isinstance(unit, DataUnit):
            return unit
        else:
            self._communicator.logger_error('unit type error')
            raise AttributeError('unit type error')
    
    ######################################## 重载部分 ########################################
   
    def __del__(self):
        """
        关闭数据块
       
        Close the data block
        """
        for unit_name in self._data_unit_table:
            self.__getattribute__(unit_name).close()
            
    ######################################## 框架运行时需要的参数 ########################################
    
    def get_program_running_state(self)->DataUnit:
        """           
        获取运行参数状态。
        
        Get the running parameter status.
        """
        if self._program_running_state is None:
            try:
                self._program_running_state = self.__getattribute__('program_running_state')
            except AttributeError:
                self._communicator.logger_error('program_running_state is not in data_module')
                raise AttributeError('program_running_state is not in data_module') 
            
        self._communicator.debug_info('get program running state{}'.format(self._program_running_state))
        return self._program_running_state 
    
    @property
    def get_program_end(self)->bool:
        """
        获取程序结束状态。
        
        Get the program end status.
        """
        return self.get_program_running_state().get_data()[0][1]
    
    ######################################## 框架运行需要的函数 ########################################  
    
    def wait_for_flag_change(self, flag, time_out:int=1, check_interval:int=0.001):
        """
        #TODO：降低CPU占用率
        等待flag标志位的变化。
        
        该函数会让进程等待flag标志位的变化，直到flag标志位发生变化，或者超时。
        
        当超时会发出警告。并且尝试继续运行程序。

        Args:
            flag: flag名称。flag必须存在于data_module中。
            time_out (int, optional): 超时时间。 Defaults to 1.
            check_interval (int, optional): 检查间隔时间。 Defaults to 0.001.
        """
        
        start_time = time.time()
        
        flag_unit = self.get_unit(flag)
        
        last_flag = deepcopy(flag_unit.get_data()[0])
        
        while True:
            flag = deepcopy(flag_unit.get_data()[0])
            
            if last_flag != flag:
                break
            # if flag:
            #     break
            if time.time() - start_time > time_out:
                self._communicator.logger_warning('wait for {} change time out'.format(flag_unit.name))
                break
            
            time.sleep(check_interval)
            
    def wait_for_flag_to_true(self, flag, time_out:int=1, check_interval:int=0.001):
        
        """
        等待flag标志位为true。
        
        该函数会让进程等待flag标志位为true，直到flag标志位为true，或者超时。
        
        当超时会发出警告。并且尝试继续运行程序。
        
        Args:
            flag: flag名称。flag必须存在于data_module中。
            time_out (int, optional): 超时时间。 Defaults to 1.
            check_interval (int, optional): 检查间隔时间。 Defaults to 0.001.
        """
        
        init_flag = self.get_unit(flag).get_data()[0]
        
        flag_name = self.get_unit(flag).name
        
        if init_flag:
            self._communicator.logger_warning('flag:{} is already true'.format(flag_name))
        else:
            self.wait_for_flag_change(
                flag=flag,
                time_out=time_out,
                check_interval=check_interval
            )
    
    def wait_for_array_flag_to_pointed_value(self, flag, value, time_out:int=1, check_interval:int=0.001):
        """
        等待flag标志位为指定值。
        
        该函数会让进程等待flag标志位为指定值，直到flag标志位为指定值，或者超时。
        
        当超时会发出警告。并且尝试继续运行程序。
        
        Args:
            flag: flag名称。flag必须存在于data_module中。
            value: 指定值。
            time_out (int, optional): 超时时间。 Defaults to 1.
            check_interval (int, optional): 检查间隔时间。 Defaults to 0.001.
        """
        
        start_time = time.time()
        
        flag_unit = self.get_unit(flag)
        
        while True:
            flag = flag_unit.get_data()
            sum_flag = np.sum(flag)
            
            if sum_flag == value:
                break
            if time.time() - start_time > time_out:
                self._communicator.logger_warning('wait for flag change time out')
                break
            
            time.sleep(check_interval)

    def wait_whole_flag_to_value(self, flag, value, time_out: int = 1, check_interval: int = 0.001):
        """
        等待flag标志位为指定值。

        该函数会让进程等待flag标志位为指定值，直到flag标志位为指定值，或者超时。

        当超时会发出警告。并且尝试继续运行程序。

        Args:
            flag: flag名称。flag必须存在于data_module中。
            value: 指定值。
            time_out (int, optional): 超时时间。 Defaults to 1.
            check_interval (int, optional): 检查间隔时间。 Defaults to 0.001.
        """

        start_time = time.time()

        flag_unit = self.get_unit(flag)

        while True:
            flag = flag_unit.get_data()
            self._communicator.logger_info('{}: {}, waiting all to {}'.format(flag_unit.name, flag, value))
            if value:
                if flag[0].all():
                    break
            else:
                flag1 = ~flag[0]
                if flag1.all():
                    break
            if time.time() - start_time > time_out:
                self._communicator.logger_warning('wait for {} change time out'.format(flag_unit.name))
                break

            time.sleep(check_interval)

    def wait_indexed_flag_to_value(self, flag, index, value, time_out: int = 1, check_interval: int = 0.001):
        """
        等待flag标志位为指定值。

        该函数会让进程等待flag标志位为指定值，直到flag标志位为指定值，或者超时。

        当超时会发出警告。并且尝试继续运行程序。

        Args:
            flag: flag名称。flag必须存在于data_module中。
            value: 指定值。
            time_out (int, optional): 超时时间。 Defaults to 1.
            check_interval (int, optional): 检查间隔时间。 Defaults to 0.001.
        """

        start_time = time.time()

        flag_unit = self.get_unit(flag)

        while True:
            flag = flag_unit.get_data()
            self._communicator.logger_info('{}: {}, value: {}, index: {}'.format(flag_unit.name, flag[0][index], value, index))
            if flag[0][index] == value:
                break
            if time.time() - start_time > time_out:
                self._communicator.logger_warning('wait for {} change time out'.format(flag_unit.name))
                break

            time.sleep(check_interval)

    def wait_for_flag_to_false(self, flag, time_out:int=1, check_interval:int=0.001):
        
        """
        等待flag标志位为false。
        
        该函数会让进程等待flag标志位为false，直到flag标志位为false，或者超时。
        
        当超时会发出警告。并且尝试继续运行程序。
        
        Args:
            flag: flag名称。flag必须存在于data_module中。
            time_out (int, optional): 超时时间。 Defaults to 1.
            check_interval (int, optional): 检查间隔时间。 Defaults to 0.001.
        
        """
        
        init_flag = self.get_unit(flag).get_data()[0]
        
        flag_name = self.get_unit(flag).name
        
        if not init_flag:
            self._communicator.logger_warning('flag:{} is already false'.format(flag_name))
        else:
            self.wait_for_flag_change(
                flag=flag,
                time_out=time_out,
                check_interval=check_interval
            )
    
    def wait_program_start(self):
        """
        等待程序开始运行。
        
        该函数会让进程等待程序开始运行，直到程序开始运行，或者超时。
        
        每隔0.1s检查一次程序是否开始运行。
        
        当超时会发出警告。并且尝试继续运行程序。
        """
        
        while True:
            program_running_state = self.get_program_running_state().get_data()[0][0]
            self._communicator.logger_info('flag:{} is {}'.format(self._program_running_state.name,
                                                                  self._program_running_state.get_data()))
            if program_running_state:
                self._communicator.logger_info('program start')
                break
            else:
                self._communicator.logger_warning('waiting for program start')
            
            time.sleep(0.05)
        

class FileDataModule:
    """
    文件数据管理模块，用于多进程，跨机器数据管理，也用于多进程信息交互管理模块。提供数据的访问，写入，删除，修改等功能。
    
    File data management module, used for multi-process, cross-machine data management, and also for multi-process information interaction management module. Provide data access, write, delete, modify and other functions.
    """
    
    def __init__(self, name_scope:str):
            
        self._name_scope = name_scope
        
            
            