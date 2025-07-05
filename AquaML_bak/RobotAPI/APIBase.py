import abc
from AquaML.core.old.DataModule import DataModule
from AquaML.core.old.FileSystem import DefaultFileSystem
from AquaML.communicator.CommunicatorBase import CommunicatorBase
import time

class APIBase(abc.ABC):
    """
    APIBase is the base class of all robot API classes.
    # TODO：未来升级为可以在外部定义的API。
    """
    def __init__(self, 
                 name:str ,
                 data_module:DataModule,
                 file_system:DefaultFileSystem,
                 communicator:CommunicatorBase,
                 run_frequency:int=600,
                #  param:dict=None
    ):
        """
        用于初始化机器人的API。

        Args:
            name (str): 实例名称，区分多个机器人。
            data_module (DataModule): 数据传输模块，用于和机器人交互。
            param (dict, optional): 机器人的参数。 Defaults to None.
        """
        self._file_system = file_system
        self._communicator = communicator
        self._data_module = data_module
        
        self._communicator.logger_info('APIBase is initializing.')
        
        # if param is not None:
        #     default_param.update(param)
        
        # self._param = default_param
        self._name = name
        self._publisher_mapping_func_dict = {}
        
        self._run_frequency = run_frequency
        self._run_time = 1.0 / self._run_frequency
        self._state_mapping_dict = {}
        self._action_mapping_dict = {}
        
        # 检测模块是否合法
        try:
            self._data_module.robot_state_dict
        except ValueError:
            raise Exception('data_module must have robot_state_dict')
        
        try:
            self._data_module.robot_control_action
        except ValueError:
            raise Exception('data_module must have robot_control_dict')
        
    @abc.abstractmethod
    def get_state(self):
        """
        Get the state from the robot.
        
        获取机器人的状态后将其写入到data_module中。
        """
        pass
    
    @abc.abstractmethod
    def control(self):
        """
        Control the robot.
        
        从data_module中获取控制信号，然后将其发送给机器人。
        """
        pass
    
    @abc.abstractmethod
    def run(self):
        """
        Run the robot.
        
        1. Get the state from the robot.
        2. Get the control signal from the data_module.
        3. Send the control signal to the robot.
        
        该函数独立进程运行。该函数将会循环执行。负责更新状态更新，并且保证更新时候，每个策略保证使用的是同一帧数据。
         
        在data module里面会有一个flag用于确认本次状态是否更新。
        
        必须实现。
        """
    
    
    ############################### 运行接口部分 #################################
    
    def _run_once(self):
        """
        Run the robot once.
        
        1. Get the state from the robot.
        2. Get the control signal from the data_module.
        3. Send the control signal to the robot.
        
        该函数负责将将数据传输到AquaML框架中，并将框架的控制信号给ROS。
        """
        
        # TODO:确认一下是不是可以啊
        
        self.get_state()
        
        self.control()
        
    def set_communicator(self, communicator:CommunicatorBase):
        """
        Set the communicator for the API.
        
        Args:
            communicator (CommunicatorBase): 通讯器。
        """
        self._communicator = communicator
        
        
        
            
            