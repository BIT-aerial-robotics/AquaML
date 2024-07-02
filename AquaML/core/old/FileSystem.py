'''

该模块用于管理文件系统，多机之间的文件传输，每个模型存储的位置等等
'''

from abc import ABC, abstractmethod
import os
import datetime
from AquaML import DataModule
from AquaML.communicator.CommunicatorBase import CommunicatorBase
import numpy as np
import yaml


def mkdir(path: str):
    """
    创建文件夹

    :param path: 文件夹路径
    :return:
    """

    current_path = os.getcwd()
    path = os.path.join(current_path, path)
    if not os.path.exists(path):
        os.makedirs(path)



class ScopeFileElement:
    """
    存储scope的文件元素
    
    """
    
    def __init__(self,
                 scope_name:str,
                 cache_root_path:str,
                 history_model_root_path:str,
                 log_root_path:str,
                 log_name:str,
                 communicator:CommunicatorBase,
                    ):
        """
        初始化文件元素
        
        我们可以将不同算法理解成不同的scope，每个scope都有自己的cache，history_model，log文件夹。

        Args:
            scope_name: scope名称
            cache_root_path: 缓存根路径
            history_model_root_path: 历史模型根路径
            log_root_path: log根路径
            log_name: log名称
            communicator: 通信模块
        """
        communicator.logger_info('ScopeFileElement '+'Add scope: {}'.format(scope_name))
        
        self.commucator = communicator
        self.scope_name = scope_name
        
        self.cache_path = os.path.join(cache_root_path, scope_name)
        self.commucator.logger_info('ScopeFileElement '+'{} scope cache path: {}'.format(scope_name, self.cache_path))
        
        self.history_model_path = os.path.join(history_model_root_path, scope_name)
        self.commucator.logger_info('ScopeFileElement '+'{} scope history model path: {}'.format(scope_name, self.history_model_path))
        
        self.log_path = os.path.join(log_root_path, scope_name)
        self.commucator.logger_info('ScopeFileElement '+'{} scope log path: {}'.format(scope_name, self.log_path))
        
        self.log_name = log_name
        
        self.log_name_path = os.path.join(self.log_path, self.log_name) # 这个是用于wandb，tensorboard等的log文件
        self.commucator.logger_info('ScopeFileElement '+'{} scope log name path: {}'.format(scope_name, self.log_name_path))
        
        self.current_history_model_path_id = 0
        
        # 创建文件夹
        self.mkdir(self.cache_path)
        self.mkdir(self.history_model_path)
        self.mkdir(self.log_path)
        self.mkdir(self.log_name_path)
    
    def get_current_history_model_path(self):
        """
        获取当前模型的路径
        """
        return self.history_model_path

    def save_history_models(self, model_dict: dict):
        dir_path = self.generate_current_history_model_path()

        for key, value in model_dict.items():
            name = key + '.h5'
            file_path = os.path.join(dir_path, name)
            value.save_weights(file_path, overwrite=True)

    def save_cache_models(self, model_dict: dict):
        for key, value in model_dict.items():
            name = key + '.h5'
            file_path = os.path.join(self.cache_path, name)
            value.save_weights(file_path, overwrite=True)
    def generate_current_history_model_path(self):
        """
        生成新的模型路径
        """
        self.current_history_model_path_id += 1
        path = os.path.join(self.history_model_path, str(self.current_history_model_path_id))
        self.commucator.logger_info('ScopeFileElement: Generate new history model path: {}'.format(path))
        self.mkdir(path)
        self.current_history_model_id_path = path
        return path
    
    def mkdir(self, path: str):
        """
        创建文件夹

        :param path: 文件夹路径
        :return:
        """

        if self.commucator.process_id == 0:
            current_path = os.getcwd()
            path = os.path.join(current_path, path)
            self.commucator.logger_info('ScopeFileElement '+'Ready to create folder: {}'.format(path))
            if not os.path.exists(path):
                os.makedirs(path)
                self.commucator.logger_success('ScopeFileElement '+'Create folder: {}'.format(path))
            else:
                self.commucator.logger_warning('ScopeFileElement '+'Folder already exists: {}'.format(path))
        else:
            self.commucator.logger_info('ScopeFileElement '+'Process id is not 0, do not create folder: {}'.format(path))
        

# TODO:多进程中如何同步文件系统
class FileSystemBase(ABC):
    """
    
    FileSystem是EvoML的基础部件，告知所有线程模块东西在哪里，以及如何访问。
    
    EvoML会在集群的每个机器上创建相似的文件系统：
    
        -project name
        -cache
            -scope_name1
                cache_file
            -scope_name2
                cache_file
            ...
        -history_model
            -scope_name1
                -history_folder
                    -1
                        history_file
            -scope_name2
                -history_folder
                    history_file
            ...
        -log
            log_file
    
    """
    
    def __init__(self,
                 project_name:str,
                 data_module:DataModule,
                 communicator:CommunicatorBase,
                 ):
        
        """
        初始化文件系统
        
        Args:
            project_name: 项目名称
            data_module: 数据模块
            communicator: 通信模块
        """
        
        self._project_name = project_name
        self._communicator = communicator
        self._data_module = data_module
        self.scope_dict = {} # 用于存储scope的文件元素,方便被其他木块调用
        
        self._root_path = os.getcwd()
        self._communicator.logger_info('FileSystem '+'Root path: {}'.format(self._root_path))
        
        self._cache_root_path = os.path.join(self._project_name, 'cache')
        self._communicator.logger_info('FileSystem '+'Cache root path: {}'.format(self._cache_root_path))
        
        self._log_root_path = os.path.join(self._project_name, 'log')
        self._communicator.logger_info('FileSystem '+'Log root path: {}'.format(self._log_root_path))
        
        self._history_model_root_path = os.path.join(self._project_name, 'history_model')
        self._communicator.logger_info('FileSystem '+'History model root path: {}'.format(self._history_model_root_path))

        self._data_unit_path = os.path.join(self._project_name, 'data_unit')
        self._communicator.logger_info('FileSystem '+'Data unit root path: {}'.format(self._data_unit_path))

        
        # 创建文件夹
        # 在每一台机器中，我们指定进程0来创建文件夹
        # 其余进程知道文件夹已经存在，不再创建
        self.mkdir(self._cache_root_path)
        self.mkdir(self._log_root_path)
        self.mkdir(self._history_model_root_path)
        self.mkdir(self._data_unit_path)
        
       
        
    def add_scope(self, scope_name:str, log_name:str=None):
        """
        添加scope. 
        
        该函数返回一个info创建信息，用于创建一个DataUnit

        Args:
            scope_name (str): 算法名称
            log_name (str, optional): log名称. Defaults to None.
            
        Returns:
            dict: 创建信息。
        
        """
        
        self._communicator.logger_info('FileSystem '+'Add scope: {}'.format(scope_name))
         
        if log_name is None:
            log_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self._communicator.logger_warning('FileSystem '+'log_name is None, use current time as log_name: {}'.format(log_name))
        else:
            self._communicator.logger_info('FileSystem '+ 'Use log_name: {}'.format(log_name))
            
        self.scope_dict[scope_name] = ScopeFileElement(
            scope_name=scope_name,
            cache_root_path=self._cache_root_path,
            history_model_root_path=self._history_model_root_path,
            log_root_path=self._log_root_path,
            log_name=log_name,
            communicator=self._communicator
        )
        
        self.__setattr__(scope_name, self.scope_dict[scope_name])
        
        ret_dict = {
            'name': 'file_system_'+scope_name,
            'dtype': np.uint32,
            'shape': (1,),
            'size': 1,
        }
        
        return ret_dict
        
    def mkdir(self, path: str):
        """
        创建文件夹

        :param path: 文件夹路径
        :return:
        """
        
        if self._communicator.process_id == 0:
            current_path = os.getcwd()
            path = os.path.join(current_path, path)
            self._communicator.logger_info('FileSystem '+'Ready to create folder: {}'.format(path))
            if not os.path.exists(path):
                os.makedirs(path)
                self._communicator.logger_success('FileSystem '+'Create folder: {}'.format(path))
            else:
                self._communicator.logger_warning('FileSystem '+'Folder already exists: {}'.format(path))
                
        else:
            self._communicator.logger_info('FileSystem '+'Process id is not 0, do not create folder: {}'.format(path))
    
    @property
    def get_project_name(self):
        return self._project_name
    
    @property
    def get_cache_root_path(self):
        return self._cache_root_path
    
    @property
    def get_log_root_path(self):
        return self._log_root_path
    
    @property
    def get_history_model_root_path(self):
        return self._history_model_root_path
    
    ############################### 功能接口 ################################
    def get_scope_file_element(self, scope_name:str)->ScopeFileElement:
        """
        获取scope文件元素

        Args:
            scope_name (str): 算法名称

        Returns:
            ScopeFileElement: scope文件元素
        """
        return self.scope_dict[scope_name]

    def write_data_unit_yaml(self,unit_name, unit_info):
        """
        将data_unit的信息写入yaml文件
        方便其他进程或者程序读取

        Args:
            unit_name (存储单元名称): unit_info (存储单元信息)
            unit_info (单元信息): 单元信息
        """
        yaml_path = os.path.join(self._data_unit_path, unit_name+'.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(unit_info, f)
        self._communicator.logger_info('FileSystem '+'Write data unit yaml file: {}'.format(yaml_path))
    
    def read_data_unit_yaml(self, unit_name):
        """
        读取data_unit的信息

        Args:
            unit_name (str): 存储单元名称

        Returns:
            dict: 存储单元信息
        """
        yaml_path = os.path.join(self._data_unit_path, unit_name+'.yaml')
        with open(yaml_path, 'r') as f:
            unit_info = yaml.load(f, Loader=yaml.FullLoader)
        self._communicator.debug_info('FileSystem '+'Read data unit yaml file: {}'.format(yaml_path))
        return unit_info

        
        
class DefaultFileSystem(FileSystemBase):
    """
    默认的文件系统，只能在单机中进行。
    """
    
    def __init__(self,
                 project_name:str,
                 data_module:DataModule,
                 communicator:CommunicatorBase,
                 ):
        communicator.logger_info('DefaultFileSystem '+'Use DefaultFileSystem')
        super().__init__(project_name, data_module, communicator)
        
        # 提供一个特殊的数据管理器，专门用于记录当前训练轮次，存储节点等信息
        
        # self.file_data_module = FileDataModule('FileSystem')
