'''

交互策略的运行机制由policy给出，一个policy包含keras model，和输出采样插件，插件具体接口定义将在EvoML中全面定义。

'''

import tensorflow.keras as keras
import abc
from multiprocessing import shared_memory
from AquaML.core.Protocol import data_unit_info_protocol
from AquaML.core.old.DataModule import DataModule
from AquaML.core.old.FileSystem import FileSystemBase
from AquaML.communicator.CommunicatorBase import CommunicatorBase
import os
import numpy as np


# def get_data_unit_info(name:str):
#     """
#     获取数据单元的基本信息
    
#     Get the basic information of the data unit
#     """
#     shm_list = shared_memory.ShareableList(name = name+'_info')
#     info_list = list(shm_list)
    
#     dic = zip(data_unit_info_protocol, info_list)
    
#     # 释放共享内存
#     shm_list.close()
#     shm_list.unlink()
    
#     return dic

class PolicyBase(abc.ABC):
    
    def __init__(self, 
                 name_scope:str, 
                 candidate_action_id:int,
                 data_module:DataModule,
                 file_system:FileSystemBase,
                 communicator:CommunicatorBase,
                 ):
        """
        初始化策略基类,这里规定了（候选）策略接口

        Args:
            name_scope (str): 策略名称。
            candidate_action_id(int): 候选策略在数据中排序.
            data_module (DataModule): 数据模块。用于获取数据的shape等信息。由系统自动传入。
            file_system (FileSystemBase): 文件系统。用于文件的存储和读取。由系统自动传入。
            communicator (CommunicatorBase): 通信模块。用于多进程通信以及log等。由系统自动传入。
        """
        
        communicator.logger_info('PolicyBase'+'Init PolicyBase')
        
        self._candidate_action_id = candidate_action_id
        self._communicator = communicator
        self._file_system = file_system
        
        self._model_dict = {}
        self._model_class_dict = {} # 用于保存模型类，创建模型的备份
        # self._model_state_dict = {} # 用于保存模型状态，用于恢复模型
        self._model_name_table = []
        
        self._name_scope = name_scope
        self._data_module = data_module
        self._input_names = None
    
    
    ######################################## 功能接口 ########################################
    def initialize_model(self, keras_model:keras.Model):
        
        input_names = keras_model.input_names
        
        # 获取模型输入的shape
        
        # data_unit_info_dict = get_data_unit_info(input_name)
        
        input_data = []
        
        for input_name in input_names:
            data_shape = self._data_module.get_unit_shape(input_name)
            # input_data.append(np.zeros(data_shape))
            shape_ = (1, *data_shape)
            data = np.empty(shape_, dtype=np.float32)
            input_data.append(data)
        
        # 初始化模型
        keras_model(*input_data)
        
        self._communicator.logger_info('PolicyBase'+'Initialize model.')
    
    # def add_keras_model(self, name:str, keras_model_class, initialize:bool=True):
    #     """
    #     添加keras模型
        
    #     Add keras model, 访问的时候不需要额外访问，Policy.name即可
    #     """
    #     # model_name = self._name_scope + '_' + name
        
    #     self._model_class_dict[name] = keras_model_class
    #     self._model_name_table.append(name)
        
    #     if initialize:
    #         self.initialize_model(self._model_dict[name])
        
        
    #     # set attribute
  
    #     self._model_dict[name] = keras_model_class()
    #     self.__setattr__(name, self._model_dict[name])
        
    #     self.initialize_model(self._model_dict[name])
    
    def copy_weight(self,source_model:keras.Model, target_model:keras.Model):
        """
        拷贝模型权重
        
        Copy model weight
        """
        
        new_weights = []
        
        for idx, weight in enumerate(source_model.get_weights()):
            new_weights.append(weight)
        # source_weights = source_model.get_weights()
        target_model.set_weights(new_weights)
        
        self._communicator.logger_info('PolicyBase','Copy weight.')
        # keras.Model.weights
    
    def save_weight(self, weight_path:str, model_name:str=None):
        """
        保存模型权重.

        Args:
            weight_path (str): 权重保存路径.
            model_name (str, optional): 需要保存的模型，如果为None默认保存所有模型。
        """
        
        if model_name is None:
            for name in self._model_name_table:
                self._model_dict[name].save_weights(os.path.join(weight_path, name+'.weight.h5'))
                
                self._communicator.logger_info('PolicyBase: Save weight of model: '+name)
        else:
            self._model_dict[model_name].save_weights(os.path.join(weight_path, model_name+'.weight.h5'))
            
            self._communicator.logger_info('PolicyBase: Save weight of model: '+model_name)
    
    def load_weight(self, weight_path:str, model_name:str=None):
        """
        加载模型权重.

        Args:
            weight_path (str): 权重保存路径.
            model_name (str, optional): 需要加载的模型，如果为None默认加载所有模型。
        """
        
        if model_name is None:
            for name in self._model_name_table:
                self._model_dict[name].load_weights(os.path.join(weight_path, name+'.weight.h5'))
                self._communicator.logger_info('PolicyBase: Load weight of model: '+name)
        else:
            self._model_dict[model_name].load_weights(os.path.join(weight_path, model_name+'.weight.h5'))
            self._communicator.logger_info('PolicyBase: Load weight of model: '+model_name)
            
    @abc.abstractmethod
    def run(self):
        """
        运行策略
        
        Run policy
        """
        pass
    
    def send_candidate_action(self, action:np.ndarray):
        """
        将候选动作发送给指定的数据单元，指定位置。

        Args:
            action (np.ndarray): 由当前策略产生的动作.
        """
        
        self._data_module.candidate_action[self._candidate_action_id] = action
        
    def rec_state(self)->list:
        """
        用于接收状态。
        
        并且返回该策略所需要的输，以list返回。

        """
        
        input_datas = []
        for input_name in self._input_names:
            data = self._data_module.robot_state_dict[input_name].get_data()
            input_datas.append(data)
        
        return input_datas