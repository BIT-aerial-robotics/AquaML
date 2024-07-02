'''
所有算法的基类。

这里面规定了所有算法的基本接口。
'''

import abc
from AquaML.communicator.CommunicatorBase import CommunicatorBase
from AquaML.core.old.DataModule import DataModule
import tensorflow.keras as keras
import numpy as np
from AquaML.core.old.FileSystem import FileSystemBase
from AquaML.core.Tool import LossTracker


class AlgoBase(abc.ABC):
    """
    Algo算法基类，用于定义Algo算法的基本功能。
    """

    def __init__(self,
                 name: str,
                 # buffer,
                 hyper_params,
                 communicator: CommunicatorBase,
                 data_module: DataModule,
                 file_system: FileSystemBase
                 ):
        """
        Algo算法基类，用于定义Algo算法的基本功能。

        Args:
            name(str): 算法的名称。
            communicator (CommunicatorBase): 通信模块。用于多进程通信以及log等。由系统自动传入。
            data_module (DataModule): 数据模块。用于获取数据的shape等信息。由系统自动传入。
            file_system (FileSystem): 文件系统。用于文件的存储和读取。由系统自动传入。
        """
        communicator.logger_info('AlgoBase Init AlgoBase')

        self._name = name
        self._hyper_params = hyper_params
        self._communicator = communicator
        self._data_module = data_module
        self._file_system = file_system
        self._buffer = None
        self._optimize_times = 0  # 优化次数

        # 一些工具的接口
        self._optimizer_pool = {}  # 优化器池子

        self._loss_tracker = LossTracker()

        self._file_system.add_scope(self._name)

    ########################功能区################################
    def initialize_network(self, model, expand_dims_idx=None):
        """
        初始化网络模型。

        Args:
            model: 网络模型。
            expand_dims_idx: 需要扩展维度的索引。默认为None。
        """

        input_data_names = model.input_names

        input_data = []

        for input_data_name in input_data_names:

            try:
                shape = self._data_module.get_unit_shape(input_data_name)
                self._communicator.logger_info('AlgoBase Get shape of input data: {}'.format(shape))
            except KeyError:
                self._communicator.logger_error('AlgoBase Can not get shape of input data: {}'.format(input_data_name))
                raise RuntimeError('Can not get shape of input data: {}'.format(input_data_name))

            shape_ = (1,*shape)
            data = np.empty(shape_, dtype=np.float32)

            input_data.append(data)

        model(*input_data)

        self._communicator.logger_info('AlgoBase Initialize network successfully.')

    def copy_weights(self, source_model, target_model):
        """
        复制网络模型的权重。

        Args:
            source_model: 源网络模型。
            target_model: 目标网络模型。
        """
        source_weights = source_model.get_weights()
        target_model.set_weights(source_weights)

        self._communicator.logger_success('AlgoBase Copy weights successfully.')

    def soft_update_weights(self, source_model, target_model, tau):
        """
        软更新网络模型的权重。

        Args:
            source_model: 源网络模型。
            target_model: 目标网络模型。
            tau: 更新速率。
        """
        source_weights = source_model.get_weights()
        target_weights = target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = (1 - tau) * target_weights[i] + tau * source_weights[i]

        target_model.set_weights(target_weights)

        self._communicator.logger_info('AlgoBase Soft update weights successfully.')

    def create_optimizer(self, optimizer_info: dict, name: str):
        """
        创建优化器。
        
        该函数会通过模型的名称自动创建一个优化器，优化器的名称为name+'_optimizer'。
        可以支持所有的keras优化器。
        
        optimizer_info格式：
        {
            'type': 'adam',
            'args': {
                'learning_rate': 0.001
            }
        }

        Args:
            optimizer_info (dict): 优化器信息。
            name (str): 模型的名称。
        """
        optimizer_type = optimizer_info.get('type')
        args = optimizer_info.get('args')

        if not optimizer_type or not args:
            raise ValueError('optimizer_info must contain "type" and "args".')

        try:
            optimizer = getattr(keras.optimizers, optimizer_type)(**args)
            self._communicator.logger_success('AlgoBase  Create optimizer successfully.')
        except AttributeError:
            self._communicator.logger_error('AlgoBase Can not create optimizer of type {optimizer_type}.')
            raise RuntimeError(f'Can not create optimizer of type {optimizer_type}.')

        setattr(self, name + '_optimizer', optimizer)

        self._optimizer_pool[name] = getattr(self, name + '_optimizer')

    def save_checkpoint(self, model_dict: dict):
        scope_file_element = self._file_system.get_scope_file_element(self._name)
        scope_file_element.save_history_models(model_dict)
        self._data_module.history_number_dict[self._name][0][0] += 1
        self._communicator.logger_info('AlgoBase: {} update history_number_dict {}.'.format(
            self._name, self._data_module.history_number_dict[self._name].get_data()))


    def save_cache_models(self, model_dict: dict):
        scope_file_element = self._file_system.get_scope_file_element(self._name)
        scope_file_element.save_cache_models(model_dict)

    @abc.abstractmethod
    def init(self, buffer):
        """
        初始化网络模型。
        """

    @abc.abstractmethod
    def optimize(self, buffer) -> LossTracker:
        """
        优化网络模型。

        Args:
            buffer: 数据池。
        """
