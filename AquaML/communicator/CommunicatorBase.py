'''
该部分具有以下功能：

1. 进程模拟器，用于debug模式下的模拟器，后续逐步完善。
2. 单机多进程接口。
3. 多机多进程接口。
4. 通信层，将兼容多种通信方式，包括共享内存。
5. 负责计算资源分配。
6. 负责分布式logger

'''

import abc
import os
from pynvml import *
from loguru import logger
import yaml
import copy


class ProcessTask:
    # TODO: 这部分要升级功能，太乱了。
    '''
    记录每个进程的功能。
    
    '''

    def __init__(self, process_id: int, ) -> None:
        self._process_id = process_id
        self._task_info = {
            'minimum_num_process': None,  # 最小进程数量
            'required_num_process': None,  # 需要的进程数量
            'GPU_required': False,  # 是否需要GPU
        }

        self._process_task_name = None  # 进程任务名称
        self._process_task_info = {}  # 进程任务信息
        # self._config_dict = {}

        self.each_process_task = {}
        self.each_id_process_task = []  # 按照id存储进程任务

    def set_task_info(self, task_info: dict):
        '''
        设置任务信息。
        
        '''
        self._task_info.update(task_info)

    def get_task_info(self):
        '''
        获取任务信息。
        
        '''
        return self._task_info

    def set_process_task(self, process_task_name: str, process_task_info: dict):
        '''
        设置进程任务。
        
        '''
        self._process_task_name = process_task_name
        self._process_task_info = process_task_info

    # def add_process_task(self, task_name:str, process_id:int):
    #     '''
    #     添加一个进程任务。

    #     '''

    #     self.__setattr__(task_name, process_id)

    # def get_process_task(self, task_name:str):
    #     '''
    #     获取一个进程任务。

    #     '''

    #     return self.__getattribute__(task_name)

    # def get_task_process(self, process_id:int):
    #     '''
    #     获取一个进程任务。

    #     '''

    #     for task_name in self.__dict__.keys():
    #         if self.__getattribute__(task_name) == process_id:
    #             return task_name

    # def serch_task(self, task_name:str):
    #     '''
    #     查找一个任务。并且返回任务的信息。

    #     '''

    #     return self.__getattribute__(task_name)


class CommunicatorBase(abc.ABC):

    def __init__(self,
                 comunicator_path: str = None,
                 machine_id=0,
                 compute_engine: str = 'tensorflow',
                 wait_time_out = 1,  # 等待超时时间
                 check_time_interval = 0.001,  # 检查时间间隔
                 debug_mode=False,
                 ):
        """
        用于初始化CommunicatorBase。
        
        args:
            # process_id (int): 线程ID，用于区分不同的进程。
            comunicator_path (str): 通讯器的工作路径。
            machine_id(int): 机器的ID。用于区分不同的机器。
            compute_engine(str): 计算引擎。默认为tensorflow。当前支持tensorflow、pytorch以及JAX。
            debug_mode(bool): 是否启用debug模式。默认为False。
        """
        ############################## 优先初始化属性 ##############################
        # 信息字典
        self._device_info = {}
        self._cluster_info = {}  # 存储当前节点信息
        ############################## 共有属性 ##############################
        # self._process_id = process_id
        self.logger = logger  # 日志模块
        self._machine_id = machine_id

        self._set_process_id()  # 设置进程ID

        self._compute_engine = compute_engine  # 计算引擎

        if comunicator_path is None:
            comunicator_path = os.path.join(os.getcwd(), 'comunicator')
            self.logger_warning('comunicator_path is None, use default path {}'.format(comunicator_path))
        else:
            self.logger_info('Use comunicator_path {}'.format(comunicator_path))

        self._comunicator_path = comunicator_path  # 通讯层工作目录
        self._log_path = os.path.join(comunicator_path, 'logs')  # 日志路径
        self._info_path = os.path.join(comunicator_path, 'info')  # 信息路径
        self._process_task = ProcessTask(self._process_id)  # 进程任务
        self._debug_mode = debug_mode  # 是否启用debug模式

        logger_file = os.path.join(self._log_path, 'machine_' + str(self._machine_id) + '_process_' + str(
            self._process_id) + '.log')  # 日志文件
        self._logger_file = self.logger.add(logger_file, rotation="500 MB")  # 日志文件

        # debug模式
        self._force_run = False  # 是否强制运行

        # 多进程同步参数
        self.wait_time_out = wait_time_out  # 等待超时时间
        self.check_time_interval = check_time_interval  # 检查时间间隔

        # 录入信息
        self._cluster_info['compute_engine'] = self._compute_engine
        self._cluster_info['machine_id'] = self._machine_id
        self._cluster_info['process_id'] = self._process_id
        self._cluster_info['GPU_enable'] = False  # 默认不启用GPU
        self._cluster_info['GPU_id'] = None  # 默认不启用GPU
        self._set_total_process_num()  # 设置总进程数量

        ############################## 进程配置模块 ##############################
        self.get_device_info()

    ############################## 硬件控制模块 ##############################

    def set_GPU(self, gpu_id: int, memory_growth=True):
        """
        用于分配GPU。
        
        Args:
            gpu_id (int): GPU的ID。
            memory_growth (bool): 是否启用内存增长。默认为True。
        """

        # 获取本机的GPU数量

        GPU_num = self._device_info['GPU']['GPU_num']

        if gpu_id > GPU_num:
            self.logger.error(
                'Machine {} has {} GPU, but you want to use {} GPU'.format(self._machine_id, GPU_num, gpu_id))

            # 退出程序
            if self._force_run:
                self.logger_warning('Force run mode is on. Continue to run the program')
            else:
                exit()
        else:
            # 分配GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            self.logger.info(
                'Machine {} process {} use GPU {}'.format(self._machine_id, self._process_id, self._process_id))

            self._cluster_info['GPU_enable'] = True
            self._cluster_info['GPU_id'] = gpu_id

            if memory_growth:
                if self._compute_engine == 'tensorflow':
                    import tensorflow as tf
                    gpus = tf.config.experimental.list_physical_devices('GPU')
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)

                # self.logger.info('Machine {} process {} use GPU {} with memory growth'.format(self._machine_id, self._process_id, self._process_id))

                self.logger_info('Use GPU {} with memory growth'.format(gpu_id))

    def get_device_info(self, out_info=True):
        """
        用于获取设备信息。
        
        一般来说运行在数据节点上即可。
        """
        nvida_flag = False
        # 获取GPU信息
        try:
            nvmlInit()
            deviceCount = nvmlDeviceGetCount()
            nvida_flag = True
        except:
            self.logger_error('Nvidia driver is not installed')
            deviceCount = 0


        if deviceCount == 0:
            self.logger_warning('No Nvidia GPU found')
        else:
            self.logger_info('Find {} GPU'.format(deviceCount))

            GPUs_info = {
                'GPU_num': deviceCount,
            }

            for i in range(deviceCount):
                handle = nvmlDeviceGetHandleByIndex(i)
                info = nvmlDeviceGetMemoryInfo(handle)
                GPUs_info['GPU_' + str(i)] = {
                    'name': nvmlDeviceGetName(handle),
                    'total_memory': info.total,
                    'free_memory': info.free,
                    'used_memory': info.used,
                }
                self.logger.info('GPU_{}: name: {}, total_memory: {}, free_memory: {}, used_memory: {}'.format(i,
                                                                                                               nvmlDeviceGetName(
                                                                                                                   handle),
                                                                                                               info.total,
                                                                                                               info.free,
                                                                                                               info.used))

            self._device_info['GPU'] = GPUs_info

        if nvida_flag:
            nvmlShutdown()

        # with open(os.path.join(self._info_path, 'machine_'+str(self._machine_id)+'_process_'+str(self._process_id)+'_device_info.yaml'), 'w') as f:
        #     yaml.dump(self._device_info, f)

        if out_info:
            with open(os.path.join(self._comunicator_path, 'device_info.yaml'), 'w') as f:
                yaml.dump(self._device_info, f)

            # 记录节点输出信息

            # self.logger.info('Machine {} process {} output device info in {}'.format(self._machine_id, self._process_id, os.path.join(self._comunicator_path, 'device_info.yaml')))

            self.logger_info(
                'Output device info in {}'.format(os.path.join(self._comunicator_path, 'device_info.yaml')))

    ############################## 日志模块 ##############################
    def logger_warning(self, sentence: str):
        """
        用于记录警告。
        
        默认的警告语句格式为：Machine {} process {}: sentence
        
        args:
            sentence (str): 警告语句。
        """
        self.logger.warning('Machine {} process {}: {}'.format(self._machine_id, self._process_id, sentence))

    def logger_error(self, sentence: str):
        """
        用于记录错误。
        
        默认的警告语句格式为：Machine {} process {}: sentence
        
        args:
            sentence (str): 错误语句。
        """

        self.logger.error('Machine {} process {}: {}'.format(self._machine_id, self._process_id, sentence))

    def logger_info(self, sentence: str):
        """
        用于记录信息。
        
        默认的警告语句格式为：Machine {} process {}: sentence
        
        args:
            sentence (str): 信息语句。
        """

        self.logger.info('Machine {} process {}: {}'.format(self._machine_id, self._process_id, sentence))

    def logger_debug(self, sentence: str):
        """
        用于记录debug信息。
        
        默认的警告语句格式为：Machine {} process {}: sentence
        
        args:
            sentence (str): debug信息。
        """

        self.logger.debug('Machine {} process {}: {}'.format(self._machine_id, self._process_id, sentence))

    def logger_critical(self, sentence: str):
        """
        用于记录critical信息。
        
        默认的警告语句格式为：Machine {} process {}: sentence
        
        args:
            sentence (str): critical信息。
        """

        self.logger.critical('Machine {} process {}: {}'.format(self._machine_id, self._process_id, sentence))

    def logger_exception(self, sentence: str):
        """
        用于记录exception信息。
        
        默认的警告语句格式为：Machine {} process {}: sentence
        
        args:
            sentence (str): exception信息。
        """

        self.logger.exception('Machine {} process {}: {}'.format(self._machine_id, self._process_id, sentence))

    def logger_success(self, sentence: str):
        """
        用于记录success信息。
        
        默认的警告语句格式为：Machine {} process {}: sentence
        
        args:
            sentence (str): success信息。
        """

        self.logger.success('Machine {} process {}: {}'.format(self._machine_id, self._process_id, sentence))

    def debug_info(self, sentence: str):

        if self._debug_mode:
            self.logger_info(sentence)

    def debug_warning(self, sentence: str):

        if self._debug_mode:
            self.logger_warning(sentence)

    def debug_error(self, sentence: str):

        if self._debug_mode:
            self.logger_error(sentence)

    def debug_debug(self, sentence: str):

        if self._debug_mode:
            self.logger_debug(sentence)

    def debug_critical(self, sentence: str):

        if self._debug_mode:
            self.logger_critical(sentence)

    def debug_exception(self, sentence: str):

        if self._debug_mode:
            self.logger_exception(sentence)

    def debug_success(self, sentence: str):

        if self._debug_mode:
            self.logger_success(sentence)

            ############################## 进程信息 ##############################

    # 读取进程任务信息
    def config_process_task_yaml(self, yaml_file: str):
        '''
        用于配置进程任务。
        
        args:
            yaml_file (str): yaml文件路径。
        '''

        # 读取进程分配信息
        self.logger_info('Read process task from {}'.format(yaml_file))

        with open(yaml_file, 'r') as f:
            process_task = yaml.load(f, Loader=yaml.FullLoader)

        # 检测当前可用资源是否能够满足最小资源需求
        self.logger_info('Check minimum resource requirement')
        process_task_info = process_task['task_info']  # 解析任务资源需求
        self._process_task.set_task_info(process_task_info)

        minimum_num_process = process_task_info['minimum_num_process']
        avaliable_num_process = self._cluster_info['total_process_num']

        if minimum_num_process > avaliable_num_process:
            self.logger_error(
                'Minimum process number is {}, but avaliable process number is {}'.format(minimum_num_process,
                                                                                          avaliable_num_process))

            if self._force_run:
                self.logger_warning('Force run mode is on. Continue to run the program')
            else:
                exit()
        else:
            self.logger_success(
                'Minimum process number is {}, avaliable process number is {}'.format(minimum_num_process,
                                                                                      avaliable_num_process))

        if process_task_info['required_num_process'] == 'AUTO':
            self.logger_info('Required process number is AUTO')

        self.logger_success('Pre-check process task success')

        # 读取进程任务,并且按照需求设置资源

        for process_task_name, process_task_info_e in process_task.items():
            if process_task_name == 'task_info':
                continue

            # 优先设置已经确定进程
            if process_task_info_e['process_id'] == self._process_id or process_task_info_e['process_id'] == 'AUTO':
                self.logger_info('Set process task {}'.format(process_task_name))
                self._process_task.set_process_task(process_task_name, process_task_info_e)  # 设置进程任务

                # GPU处于AUTO时候，当GPU可用时候，自动分配GPU
                GPU_enable = process_task_info_e['GPU_enabled']
                GPU_id = process_task_info_e['GPU_id']

                local_GPU_num = self._device_info['GPU']['GPU_num']

                if GPU_enable == False:
                    self.logger_success('GPU is not used in this task')
                    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

                elif GPU_enable == 'AUTO' and local_GPU_num > 0:
                    self.logger_success('GPU is AUTO and GPU is available')
                    if GPU_id == 'AUTO':
                        used_gpu_id = self._process_id % local_GPU_num
                        self.set_GPU(used_gpu_id)
                    else:
                        self.set_GPU(GPU_id)
                elif GPU_enable == 'AUTO' and local_GPU_num == 0:
                    self.logger_warning('GPU is AUTO but GPU is not available')
                elif GPU_enable == True:
                    if GPU_id == 'AUTO':
                        used_gpu_id = self._process_id % local_GPU_num
                        self.set_GPU(used_gpu_id)
                    else:
                        self.set_GPU(GPU_id)

                break  # 退出循环

            elif process_task_info_e['process_id'] == 'AUTO':
                # 剩下的进程，任务设置为AUTO
                self.logger_info('Set AUTO process task {}'.format(process_task_name))

                GPU_enable = process_task_info_e['GPU_enabled']
                GPU_id = process_task_info_e['GPU_id']

                local_GPU_num = self._device_info['GPU']['GPU_num']

                if GPU_enable == 'AUTO' and local_GPU_num > 0:
                    self.logger_info('GPU is AUTO and GPU is avaliable')
                    if GPU_id == 'AUTO':
                        used_gpu_id = self._process_id % local_GPU_num
                        self.set_GPU(used_gpu_id)
                    else:
                        self.set_GPU(GPU_id)
                elif GPU_enable == 'AUTO' and local_GPU_num == 0:
                    self.logger_warning('GPU is AUTO but GPU is not avaliable')
                elif GPU_enable == True:
                    if GPU_id == 'AUTO':
                        used_gpu_id = self._process_id % local_GPU_num
                        self.set_GPU(used_gpu_id)
                    else:
                        self.set_GPU(GPU_id)

        # 用一定的规则将所有进程任务存储在_process_task中
        # 直接存储在_process_task中

        for process_task_name, process_task_info_e in process_task.items():
            if process_task_name == 'task_info':
                continue

            self._process_task.each_process_task[process_task_name] = process_task_info_e  # 存储每个进程任务

        # 找出AUTO的进程任务

        AUTO_task = {}
        for process_task_name, process_task_info_e in process_task.items():
            if process_task_name == 'task_info':
                continue

            if process_task_info_e['process_id'] == 'AUTO':
                AUTO_task = process_task_info_e

        # 根据process id创建任务表
        for i in range(self._cluster_info['total_process_num']):
            for process_task_name, process_task_info_e in process_task.items():
                if process_task_name == 'task_info':
                    continue

                if process_task_info_e['process_id'] == i:
                    self._process_task.each_id_process_task.append(process_task_info_e)
                    break

                # 如果没有找到对应的进程任务，那么使用process id为AUTO的进程任务
                self._process_task.each_id_process_task.append(copy.deepcopy(AUTO_task))

        self.logger_success('Set process task success')

    ############################## 通用实现接口 ##############################
    @abc.abstractmethod
    def get_total_process_num(self):
        """
        获取总进程数量。
        
        Returns: int
        """
        pass

    def _set_total_process_num(self):
        """
        设置总进程数量。
        
        将信息设置入_cluster_info中。
        """
        self.logger_info('Aquire total process number')
        self._cluster_info['total_process_num'] = self.get_total_process_num()
        self.logger_info('Total process number is {}'.format(self._cluster_info['total_process_num']))

    @abc.abstractmethod
    def get_process_id(self):
        """
        获取进程ID。
        需要实现如何获取进程ID，进程ID如何分配。
        
        Returns: int
        """
        pass

    def _set_process_id(self):
        """
        设置进程ID。
        
        将信息设置入_cluster_info中。
        """
        self._cluster_info['process_id'] = self.get_process_id()
        self._process_id = self._cluster_info['process_id']

        self.logger_info('Aquire process id')

        self.logger_info('Process id is {}'.format(self._cluster_info['process_id']))

    ############################## 同步进程接口 ##############################

    @abc.abstractmethod
    def barrier(self):
        """
        用于同步进程。
        """
        pass

    ############################## 函数运行辅助接口 ##############################
    def run_function(self, function, process_id: int, args_dict: dict = {}, args_list: list = []):
        """
        该函数可以指定进程运行函数。
        
        args:
            function (function): 函数。
            process_id (int): 进程ID。
            args_dict (dict): 参数字典。
            args_list (list): 参数列表。
        """

        self.debug_info('Run function {} in process {}'.format(function.__name__, process_id))

        if args_dict != {}:
            return function(**args_dict)
        elif args_list != []:
            return function(*args_list)
        else:
            return function()

    ############################## property ##############################
    @property
    def process_id(self):
        return self._process_id

    @property
    def total_process_num(self):
        return self._cluster_info['total_process_num']

    @property
    def force_run(self):
        return self._force_run
