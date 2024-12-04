'''
用于规定下一代框架的基本结构
'''

import abc
from AquaML import logger, communicator, settings, file_system, data_module, aqua_tool, mkdir
from AquaML.algo.AlgoBase import AlgoBase
import numpy as np
import os
import torch

class FrameWorkBase(abc.ABC):
    
    """
    框架基类，用于定义框架的基本功能。提供以下功能：
    1. 进程资源配置模块，用于更加进程信息。
    """
    
    def __init__(self):
        pass
    
    
    ########################################
    # 进程资源配置模块
    ########################################
    def config_task(self, task:dict):
        """
        配置任务。
        
        Args:
            task (dict): 任务字典。
        """
        
        def GPU_config(task_info:dict):
            """
            配置GPU资源。
            
            Args:
                task_info (dict): 任务信息。
            """
            GPU_enable = task_info['GPU_enabled']
            GPU_id = task_info['GPU_id']
            
            if GPU_enable == False:
                logger.success('GPU is not used in this task')
                os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
            elif settings.GPU_num == 0:
                logger.error('No GPU is available, progaram continue to run, but may cause key error.')
            # GPU_enable == 'AUTO' 有GPU资源，自动分配
            elif GPU_enable == 'AUTO':
                logger.info("GPU allocation mode: AUTO")
                if settings.GPU_num > 0:
                    if GPU_id == 'AUTO':
                        used_gpu_id = communicator.rank % settings.GPU_num
                        os.environ['CUDA_VISIBLE_DEVICES'] = str(used_gpu_id)
                        logger.info("Use GPU: {}".format(used_gpu_id))
                        if settings.engine == 'tensorflow':
                            self.tf_gpu_setting()
                else:
                    logger.warning("No GPU is available, progaram continue to run, but may cause key error.")
            else:
                logger.info("GPU allocation mode: MANUAL")
                
                if settings.GPU_num<0:
                    logger.error("Must have at least one GPU.")
                    logger.warning("Program continue to run, but may cause key error.")
                else:
                    if GPU_id == 'AUTO':
                        used_gpu_id = communicator.rank % settings.GPU_num
                        os.environ['CUDA_VISIBLE_DEVICES'] = str(used_gpu_id)
                        logger.info("Use GPU: {}".format(used_gpu_id))
                        if settings.engine == 'tensorflow':
                            self.tf_gpu_setting()
                    else:
                        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)
                        logger.info("Use GPU: {}".format(GPU_id))
                        if settings.engine == 'tensorflow':
                            self.tf_gpu_setting()
                            
        # 如果是torch,自动为每个进程选择合适的device
        if settings.engine == 'torch':
            import torch
            device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
            )
            
            settings.set_device(device)
        ########################################
        # 1. 0号进程检测当前资源是否满足任务需求
        ########################################
        if communicator.rank == 0:
            task_info = task['task_info']
            minimum_num_process = task_info['minimum_num_process']
            if communicator.size < minimum_num_process:
                logger.error("AquaML try to use {} process, but only {} process is available.".format(minimum_num_process, communicator.size))    
                logger.error("Program continue to run, but may cause key error.")
            
            GPU_required = task_info['GPU_required']
            if GPU_required == True:
                if settings.GPU_num == 0:
                    logger.error("GPU is required, but no GPU is available.")
                    logger.error("Program continue to run, but may cause key error.")
        
        communicator.Barrier()
        
        ########################################
        # 2. 根据任务信息对进程进行配置
        ########################################
        for process_task_name, process_task_info in task.items():
            if process_task_name == 'task_info':
                continue
            if process_task_info['process_id'] != communicator.rank:
                continue
            elif process_task_info['process_id'] == communicator.rank:
                logger.info("Task name: {}".format(process_task_name))
                GPU_config(process_task_info)
                break
            elif process_task_info['process_id'] == 'AUTO':
                logger.info("Task name: {}".format(process_task_name))
                GPU_config(process_task_info)
                break
            
            logger.warning("No task is set for process {}".format(communicator.rank))
                
    def tf_gpu_setting(self):
        """
        在使用tensorflow时，将GPU内存设置为按需分配。
        """
        
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    ########################################
    # 通用模块的配置
    ########################################
    def config_file_system(self, algo_name:str):
        """
        配置文件系统。
        
        为不同的算法配置不同的文件系统。
        
        Args:
            algo_name (str): 算法名称。
        """
        if communicator.rank == 0:
            file_system.config_algo(algo_name)
            logger.info("File system create algorithm folder: {}".format(algo_name))
        
        communicator.Barrier()
        
        if communicator.rank > 0:
            file_system.config_algo(algo_name)
            logger.info("File system detect algorithm folder: {}".format(algo_name))
        
        communicator.Barrier()
    
    
    def config_data_units(self,  
                         data_infos:tuple,
                         units_name:str,
                         size:int,
                         using_org_shape:bool=False
                            ):
        """
        配置数据模块。
        自动在主进程中创建数据单元，子进程中检测数据单元是否存在。
        

        Args:
            data_infos (tuple): 数据信息。
            units_name (str): 数据所在的算法区域。
            size (int): 需要多个重复数据单元。
        """
        
        # 主进程创建数据单元
        if communicator.rank == 0:
            logger.info("Create data unit in data module")
            
            for data_info in data_infos:
                data_module.create_data_unit_from_data_info(
                    data_info=data_info,
                    units_name=units_name,
                    size=size,
                    exist=False,
                    using_org_shape=using_org_shape
                )
        
        communicator.Barrier()
        
        # 子进程读取数据单元
        if communicator.rank > 0:
            logger.info("Read data unit in data module")
            
            for data_info in data_infos:
                data_module.create_data_unit_from_datainfo(
                    data_info=data_info,
                    units_name=units_name,
                    size=size,
                    exist=True
                )
        
        communicator.Barrier()
        

        
    ########################################
    # 模型存储与加载
    ########################################
    def save_cache_checkpoint(self,algo:AlgoBase,epoch:int):
        """
        保存缓存的检查点。
        """
        
        model_dict = algo.model_dict
        cache_path = file_system.query_cache(algo.name)
        
        for name, model in model_dict.items():
            aqua_tool.save_weights_fn(
                model=model,
                name=name,
                path=cache_path
            )
    
    def save_history_checkpoint(self,algo:AlgoBase,epoch:int):
        """
        保存历史的检查点。
        """
        
        model_dict = algo.model_dict
        history_path = file_system.query_history_model(algo.algo_name)
        current_path = os.path.join(history_path, str(epoch))
        mkdir(current_path)
        
        saved_dict = {}
        
        for name, model in model_dict.items():
            saved_dict[name] = model.state_dict()
        
        # 保存模型参数
        torch.save(saved_dict, os.path.join(current_path, 'model.pt'))
    
    def load_checkpoint(self,algo:AlgoBase,file_path:str):
        """
        加载检查点。
        
        args:
            algo (AlgoBase): 算法。
            file_path (str): 文件路径。
        """
        
        loaded_dict = torch.load(file_path)
        
        for name, model in algo.model_dict.items():
            model.load_state_dict(loaded_dict[name])
        
        
        
    def save_trajectory(self,algo:AlgoBase,epoch:int, trajectory:dict):
        """
        保存轨迹。
        """
        
        trajectory_root_path = file_system.query_trajectory(algo.algo_name)
        trajectory_path = os.path.join(trajectory_root_path, str(epoch))
        mkdir(trajectory_path)
        
        for name, data in trajectory.items():
            if not isinstance(data, np.ndarray):
                data = aqua_tool.convert_numpy_fn(data)
            np.save(os.path.join(trajectory_path, name+'.npy'), data)
        
    ########################################
    # 必须实现的接口
    ########################################
    @abc.abstractmethod
    def run(self):
        """
        运行框架。
        """
        pass