# 优先初始化
import sys
from loguru import logger  # logger
import os
import time
def mkdir(path: str):
    """
    创建文件夹

    :param path: 文件夹路径
    :return:
    """

    # current_path = os.getcwd()
    # path = os.path.join(current_path, path)
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info('Create folder: ' + path+'.')
    else:
        logger.info('Folder already exists using: ' + path+'.')

# 二级初始化
from AquaML.param.AquaParam import AquaParam
from AquaML.param.ParamBase import ParamBase

# 三级初始化，全局变量
settings = AquaParam() # 框架的全局设置

# 四级初始化
from AquaML.core.FileSystem import FileSystem
from AquaML.core.Communicator import Communicator


file_system = FileSystem() # 文件系统
communicator = Communicator() # 通信器

# 五级初始化
from AquaML.core.DataModule import DataModule
data_module = DataModule()

# 随机import
from AquaML.algo.ModelBase import ModelBase
from AquaML.core.DataInfo import DataInfo
from AquaML.core.Recorder import Recorder
from AquaML.tool import AquaTool


aqua_tool = AquaTool()
recorder = Recorder()


def init(
    hyper_params: ParamBase,
    root_path: str=None,
    memory_path: str=None,
    wandb_project: str=None,
    engine: str='tensorflow',
):
    """
    
    初始化AquaML框架。

    Args:
        hyper_params (ParamBase): 超参数.
        root_path (str, optional): 可以当作工作路径，后面的文件夹都会在这个路径下创建。默认为None。
        memory_path (str, optional): 共享内存创建路径。默认为None。
        wandb_project (str, optional): wandb的项目名称。默认为None。
        engine (str, optional): 计算引擎。默认为'tensorflow'。
    """
    
    ########################################################
    # 1. 配置logger输出
    ########################################################
    logger.remove()
    
    colorize = {
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold red on white",
        "SUCCESS": "green",
        "DEBUG": "blue",
        }
    logger_format = "<level> Rank" + str(communicator.rank) + " {time:YYYY-MM-DD HH:mm:ss:ms} | {level} | {name}:{function}:{line} - {message} </level>"
    logger.add(sys.stderr, format=logger_format, colorize=colorize)
    
    ########################################################
    
    ########################################################
    # 2. 设置Wandb
    ########################################################
        # # 使用默认的wandb设置     
    if settings.wandb_config is None:
        logger.info("Wandb config not set, use default config")
        hyper_params_dict = hyper_params.get_param_dict()
        
        # TODO:未来收集模型的学习率
        # 搜集模型的参数
        # model_params = {}
        
        # for model_name, model in model_dict.items():
        #     model_params[model_name+'_learning_rate'] = model.learning_rate
        
        wandb_config = {
            **hyper_params_dict,
            # **model_params
        }
        
    # 使用用户设置的wandb设置
    else:
        wandb_config = settings.wandb_config
    
    # run_name = hyper_params.env_args['env_name']
    # os.path.basename(__file__)[: -len(".py")
    run_name = f"{hyper_params.algo_name}__{hyper_params.wandb_other_name}__{int(time.time())}"

    recorder.init(wandb_project=wandb_project, config=wandb_config,run_name=run_name)
    
    
    ########################################################
    # 2. 获取工作路径
    ########################################################
    if root_path is None:
        logger.warning('using default root_path')
        
        # 获取当前路径
        current_path = os.getcwd()
        
        # 获取当前路径的所有文件夹
        folders = os.listdir(current_path)
        
        # 检测是否有Default文件夹
        default_ids = []
        for folder in folders:
            if 'Default' in folder:
                logger.info('detected Default in file name: ' + folder)
                Default_id = folder[7:]
                default_ids.append(eval(Default_id))
                
        if len(default_ids) == 0:
            
            root_path = os.path.join(current_path, 'Default0')
            logger.info('No Default folder detected, use ' + root_path + ' as root_path')
        else:
            default_ids.sort()
            Default_id = default_ids[-1] + 1
            
            logger.info('Detected Default folder, number of Default folders: ' + str(len(default_ids)))
            logger.info('Max Default folder id: ' + str(default_ids[-1]))
            
            root_path = os.path.join(current_path, 'Default' + str(Default_id))
            
            logger.info('Use ' + root_path + ' as root_path')
    
    else:
        logger.info('using user defined root_path: ' + root_path)
    
    settings.set_root_path(root_path)
        
    if communicator.rank == 0:
        # 创建文件夹
        file_system.init()
        
    communicator.Barrier()
    
    
    ########################################################
    # 3. 配置logger输出
    ########################################################
    logger_file_name = "rank_" + str(communicator.rank) + ".log"
    log_file = os.path.join(file_system.log_path, logger_file_name)
    logger.add(log_file, rotation="500 MB")  # 日志文件
        
    ########################################################
    # 4. 设置全局变量
    ########################################################
    if memory_path is None:
        # 使用Default进行命名，并且和root_path序号一致
        logger.info('using default memory_path')
        # 获取当前路径
        current_path = os.getcwd()
        
        # 获取当前路径的所有文件夹
        folders = os.listdir(current_path)
        
        # 检测是否有Default文件夹
        default_ids = []
        for folder in folders:
            if 'Default' in folder:
                logger.info('detected Default in file name: ' + folder)
                Default_id = folder[7:]
                default_ids.append(eval(Default_id))
        
        if len(default_ids) == 0:
            memory_path = 'Default0'
            logger.info('No Default folder detected, use ' + memory_path + ' as memory_path')
        else:
            default_ids.sort()
            Default_id = default_ids[-1] + 1
            
            logger.info('Detected Default folder, number of Default folders: ' + str(len(default_ids)))
            logger.info('Max Default folder id: ' + str(default_ids[-1]))
            
            memory_path = 'Default' + str(Default_id)
            
            logger.info('Use ' + memory_path + ' as memory_path')
    else:
        logger.info('using user defined memory_path: ' + memory_path)
    
    settings.set_memory_path(memory_path)
    
    data_module.init() # 初始化数据模块
    
    ########################################################
    # 5. Wandb名称配置
    ########################################################
    if wandb_project is None:
        wandb_project = 'AquaML'
        logger.info('using default wandb_project: ' + wandb_project)
    else:
        logger.info('using user defined wandb_project: ' + wandb_project)
    settings.set_wandb_project(wandb_project)
    
    ########################################################
    # 6. 获取该计算机的信息
    ########################################################
    
    import pynvml as nvml
    
    try:
        nvml.nvmlInit()
        GPU_device_count = nvml.nvmlDeviceGetCount()
        logger.info('Detected GPU device count: ' + str(GPU_device_count))
    except:
        logger.error('Failed to detect GPU device count, your computer may not have nvidia GPU or pynvml is not installed.')
        GPU_device_count = 0
    
    settings.set_GPU_num(GPU_device_count)
    
    ########################################################
    # 7. 设置计算引擎及其相关函数
    ########################################################
    logger.info('Using ' + engine + ' as compute engine')
    settings.set_engine(engine)
    aqua_tool.set_convert_numpy_fn(engine)
    
    # if engine == 'tensorflow':
    #     convert_numpy_fn = lambda x: x.numpy()
    # if engine == 'torch':
    #     # import torch
    #     convert_numpy_fn = lambda x: x.cpu().numpy()

 
    
        
    
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
    
    communicator.barrier()
    
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
    
    communicator.barrier()    
    

def config_data_lists(data_infos:tuple,
                        lists_name:str,
                        size:int,
                        ):
    """
    配置数据模块。
    自动在主进程中创建数据列表，子进程中检测数据列表是否存在。
    
    Args:
        data_infos (tuple): 数据信息。
        lists_name (str): 数据所在的算法区域。
        size (int): 需要多个重复数据列表。
    """
    
    # 主进程创建数据列表
    
    for data_info in data_infos:
        data_module.create_data_list_from_data_info(
            data_info=data_info,
            lists_name=lists_name,
            size=size,
        )
        
    


__all__ = ['sys','init','os','AquaParam','AquaTool','Recorder','time']
