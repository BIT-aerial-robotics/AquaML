from AquaML import logger
from AquaML.param.ParamBase import ParamBase

class AquaParam:
    """
    AquaParam为AquaML提供全局参数，以及存储该机器的信息。
    
    memory_path用于识别该项目得内存地址。
    root_path用于识别该项目得文件地址。
    wandb_project识别Wandb。
    """
    def __init__(self):
        
        self._memory_path = None
        self._root_path = None
        self._wandb_project = None
        self._wandb_config = None # Wandb中训练参数配置
        
        # 设备信息参数
        self._GPU_num = None
        
        # 默认计算引擎
        self._engine = 'tensorflow'
        
        # 设置device信息，目前在torch中使用
        self._device = None
        
        # 强化学习框架参数设置
        self._env_num = None
        self._steps = None
        self._hyper_params:ParamBase = None
        
    def set_root_path(self, root_path:str):
        """
        设置root_path。
        
        Args:
            root_path (str): 文件地址。
        """
        self._root_path = root_path
        logger.info('root_path set: ' + root_path)
        
    def set_memory_path(self, memory_path:str):
        """
        设置memory_path。
        
        Args:
            memory_path (str): 内存地址。
        """
        self._memory_path = memory_path
        logger.info('memory_path set: ' + memory_path)
    
    def set_wandb_project(self, wandb_project:str):
        """
        设置wandb_project。
        
        Args:
            wandb_project (str): Wandb项目名称。
        """
        self._wandb_project = wandb_project
        logger.info('wandb_project set: ' + wandb_project)
    
    def set_wandb_config(self, wandb_config:dict):
        """
        设置wandb_config。
        
        Args:
            wandb_config (dict): Wandb配置。
        """
        self._wandb_config = wandb_config
        # logger.info('User defined wandb_config!')
        
    def set_GPU_num(self, GPU_num:int):
        """
        设置GPU_num。
        
        Args:
            GPU_num (int): GPU数量。
        """
        self._GPU_num = GPU_num
        logger.info('GPU_num set: ' + str(GPU_num))
        
    def set_engine(self, engine:str):
        """
        设置engine。
        
        Args:
            engine (str): 计算引擎。
        """
        self._engine = engine
        logger.info('compute engine set: ' + engine)
        
    def  set_env_num(self, env_num:int):
        """
        设置env_num。
        
        Args:
            env_num (int): 环境数量。
        """
        self._env_num = env_num
        logger.info('env_num set: ' + str(env_num))
    
    def set_steps(self, steps:int):
        """
        设置steps。
        
        Args:
            steps (int): 训练步数。
        """
        self._steps = steps
        logger.info('steps set: ' + str(steps))
        
    def set_device(self, device):
        """
        设置device。
        
        Args:
            device: 设备。
        """
        self._device = device
        logger.info('device set: {}'.format(device))
    
    @property
    def memory_path(self):
        if self._memory_path is None:
            logger.error('memory_path is None')
            return None
        return self._memory_path
    
    
    @property
    def root_path(self):
        if self._root_path is None:
            logger.error('file_path is None')
            return None
        return self._root_path
    
    @property
    def wandb_project(self):
        # if self._wandb_project is None:
        #     logger.error('wandb_project is None')
        #     return None
        return self._wandb_project
    
    @property
    def GPU_num(self):
        # if self._GPU_num is None:
        #     logger.error('GPU_num is None')
        #     return None
        return self._GPU_num
    
    @property
    def wandb_config(self):
        return self._wandb_config
    
    @property
    def engine(self):
    
        return self._engine
    
    @property
    def env_num(self):
        # if self._env_num is None:
        #     logger.error('env_num is None')
        #     return None
        return self._env_num
    
    @property
    def steps(self):
        # if self._steps is None:
        #     logger.error('steps is None')
        #     return None
        return self._steps
    
    @property
    def hyper_params(self):
        return self._hyper_params
    
    @property
    def device(self):
        return self._device