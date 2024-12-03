import wandb
from AquaML import logger



class WandbRecorder:
    """
    WandbRecorder用于记录训练过程中的数据。
    """
    def __init__(self, 
                #  wandb_project: str,
                #  config: dict = {}
                 ):
        
        """
        Recorder作为记录器，用于记录训练过程中的数据。
        一般来说只有在训练进程里面才会调用Recorder。
        
     
        """
        
        # self.wandb_project = wandb_project
        # self.config = config
        
        logger.info('Recorder was initialized')
        
        
        # self._wandb = wandb
        # self._wandb.init(
        #     project=wandb_project,
        #     config=config
        # )
        
        # logger.success('Recorder initialized')
        
    def init(self, wandb_project: str, run_name:str,config: dict = {}):
        """
        初始化wandb。
        """
        
        self._wandb = wandb
        self.config = config
        self.wandb_project = wandb_project
        
        self._wandb.init(
            project=self.wandb_project,
            config=self.config,
            name=run_name
        )
        
        logger.success('Recorder initialized')
    
    
    def record_scalar(self, data_dict: dict, prefix=None, step: int = None):
        """
        记录标量数据。
        
        Args:
            data_dict (dict): 数据字典。
            prefix (str, optional): 数据前缀，用于区分不同的数据来源。 Defaults to None.
            step (int, optional): 记录的步数。 Defaults to None.
        """
        
        for key, value in data_dict.items():
            if prefix is not None:
                key = prefix + '/' + key
            self._wandb.log({key: value}, step=step)
            logger.info('Step: {}, {}: {}'.format(step, key, value))
                