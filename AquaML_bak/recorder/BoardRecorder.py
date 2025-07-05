import tensorboardX.summary
from AquaML import logger
import tensorboardX

class BoardRecorder:
    """
    BoardRecorder用于记录训练过程中的数据。
    """
    def __init__(self,
                 log_file_name: str,
                 ):
        
        """
        创建一个tensorboardX的记录器。
        """
        
        self.writer = tensorboardX.SummaryWriter(log_file_name)
        logger.info('Recorder was initialized')
        
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
                key = f'{prefix}/{key}'
            self.writer.add_scalar(key, value, step)
            logger.info('Step: {}, {}: {}'.format(step, key, value))
