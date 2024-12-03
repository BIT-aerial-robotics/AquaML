import abc

class RecorderBase:
    
    @abc.abstractmethod
    def record_scalar(self, data_dict: dict, prefix=None, step: int = None):
        """
        记录标量数据。
        
        Args:
            data_dict (dict): 数据字典。
            prefix (str, optional): 数据前缀，用于区分不同的数据来源。 Defaults to None.
            step (int, optional): 记录的步数。 Defaults to None.
        """