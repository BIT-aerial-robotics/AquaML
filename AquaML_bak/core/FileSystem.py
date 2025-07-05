'''
逐渐将使用该模块的FileSystem。
这里将简化FileSystem的设计，对文件系统作进一步优化，和明确的规定。
与之类似功能的有DataModule用于管理内存模块。

上一代版本系统空间管理过于复杂，不易维护。
'''

from AquaML import logger,mkdir,settings
import os
import yaml


class FileSystem:
    """
    FileSystem用于管理文件系统。
    
    每个任务的文件系统将如下所示：
    - file_path
        - cache
        - data_unit
        - history_model
        - log
    """
    
    def __init__(self):
        
        self._root_path = None
        self._cache_path = None
        self._data_unit_path = None
        self._history_model_path = None
        self._log_path = None

        
    def init(self):
        # 创建文件夹
        self._root_path = settings.root_path
        self._cache_path = os.path.join(self._root_path, 'cache')
        self._data_unit_path = os.path.join(self._root_path, 'data_unit')
        self._history_model_path = os.path.join(self._root_path, 'history_model')
        self._log_path = os.path.join(self._root_path, 'log')
        mkdir(self._root_path)
        mkdir(self._cache_path)
        mkdir(self._data_unit_path)
        mkdir(self._history_model_path)
        mkdir(self._log_path)
    
    def config_algo(self, algo_name: str):
        """
        配置算法文件夹。
        
        Args:
            algo_name (str): 算法名称。
        """
        
        self.query_cache(algo_name)
        self.query_data_unit(algo_name)
        self.query_history_model(algo_name)
        self.query_log(algo_name)
        
    def query_cache(self, name: str)-> str:
        """
        查询算法对应的缓存文件夹位置，如果不存在则创建。
        
        返回文件夹位置。
        
        Args:
            name (str): 文件名。
        Returns:
            str: 文件夹位置。
        """
        
        path = os.path.join(self._cache_path, name)
        if not os.path.exists(path):
            mkdir(path)
            logger.info('Create folder: ' + self._cache_path+'.')
        
        return path
    
    def query_data_unit(self, name: str)-> str:
        """
        查询数据单元对应的文件夹位置，如果不存在则创建。
        
        返回文件夹位置。
        
        Args:
            name (str): 文件名。
        Returns:
            str: 文件夹位置。
        """
        
        path = os.path.join(self._data_unit_path, name)
        if not os.path.exists(path):
            mkdir(path)
            logger.info('Create folder: ' + self._data_unit_path+'.')
        
        return path
    
    def query_history_model(self, name: str)-> str:
        """
        查询历史模型对应的文件夹位置，如果不存在则创建。
        
        返回文件夹位置。
        
        Args:
            name (str): 文件名。
        Returns:
            str: 文件夹位置。
        """
        
        path = os.path.join(self._history_model_path, name)
        if not os.path.exists(path):
            mkdir(path)
            logger.info('Create folder: ' + self._history_model_path+'.')
        
        return path
    
    def query_log(self, name: str)-> str:
        """
        查询日志对应的文件夹位置，如果不存在则创建。
        
        返回文件夹位置。
        
        Args:
            name (str): 文件名。
        Returns:
            str: 文件夹位置。
        """
        
        path = os.path.join(self._log_path, name)
        if not os.path.exists(path):
            mkdir(path)
            logger.info('Create folder: ' + self._log_path+'.')
        
        return path
    
    def query_trajectory(self, name: str)-> str:
        """
        查询轨迹对应的文件夹位置，如果不存在则创建。
        一般用于存储测试轨迹。

        Args:
            name (str): 算法名称。

        Returns:
            str: 文件夹位置。
        """
        
        path = os.path.join(self._root_path, 'trajectory', name)
        if not os.path.exists(path):
            mkdir(path)
            logger.info('Create folder: ' + path+'.')
        return path
        
    
    def write_yaml(self, path: str, data: dict):
        """
        写入yaml文件。
        
        Args:
            path (str): 文件路径。
            data (dict): 数据。
        """
        
        with open(path, 'w') as f:
            yaml.dump(data, f)
            
    def read_yaml(self, path: str)-> dict:
        """
        读取yaml文件。
        
        Args:
            path (str): 文件路径。
        Returns:
            dict: 数据。
        """
        
        with open(path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            
        return data
        
    @property
    def root_path(self):
        return self._root_path
    
    @property
    def cache_path(self):
        return self._cache_path
    
    @property
    def data_unit_path(self):
        return self._data_unit_path
    
    @property
    def history_model_path(self):
        return self._history_model_path
    
    @property
    def log_path(self):
        return self._log_path
            
