import abc
import yaml

class ParamBase(abc.ABC):
    """
    参数基类，用于定义参数的基本功能。
    """
    
    def __init__(self,):
        '''
        
        算法参数，运行参数等等的基类。
        '''
        
        self._param_dict = {} # 参数字典
        
    def save_param_yaml(self, path:str):
        '''
        保存参数
        
        Args:
            path (str): 保存路径
        '''
        with open(path, 'w') as f:
            yaml.dump(self._param_dict, f)


