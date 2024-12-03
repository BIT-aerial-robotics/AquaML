from AquaML import logger
import os
import numpy as np
from AquaML.recorder.RecorderBase import RecorderBase

class AquaTool:
    def __init__(self):
        
        self._convert_numpy_fn = None
        self._save_weights_fn = None
        self._load_weights_fn = None
        self.recorder:RecorderBase = None
        

    def set_convert_numpy_fn(self, engine:str):
        """
        设置convert_numpy_fn。
        
        Args:
            engine (str): 计算引擎。
        """
        if engine == 'tensorflow':
            def convert_numpy_fn(x):
                if isinstance(x, np.ndarray):
                    return x
                elif isinstance(x, np.float32):
                    return x
                return x.numpy()
            self.convert_numpy_fn = convert_numpy_fn
        if engine == 'torch':
            self.convert_numpy_fn = lambda x: x.cpu().numpy()
        logger.info('convert_numpy_fn set for engine:' + engine)
        
    def set_save_weights_fn(self, engine:str):
        """
        设置save_weights_fn。
        用于存储模型参数。
        
        Args:
            engine (str): 计算引擎。
        """
        
        if engine == 'tensorflow':
            def save_weights_fn(model, name, path):
                file = os.path.join(path, name+'.h5')
                model.save_weights(file)
        elif engine == 'torch':
            import torch
            def save_weights_fn(model, name, path):
                file = os.path.join(path, name+'.pth')
                torch.save(model.state_dict(), file)
        logger.info('save_weights_fn set for engine:' + engine)
                
        self._save_weights_fn = save_weights_fn
        
    def set_load_weights_fn(self, engine:str):
        """
        设置load_weights_fn。
        用于加载模型参数。
        
        Args:
            engine (str): 计算引擎。
        """
        
        if engine == 'tensorflow':
            def load_weights_fn(model, name, path):
                file = os.path.join(path, name+'.h5')
                model.load_weights(file)
                logger.success('model {} loaded from {}'.format(name, file))
        elif engine == 'torch':
            import torch
            def load_weights_fn(model, name, path):
                file = os.path.join(path, name+'.pth')
                model.load_state_dict(torch.load(file))
                logger.success('model {} loaded from {}'.format(name, file))
        logger.info('load_weights_fn set for engine:' + engine)
                
        self._load_weights_fn = load_weights_fn
                
    #########################################
    # 接口
    #########################################
    def convert_numpy_fn(self, x)->np.ndarray:
        """
        将数据转换为numpy格式。
        
        Args:
            x: 待转换数据。
        """
        return self._convert_numpy_fn(x)
    
    def save_weights_fn(self, model, name, path):
        """
        保存模型参数。
        
        Args:
            model: 待保存模型。
            name (str): 模型名称。
            path (str): 保存路径。
        """
        return self._save_weights_fn(model, name, path)
    
    def load_weights_fn(self, model, name, path):
        """
        加载模型参数。
        
        Args:
            model: 待加载模型。
            name (str): 模型名称。
            path (str): 加载路径。
        """
        return self._load_weights_fn(model, name, path)