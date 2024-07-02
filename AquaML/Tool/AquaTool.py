from AquaML import logger

class AquaTool:
    def __init__(self):
        
        self.convert_numpy_fn = lambda x: x
        

    def set_convert_numpy_fn(self, engine:str):
        """
        设置convert_numpy_fn。
        
        Args:
            engine (str): 计算引擎。
        """
        if engine == 'tensorflow':
            self.convert_numpy_fn = lambda x: x.numpy()
        if engine == 'torch':
            self.convert_numpy_fn = lambda x: x.cpu().numpy()
        logger.info('convert_numpy_fn set: ' + engine)