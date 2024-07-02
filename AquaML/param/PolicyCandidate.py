from AquaML.param.old.ParamBase import ParamBase


class PEXParams(ParamBase):
    
    def __init__(self,
                 inv_temperature: int,
                 algo: str,
                 eps: float = 0,
                 ):
        """
        PEX算法参数。

        Args:
            inv_temperature (int): 逆温度。
            eps (float): 随机选择action的概率。当eps=0时，完全按照Q值选择action。当eps=1时，完全随机选择action。
        """
        super().__init__()
        
        self.inv_temperature = inv_temperature
        self.algo = algo
        self.eps = eps
        
        self._param_dict['inv_temperature'] = self.inv_temperature
        self._param_dict['eps'] = self.eps
        self._param_dict['algo'] = self.algo

        