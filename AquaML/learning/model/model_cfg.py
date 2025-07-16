from torch import concat
from AquaML.config.configclass import configclass
from dataclasses import MISSING
from AquaML.data import unitCfg



@configclass
class ModelCfg:
    
    
    device: str = 'auto' 
    """
    模型的设备，可以为'auto', 'cpu', 'cuda:0'等
    """
    
    inputs_name: "list[str]" = MISSING # type: ignore
    """
    模型的输入名称，可以为'input_1', 'input_2'等
    """
    
    outputs_info: "list[unitCfg]" = MISSING # type: ignore
    """
    模型的输出信息，可以为'output_1', 'output_2'等
    """

    concat_dict: bool = False
    """
    是否将输入的多个tensor拼接成一个tensor
    """
    